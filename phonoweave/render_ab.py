from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import re
import wave

import numpy as np

from .audio import AudioReadError, AudioSegment, consonant_segment, read_wav, slice_segment
from .mandarin import collect_observations, context_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .splice import boundary_penalty, splice_relevance_test


@dataclass(frozen=True)
class RenderSample:
    entry: OtoEntry
    subbank: str
    context: str
    final: str
    alias: str
    core: AudioSegment
    late: AudioSegment


@dataclass(frozen=True)
class RenderedPair:
    subbank: str
    target_alias: str
    rounded_donor_alias: str
    plain_donor_alias: str
    delta: float
    natural_path: Path
    a_path: Path
    b_path: Path


@dataclass(frozen=True)
class RenderResult:
    output_dir: Path
    pairs: list[RenderedPair]


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return cleaned or "sample"


def _subbank_name(root: Path, entry: OtoEntry) -> str:
    directory = entry.oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def _resample_to(samples: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("invalid target length")
    if len(samples) == length:
        return samples.astype(np.float64, copy=True)
    if len(samples) < 2:
        value = float(samples[0]) if len(samples) else 0.0
        return np.full(length, value, dtype=np.float64)
    source_x = np.linspace(0.0, 1.0, len(samples), endpoint=True)
    target_x = np.linspace(0.0, 1.0, length, endpoint=True)
    return np.interp(target_x, source_x, samples).astype(np.float64)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)) + 1e-12)


def _match_rms(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    gain = _rms(reference) / _rms(samples)
    gain = min(max(gain, 0.25), 4.0)
    return samples * gain


def _crossfade(left: np.ndarray, right: np.ndarray, sample_rate: int, ms: float = 5.0) -> np.ndarray:
    count = int(round(sample_rate * ms / 1000.0))
    count = min(count, len(left), len(right))
    if count < 2:
        return np.concatenate([left, right])

    fade = np.linspace(0.0, 1.0, count, endpoint=True)
    mixed = left[-count:] * (1.0 - fade) + right[:count] * fade
    return np.concatenate([left[:-count], mixed, right[count:]])


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak > 0.98:
        samples = samples * (0.98 / peak)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def _tail(entry: OtoEntry, start_ms: float, extra_ms: float = 220.0) -> tuple[np.ndarray, int]:
    samples, sample_rate = read_wav(entry.wav_path)
    end_ms = min(
        1000.0 * len(samples) / sample_rate,
        entry.offset + entry.preutterance + extra_ms,
    )
    start = max(0, int(round(start_ms * sample_rate / 1000.0)))
    end = min(len(samples), int(round(end_ms * sample_rate / 1000.0)))
    if end <= start:
        raise AudioReadError("target tail is empty")
    return samples[start:end], sample_rate


def _build_samples(root: Path, base_unit: str) -> dict[str, dict[str, list[RenderSample]]]:
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)
    grouped: dict[str, dict[str, list[RenderSample]]] = defaultdict(lambda: defaultdict(list))

    for observation in observations:
        if observation.base_unit != base_unit:
            continue
        context = context_for(observation.base_unit, observation.final)
        if context not in {"plain", "rounded"}:
            continue
        try:
            whole = consonant_segment(observation.entry)
            core = slice_segment(whole, 0.20, 0.60)
            late = slice_segment(whole, 0.60, 0.92)
        except (AudioReadError, ValueError):
            continue
        sample = RenderSample(
            entry=observation.entry,
            subbank=_subbank_name(root, observation.entry),
            context=context,
            final=observation.final,
            alias=observation.entry.alias,
            core=core,
            late=late,
        )
        grouped[sample.subbank][context].append(sample)
    return grouped


def _closest_to(values: list[tuple[RenderSample, float]], target: float) -> RenderSample:
    return min(values, key=lambda item: abs(item[1] - target))[0]


def _representative_targets(
    root: Path,
    base_unit: str,
    per_subbank: int,
) -> dict[str, list[tuple[str, str, float]]]:
    relevance = splice_relevance_test(root, base_unit)
    grouped: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for item in relevance.target_penalties:
        grouped[item.subbank].append((item.alias, item.final, item.delta))

    selected: dict[str, list[tuple[str, str, float]]] = {}
    for subbank, rows in grouped.items():
        median = float(np.median([row[2] for row in rows]))
        selected[subbank] = sorted(rows, key=lambda row: abs(row[2] - median))[:per_subbank]
    return selected


def _render_audio(core: np.ndarray, tail: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = _crossfade(core, tail, sample_rate)
    silence = np.zeros(int(round(sample_rate * 0.08)), dtype=np.float64)
    return np.concatenate([silence, audio, silence])


def render_ab_pairs(
    root: Path,
    output_dir: Path,
    base_unit: str = "sh",
    per_subbank: int = 3,
) -> RenderResult:
    root = root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    grouped = _build_samples(root, base_unit)
    selected = _representative_targets(root, base_unit, per_subbank)
    pairs: list[RenderedPair] = []

    for subbank in sorted(selected):
        plain = grouped[subbank].get("plain", [])
        rounded = grouped[subbank].get("rounded", [])
        if not plain or len(rounded) < 2:
            continue

        for index, (target_alias, target_final, delta) in enumerate(selected[subbank], start=1):
            candidates = [sample for sample in rounded if sample.alias == target_alias and sample.final == target_final]
            if not candidates:
                continue
            target = candidates[0]
            rounded_donors = [
                sample
                for sample in rounded
                if sample.entry.wav_path != target.entry.wav_path or sample.alias != target.alias
            ]
            if not rounded_donors:
                continue

            rounded_scores = [(donor, boundary_penalty(donor.core, target.late)) for donor in rounded_donors]
            plain_scores = [(donor, boundary_penalty(donor.core, target.late)) for donor in plain]
            rounded_median = float(np.median([score for _, score in rounded_scores]))
            plain_median = float(np.median([score for _, score in plain_scores]))
            rounded_donor = _closest_to(rounded_scores, rounded_median)
            plain_donor = _closest_to(plain_scores, plain_median)

            tail, sample_rate = _tail(target.entry, target.late.start_ms)
            target_length = len(target.core.samples)
            natural_core = _resample_to(target.core.samples, target_length)
            rounded_core = _resample_to(rounded_donor.core.samples, target_length)
            plain_core = _resample_to(plain_donor.core.samples, target_length)
            rounded_core = _match_rms(rounded_core, natural_core)
            plain_core = _match_rms(plain_core, natural_core)

            natural_audio = _render_audio(natural_core, tail, sample_rate)
            a_audio = _render_audio(rounded_core, tail, sample_rate)
            b_audio = _render_audio(plain_core, tail, sample_rate)

            stem = f"{_safe_name(subbank)}_{index:02d}_{_safe_name(target_alias)}"
            natural_path = output_dir / f"{stem}_N_natural.wav"
            a_path = output_dir / f"{stem}_A_rounded.wav"
            b_path = output_dir / f"{stem}_B_plain.wav"
            _write_wav(natural_path, natural_audio, sample_rate)
            _write_wav(a_path, a_audio, sample_rate)
            _write_wav(b_path, b_audio, sample_rate)

            pairs.append(
                RenderedPair(
                    subbank=subbank,
                    target_alias=target_alias,
                    rounded_donor_alias=rounded_donor.alias,
                    plain_donor_alias=plain_donor.alias,
                    delta=delta,
                    natural_path=natural_path,
                    a_path=a_path,
                    b_path=b_path,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "subbank",
            "target_alias",
            "rounded_donor",
            "plain_donor",
            "delta",
            "natural_file",
            "a_file",
            "b_file",
        ])
        for pair in pairs:
            writer.writerow([
                pair.subbank,
                pair.target_alias,
                pair.rounded_donor_alias,
                pair.plain_donor_alias,
                f"{pair.delta:.6f}",
                pair.natural_path.name,
                pair.a_path.name,
                pair.b_path.name,
            ])

    return RenderResult(output_dir=output_dir, pairs=pairs)
