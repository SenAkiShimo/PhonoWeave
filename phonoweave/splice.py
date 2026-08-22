from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, AudioSegment, consonant_segment, slice_segment
from .mandarin import collect_observations, context_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


@dataclass(frozen=True)
class SpliceSample:
    subbank: str
    context: str
    final: str
    alias: str
    wav_path: Path
    core: AudioSegment
    late: AudioSegment


@dataclass(frozen=True)
class TargetPenalty:
    subbank: str
    alias: str
    final: str
    natural_penalty: float
    rounded_control_penalty: float
    plain_substitution_penalty: float
    delta: float
    relative_delta: float


@dataclass(frozen=True)
class SubbankSpliceResult:
    subbank: str
    targets: int
    mean_natural_penalty: float
    mean_rounded_control_penalty: float
    mean_plain_substitution_penalty: float
    mean_delta: float
    mean_relative_delta: float
    permutation_p: float


@dataclass(frozen=True)
class SpliceTestResult:
    base_unit: str
    targets: int
    skipped: int
    mean_delta: float | None
    mean_relative_delta: float | None
    permutation_p: float | None
    subbanks: list[SubbankSpliceResult]
    target_penalties: list[TargetPenalty]


def _subbank_name(root: Path, entry: OtoEntry) -> str:
    directory = entry.oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def _edge_spectrum(segment: AudioSegment, side: str, window_ms: float = 20.0) -> tuple[np.ndarray, np.ndarray, float]:
    count = max(32, int(round(segment.sample_rate * window_ms / 1000.0)))
    count = min(count, len(segment.samples))
    if count < 32:
        raise AudioReadError("edge window is too short")

    samples = segment.samples[:count] if side == "start" else segment.samples[-count:]
    samples = samples.astype(np.float64, copy=False)
    samples = samples - np.mean(samples)
    rms = float(np.sqrt(np.mean(samples ** 2)) + 1e-12)

    window = np.hanning(len(samples))
    power = np.abs(np.fft.rfft(samples * window)) ** 2
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / segment.sample_rate)
    return freqs, power, rms


def boundary_penalty(left: AudioSegment, right: AudioSegment) -> float:
    left_freqs, left_power, left_rms = _edge_spectrum(left, "end")
    right_freqs, right_power, right_rms = _edge_spectrum(right, "start")

    upper = min(12000.0, left.sample_rate * 0.47, right.sample_rate * 0.47)
    if upper <= 1200.0:
        raise AudioReadError("sample rate is too low for fricative boundary analysis")

    grid = np.linspace(1000.0, upper, 128)
    left_interp = np.interp(grid, left_freqs, left_power)
    right_interp = np.interp(grid, right_freqs, right_power)

    left_norm = left_interp / (np.sum(left_interp) + 1e-18)
    right_norm = right_interp / (np.sum(right_interp) + 1e-18)
    left_log = np.log10(left_norm + 1e-12)
    right_log = np.log10(right_norm + 1e-12)
    spectral = float(np.sqrt(np.mean((left_log - right_log) ** 2)))

    energy_db = abs(20.0 * np.log10(left_rms / right_rms))
    return spectral + 0.03 * float(energy_db)


def _paired_permutation_p(deltas: np.ndarray, permutations: int = 10000, seed: int = 7319) -> float:
    if len(deltas) == 0:
        return 1.0
    observed = float(np.mean(deltas))
    if observed <= 0:
        return 1.0

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(deltas))
        if float(np.mean(deltas * signs)) >= observed:
            exceed += 1
    return float((exceed + 1) / (permutations + 1))


def _target_penalty(target: SpliceSample, plain_donors: list[SpliceSample], rounded_donors: list[SpliceSample]) -> TargetPenalty:
    natural = boundary_penalty(target.core, target.late)
    plain_scores = np.array([boundary_penalty(donor.core, target.late) for donor in plain_donors], dtype=np.float64)
    rounded_scores = np.array([boundary_penalty(donor.core, target.late) for donor in rounded_donors], dtype=np.float64)

    plain_penalty = float(np.median(plain_scores))
    rounded_penalty = float(np.median(rounded_scores))
    delta = plain_penalty - rounded_penalty
    relative_delta = delta / max(rounded_penalty, 1e-9)
    return TargetPenalty(
        subbank=target.subbank,
        alias=target.alias,
        final=target.final,
        natural_penalty=natural,
        rounded_control_penalty=rounded_penalty,
        plain_substitution_penalty=plain_penalty,
        delta=delta,
        relative_delta=relative_delta,
    )


def _summarize_subbank(subbank: str, penalties: list[TargetPenalty], seed: int) -> SubbankSpliceResult:
    deltas = np.array([item.delta for item in penalties], dtype=np.float64)
    return SubbankSpliceResult(
        subbank=subbank,
        targets=len(penalties),
        mean_natural_penalty=float(np.mean([item.natural_penalty for item in penalties])),
        mean_rounded_control_penalty=float(np.mean([item.rounded_control_penalty for item in penalties])),
        mean_plain_substitution_penalty=float(np.mean([item.plain_substitution_penalty for item in penalties])),
        mean_delta=float(np.mean(deltas)),
        mean_relative_delta=float(np.mean([item.relative_delta for item in penalties])),
        permutation_p=_paired_permutation_p(deltas, seed=seed),
    )


def splice_relevance_test(root: Path, base_unit: str = "sh") -> SpliceTestResult:
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)

    grouped: dict[str, dict[str, list[SpliceSample]]] = defaultdict(lambda: defaultdict(list))
    skipped = 0

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
            skipped += 1
            continue

        sample = SpliceSample(
            subbank=_subbank_name(root, observation.entry),
            context=context,
            final=observation.final,
            alias=observation.entry.alias,
            wav_path=observation.entry.wav_path,
            core=core,
            late=late,
        )
        grouped[sample.subbank][context].append(sample)

    penalties: list[TargetPenalty] = []
    subbanks: list[SubbankSpliceResult] = []

    for index, subbank in enumerate(sorted(grouped)):
        plain = grouped[subbank].get("plain", [])
        rounded = grouped[subbank].get("rounded", [])
        if not plain or len(rounded) < 2:
            continue

        bank_penalties: list[TargetPenalty] = []
        for target in rounded:
            rounded_donors = [donor for donor in rounded if donor.wav_path != target.wav_path or donor.alias != target.alias]
            if not rounded_donors:
                continue
            try:
                item = _target_penalty(target, plain, rounded_donors)
            except (AudioReadError, ValueError):
                skipped += 1
                continue
            bank_penalties.append(item)
            penalties.append(item)

        if bank_penalties:
            subbanks.append(_summarize_subbank(subbank, bank_penalties, seed=7319 + index))

    if penalties:
        deltas = np.array([item.delta for item in penalties], dtype=np.float64)
        mean_delta = float(np.mean(deltas))
        mean_relative_delta = float(np.mean([item.relative_delta for item in penalties]))
        permutation_p = _paired_permutation_p(deltas, seed=9127)
    else:
        mean_delta = None
        mean_relative_delta = None
        permutation_p = None

    return SpliceTestResult(
        base_unit=base_unit,
        targets=len(penalties),
        skipped=skipped,
        mean_delta=mean_delta,
        mean_relative_delta=mean_relative_delta,
        permutation_p=permutation_p,
        subbanks=subbanks,
        target_penalties=penalties,
    )
