from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, AudioSegment, consonant_segment, slice_segment
from .mandarin import collect_observations, context_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


_CONTEXTS = ("plain", "front", "rounded")
_COMPARISONS = (
    ("front", "plain"),
    ("front", "rounded"),
    ("plain", "rounded"),
    ("rounded", "plain"),
)


@dataclass(frozen=True)
class RhoticSpliceSample:
    subbank: str
    context: str
    alias: str
    final: str
    wav_path: Path
    core: AudioSegment
    late: AudioSegment


@dataclass(frozen=True)
class RhoticTargetPenalty:
    subbank: str
    target_context: str
    substitution_context: str
    alias: str
    final: str
    control_penalty: float
    substitution_penalty: float
    delta: float
    relative_delta: float


@dataclass(frozen=True)
class RhoticComparisonSubbankResult:
    subbank: str
    targets: int
    mean_control_penalty: float
    mean_substitution_penalty: float
    mean_delta: float
    mean_relative_delta: float
    permutation_p: float


@dataclass(frozen=True)
class RhoticComparisonResult:
    target_context: str
    substitution_context: str
    targets: int
    mean_delta: float | None
    mean_relative_delta: float | None
    permutation_p: float | None
    subbanks: list[RhoticComparisonSubbankResult]
    target_penalties: list[RhoticTargetPenalty]


@dataclass(frozen=True)
class RhoticRelevanceResult:
    samples: int
    skipped: int
    comparisons: list[RhoticComparisonResult]


def _subbank_name(root: Path, entry: OtoEntry) -> str:
    directory = entry.oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def _edge_features(segment: AudioSegment, side: str, window_ms: float = 25.0) -> tuple[np.ndarray, float, float]:
    count = max(64, int(round(segment.sample_rate * window_ms / 1000.0)))
    count = min(count, len(segment.samples))
    if count < 64:
        raise AudioReadError("edge window is too short")

    samples = segment.samples[:count] if side == "start" else segment.samples[-count:]
    samples = samples.astype(np.float64, copy=False)
    samples = samples - np.mean(samples)
    rms = float(np.sqrt(np.mean(samples ** 2)) + 1e-12)

    window = np.hanning(len(samples))
    power = np.abs(np.fft.rfft(samples * window)) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / segment.sample_rate)
    upper = min(5000.0, segment.sample_rate * 0.45)
    if upper <= 500.0:
        raise AudioReadError("sample rate is too low for rhotic boundary analysis")
    grid = np.linspace(200.0, upper, 128)
    interp = np.interp(grid, freqs, power)
    interp = interp / (np.sum(interp) + 1e-18)

    min_lag = max(1, int(segment.sample_rate / 500.0))
    max_lag = min(len(samples) - 2, int(segment.sample_rate / 70.0))
    periodicity = 0.0
    for lag in range(min_lag, max_lag + 1):
        left = samples[:-lag]
        right = samples[lag:]
        denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
        if denom > 1e-12:
            periodicity = max(periodicity, float(np.dot(left, right) / denom))
    periodicity = max(0.0, min(periodicity, 1.0))
    return np.log10(interp + 1e-12), rms, periodicity


def rhotic_boundary_penalty(left: AudioSegment, right: AudioSegment) -> float:
    left_spectrum, left_rms, left_periodicity = _edge_features(left, "end")
    right_spectrum, right_rms, right_periodicity = _edge_features(right, "start")
    spectral = float(np.sqrt(np.mean((left_spectrum - right_spectrum) ** 2)))
    energy_db = abs(20.0 * np.log10(left_rms / right_rms))
    periodicity = abs(left_periodicity - right_periodicity)
    return spectral + 0.02 * float(energy_db) + 0.35 * periodicity


def _paired_permutation_p(deltas: np.ndarray, permutations: int = 10000, seed: int = 9917) -> float:
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


def _build_samples(root: Path) -> tuple[dict[str, dict[str, list[RhoticSpliceSample]]], int]:
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)
    grouped: dict[str, dict[str, list[RhoticSpliceSample]]] = defaultdict(lambda: defaultdict(list))
    skipped = 0

    for observation in observations:
        if observation.base_unit != "r":
            continue
        context = context_for("r", observation.final)
        if context not in _CONTEXTS:
            continue
        try:
            whole = consonant_segment(observation.entry, edge_trim=0.04)
            core = slice_segment(whole, 0.25, 0.62)
            late = slice_segment(whole, 0.62, 0.94)
        except (AudioReadError, ValueError):
            skipped += 1
            continue
        sample = RhoticSpliceSample(
            subbank=_subbank_name(root, observation.entry),
            context=context,
            alias=observation.entry.alias,
            final=observation.final,
            wav_path=observation.entry.wav_path,
            core=core,
            late=late,
        )
        grouped[sample.subbank][context].append(sample)
    return grouped, skipped


def _comparison(
    grouped: dict[str, dict[str, list[RhoticSpliceSample]]],
    target_context: str,
    substitution_context: str,
    seed: int,
) -> RhoticComparisonResult:
    penalties: list[RhoticTargetPenalty] = []
    subbanks: list[RhoticComparisonSubbankResult] = []

    for subbank_index, subbank in enumerate(sorted(grouped)):
        targets = grouped[subbank].get(target_context, [])
        substitutes = grouped[subbank].get(substitution_context, [])
        bank_penalties: list[RhoticTargetPenalty] = []
        if not substitutes:
            continue

        for target in targets:
            controls = [
                donor
                for donor in targets
                if donor.wav_path != target.wav_path or donor.alias != target.alias
            ]
            if not controls:
                continue
            try:
                control_scores = np.array(
                    [rhotic_boundary_penalty(donor.core, target.late) for donor in controls],
                    dtype=np.float64,
                )
                substitution_scores = np.array(
                    [rhotic_boundary_penalty(donor.core, target.late) for donor in substitutes],
                    dtype=np.float64,
                )
            except (AudioReadError, ValueError):
                continue
            control = float(np.median(control_scores))
            substitution = float(np.median(substitution_scores))
            delta = substitution - control
            item = RhoticTargetPenalty(
                subbank=subbank,
                target_context=target_context,
                substitution_context=substitution_context,
                alias=target.alias,
                final=target.final,
                control_penalty=control,
                substitution_penalty=substitution,
                delta=delta,
                relative_delta=delta / max(control, 1e-9),
            )
            bank_penalties.append(item)
            penalties.append(item)

        if bank_penalties:
            deltas = np.array([item.delta for item in bank_penalties], dtype=np.float64)
            subbanks.append(
                RhoticComparisonSubbankResult(
                    subbank=subbank,
                    targets=len(bank_penalties),
                    mean_control_penalty=float(np.mean([item.control_penalty for item in bank_penalties])),
                    mean_substitution_penalty=float(np.mean([item.substitution_penalty for item in bank_penalties])),
                    mean_delta=float(np.mean(deltas)),
                    mean_relative_delta=float(np.mean([item.relative_delta for item in bank_penalties])),
                    permutation_p=_paired_permutation_p(deltas, seed=seed + subbank_index),
                )
            )

    if penalties:
        deltas = np.array([item.delta for item in penalties], dtype=np.float64)
        mean_delta = float(np.mean(deltas))
        mean_relative_delta = float(np.mean([item.relative_delta for item in penalties]))
        permutation_p = _paired_permutation_p(deltas, seed=seed + 100)
    else:
        mean_delta = None
        mean_relative_delta = None
        permutation_p = None

    return RhoticComparisonResult(
        target_context=target_context,
        substitution_context=substitution_context,
        targets=len(penalties),
        mean_delta=mean_delta,
        mean_relative_delta=mean_relative_delta,
        permutation_p=permutation_p,
        subbanks=subbanks,
        target_penalties=penalties,
    )


def rhotic_relevance_test(root: Path) -> RhoticRelevanceResult:
    root = root.expanduser().resolve()
    grouped, skipped = _build_samples(root)
    comparisons = [
        _comparison(grouped, target, substitute, 9917 + index * 1000)
        for index, (target, substitute) in enumerate(_COMPARISONS)
    ]
    samples = sum(len(rows) for bank in grouped.values() for rows in bank.values())
    return RhoticRelevanceResult(samples=samples, skipped=skipped, comparisons=comparisons)
