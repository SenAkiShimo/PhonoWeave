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
    body: AudioSegment
    core: AudioSegment
    late: AudioSegment


@dataclass(frozen=True)
class RhoticTargetPenalty:
    subbank: str
    target_context: str
    substitution_context: str
    alias: str
    final: str
    control_body_spectral: float
    substitution_body_spectral: float
    body_spectral_delta: float
    control_body_periodicity: float
    substitution_body_periodicity: float
    body_periodicity_delta: float
    control_boundary: float
    substitution_boundary: float
    boundary_delta: float


@dataclass(frozen=True)
class RhoticComparisonSubbankResult:
    subbank: str
    targets: int
    mean_control_body_spectral: float
    mean_substitution_body_spectral: float
    mean_body_spectral_delta: float
    body_spectral_p: float
    mean_control_body_periodicity: float
    mean_substitution_body_periodicity: float
    mean_body_periodicity_delta: float
    body_periodicity_p: float
    mean_control_boundary: float
    mean_substitution_boundary: float
    mean_boundary_delta: float
    boundary_p: float


@dataclass(frozen=True)
class RhoticComparisonResult:
    target_context: str
    substitution_context: str
    targets: int
    mean_body_spectral_delta: float | None
    body_spectral_p: float | None
    mean_body_periodicity_delta: float | None
    body_periodicity_p: float | None
    mean_boundary_delta: float | None
    boundary_p: float | None
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


def _periodicity(samples: np.ndarray, sample_rate: int) -> float:
    samples = samples.astype(np.float64, copy=False)
    samples = samples - np.mean(samples)
    min_lag = max(1, int(sample_rate / 500.0))
    max_lag = min(len(samples) - 2, int(sample_rate / 70.0))
    if max_lag <= min_lag:
        return 0.0

    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        left = samples[:-lag]
        right = samples[lag:]
        denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
        if denom > 1e-12:
            best = max(best, float(np.dot(left, right) / denom))
    return max(0.0, min(best, 1.0))


def _log_spectrum(samples: np.ndarray, sample_rate: int, grid: np.ndarray) -> np.ndarray:
    samples = samples.astype(np.float64, copy=False)
    samples = samples - np.mean(samples)
    window = np.hanning(len(samples))
    power = np.abs(np.fft.rfft(samples * window)) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    interp = np.interp(grid, freqs, power)
    interp = interp / (np.sum(interp) + 1e-18)
    return np.log10(interp + 1e-12)


def _body_windows(segment: AudioSegment, parts: int = 3) -> list[np.ndarray]:
    samples = segment.samples
    if len(samples) < parts * 64:
        return [samples]
    edges = np.linspace(0, len(samples), parts + 1, dtype=int)
    return [samples[edges[index]:edges[index + 1]] for index in range(parts)]


def rhotic_body_penalty(left: AudioSegment, right: AudioSegment) -> tuple[float, float]:
    upper = min(5000.0, left.sample_rate * 0.45, right.sample_rate * 0.45)
    if upper <= 500.0:
        raise AudioReadError("sample rate is too low for rhotic body analysis")
    grid = np.linspace(200.0, upper, 128)

    left_windows = _body_windows(left)
    right_windows = _body_windows(right)
    parts = min(len(left_windows), len(right_windows))
    if parts == 0:
        raise AudioReadError("rhotic body is empty")

    spectral_scores = []
    for index in range(parts):
        left_spectrum = _log_spectrum(left_windows[index], left.sample_rate, grid)
        right_spectrum = _log_spectrum(right_windows[index], right.sample_rate, grid)
        spectral_scores.append(float(np.sqrt(np.mean((left_spectrum - right_spectrum) ** 2))))

    spectral = float(np.mean(spectral_scores))
    periodicity = abs(
        _periodicity(left.samples, left.sample_rate)
        - _periodicity(right.samples, right.sample_rate)
    )
    return spectral, periodicity


def _edge_features(segment: AudioSegment, side: str, window_ms: float = 25.0) -> tuple[np.ndarray, float, float]:
    count = max(64, int(round(segment.sample_rate * window_ms / 1000.0)))
    count = min(count, len(segment.samples))
    if count < 64:
        raise AudioReadError("edge window is too short")

    samples = segment.samples[:count] if side == "start" else segment.samples[-count:]
    samples = samples.astype(np.float64, copy=False)
    samples = samples - np.mean(samples)
    rms = float(np.sqrt(np.mean(samples ** 2)) + 1e-12)

    upper = min(5000.0, segment.sample_rate * 0.45)
    if upper <= 500.0:
        raise AudioReadError("sample rate is too low for rhotic boundary analysis")
    grid = np.linspace(200.0, upper, 128)
    spectrum = _log_spectrum(samples, segment.sample_rate, grid)
    periodicity = _periodicity(samples, segment.sample_rate)
    return spectrum, rms, periodicity


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
            body = slice_segment(whole, 0.18, 0.82)
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
            body=body,
            core=core,
            late=late,
        )
        grouped[sample.subbank][context].append(sample)
    return grouped, skipped


def _median_scores(
    donors: list[RhoticSpliceSample],
    target: RhoticSpliceSample,
) -> tuple[float, float, float]:
    body_spectral = []
    body_periodicity = []
    boundary = []
    for donor in donors:
        spectral, periodicity = rhotic_body_penalty(donor.body, target.body)
        body_spectral.append(spectral)
        body_periodicity.append(periodicity)
        boundary.append(rhotic_boundary_penalty(donor.core, target.late))
    return (
        float(np.median(body_spectral)),
        float(np.median(body_periodicity)),
        float(np.median(boundary)),
    )


def _summarize_subbank(
    subbank: str,
    penalties: list[RhoticTargetPenalty],
    seed: int,
) -> RhoticComparisonSubbankResult:
    body_spectral_deltas = np.array([item.body_spectral_delta for item in penalties], dtype=np.float64)
    body_periodicity_deltas = np.array([item.body_periodicity_delta for item in penalties], dtype=np.float64)
    boundary_deltas = np.array([item.boundary_delta for item in penalties], dtype=np.float64)
    return RhoticComparisonSubbankResult(
        subbank=subbank,
        targets=len(penalties),
        mean_control_body_spectral=float(np.mean([item.control_body_spectral for item in penalties])),
        mean_substitution_body_spectral=float(np.mean([item.substitution_body_spectral for item in penalties])),
        mean_body_spectral_delta=float(np.mean(body_spectral_deltas)),
        body_spectral_p=_paired_permutation_p(body_spectral_deltas, seed=seed),
        mean_control_body_periodicity=float(np.mean([item.control_body_periodicity for item in penalties])),
        mean_substitution_body_periodicity=float(np.mean([item.substitution_body_periodicity for item in penalties])),
        mean_body_periodicity_delta=float(np.mean(body_periodicity_deltas)),
        body_periodicity_p=_paired_permutation_p(body_periodicity_deltas, seed=seed + 100),
        mean_control_boundary=float(np.mean([item.control_boundary for item in penalties])),
        mean_substitution_boundary=float(np.mean([item.substitution_boundary for item in penalties])),
        mean_boundary_delta=float(np.mean(boundary_deltas)),
        boundary_p=_paired_permutation_p(boundary_deltas, seed=seed + 200),
    )


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
                control_body_spectral, control_body_periodicity, control_boundary = _median_scores(controls, target)
                substitution_body_spectral, substitution_body_periodicity, substitution_boundary = _median_scores(
                    substitutes,
                    target,
                )
            except (AudioReadError, ValueError):
                continue

            item = RhoticTargetPenalty(
                subbank=subbank,
                target_context=target_context,
                substitution_context=substitution_context,
                alias=target.alias,
                final=target.final,
                control_body_spectral=control_body_spectral,
                substitution_body_spectral=substitution_body_spectral,
                body_spectral_delta=substitution_body_spectral - control_body_spectral,
                control_body_periodicity=control_body_periodicity,
                substitution_body_periodicity=substitution_body_periodicity,
                body_periodicity_delta=substitution_body_periodicity - control_body_periodicity,
                control_boundary=control_boundary,
                substitution_boundary=substitution_boundary,
                boundary_delta=substitution_boundary - control_boundary,
            )
            bank_penalties.append(item)
            penalties.append(item)

        if bank_penalties:
            subbanks.append(
                _summarize_subbank(
                    subbank,
                    bank_penalties,
                    seed=seed + subbank_index * 1000,
                )
            )

    if penalties:
        body_spectral_deltas = np.array([item.body_spectral_delta for item in penalties], dtype=np.float64)
        body_periodicity_deltas = np.array([item.body_periodicity_delta for item in penalties], dtype=np.float64)
        boundary_deltas = np.array([item.boundary_delta for item in penalties], dtype=np.float64)
        mean_body_spectral_delta = float(np.mean(body_spectral_deltas))
        body_spectral_p = _paired_permutation_p(body_spectral_deltas, seed=seed + 10000)
        mean_body_periodicity_delta = float(np.mean(body_periodicity_deltas))
        body_periodicity_p = _paired_permutation_p(body_periodicity_deltas, seed=seed + 10100)
        mean_boundary_delta = float(np.mean(boundary_deltas))
        boundary_p = _paired_permutation_p(boundary_deltas, seed=seed + 10200)
    else:
        mean_body_spectral_delta = None
        body_spectral_p = None
        mean_body_periodicity_delta = None
        body_periodicity_p = None
        mean_boundary_delta = None
        boundary_p = None

    return RhoticComparisonResult(
        target_context=target_context,
        substitution_context=substitution_context,
        targets=len(penalties),
        mean_body_spectral_delta=mean_body_spectral_delta,
        body_spectral_p=body_spectral_p,
        mean_body_periodicity_delta=mean_body_periodicity_delta,
        body_periodicity_p=body_periodicity_p,
        mean_boundary_delta=mean_boundary_delta,
        boundary_p=boundary_p,
        subbanks=subbanks,
        target_penalties=penalties,
    )


def rhotic_relevance_test(root: Path) -> RhoticRelevanceResult:
    root = root.expanduser().resolve()
    grouped, skipped = _build_samples(root)
    comparisons = [
        _comparison(grouped, target, substitute, 9917 + index * 20000)
        for index, (target, substitute) in enumerate(_COMPARISONS)
    ]
    samples = sum(len(rows) for bank in grouped.values() for rows in bank.values())
    return RhoticRelevanceResult(samples=samples, skipped=skipped, comparisons=comparisons)
