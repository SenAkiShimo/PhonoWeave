from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .audio import AudioReadError, AudioSegment, consonant_segment, slice_segment
from .mandarin import collect_observations, normalize_alias, structure_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


_FAMILIES = ("i_series", "u_series", "other")
_COMPARISONS = tuple(
    (target, substitute)
    for target in _FAMILIES
    for substitute in _FAMILIES
    if target != substitute
)


@dataclass(frozen=True)
class LateralRelevanceSample:
    oto_set: str
    role: str
    family: str
    final: str
    alias: str
    wav_path: Path
    segment_key: tuple[Path, float, float]
    body: AudioSegment
    core: AudioSegment
    late: AudioSegment


@dataclass(frozen=True)
class LateralTargetPenalty:
    oto_set: str
    role: str
    target_family: str
    substitution_family: str
    alias: str
    final: str
    control_body_spectral: float
    substitution_body_spectral: float
    body_spectral_delta: float
    control_body_periodicity: float
    substitution_body_periodicity: float
    body_periodicity_delta: float
    control_boundary_spectral: float
    substitution_boundary_spectral: float
    boundary_spectral_delta: float
    control_boundary_periodicity: float
    substitution_boundary_periodicity: float
    boundary_periodicity_delta: float


@dataclass(frozen=True)
class LateralRelevanceOtoSetResult:
    oto_set: str
    targets: int
    mean_body_spectral_delta: float
    mean_body_periodicity_delta: float
    mean_boundary_spectral_delta: float
    mean_boundary_periodicity_delta: float


@dataclass(frozen=True)
class LateralRelevanceComparison:
    role: str
    target_family: str
    substitution_family: str
    targets: int
    mean_body_spectral_delta: float | None
    body_spectral_p: float | None
    mean_body_periodicity_delta: float | None
    body_periodicity_p: float | None
    mean_boundary_spectral_delta: float | None
    boundary_spectral_p: float | None
    boundary_spectral_p_holm: float | None
    mean_boundary_periodicity_delta: float | None
    boundary_periodicity_p: float | None
    oto_sets: tuple[LateralRelevanceOtoSetResult, ...]
    target_penalties: tuple[LateralTargetPenalty, ...]


@dataclass(frozen=True)
class LateralRelevanceRoleResult:
    role: str
    comparisons: tuple[LateralRelevanceComparison, ...]


@dataclass(frozen=True)
class LateralRelevanceResult:
    samples: int
    skipped: int
    duplicate_segments: int
    roles: tuple[LateralRelevanceRoleResult, ...]


def _oto_set_name(root: Path, entry: OtoEntry) -> str:
    directory = entry.oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def _segment_key(entry: OtoEntry) -> tuple[Path, float, float]:
    return (
        entry.wav_path,
        round(entry.offset, 3),
        round(entry.offset + entry.preutterance, 3),
    )


def _alias_role(alias: str, affixes: set[tuple[str, str]]) -> str:
    normalized = normalize_alias(alias, affixes).strip()
    return "initial" if normalized.startswith("-") else "internal"


def _family(final: str) -> str | None:
    if final.startswith("i"):
        return "i_series"
    if final.startswith("u"):
        return "u_series"
    if final.startswith("v") or final == "ü":
        return None
    return "other"


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


def lateral_body_penalty(left: AudioSegment, right: AudioSegment) -> tuple[float, float]:
    upper = min(5000.0, left.sample_rate * 0.45, right.sample_rate * 0.45)
    if upper <= 500.0:
        raise AudioReadError("sample rate is too low for lateral body analysis")
    grid = np.linspace(200.0, upper, 128)

    left_windows = _body_windows(left)
    right_windows = _body_windows(right)
    parts = min(len(left_windows), len(right_windows))
    if parts == 0:
        raise AudioReadError("lateral body is empty")

    spectral_scores = []
    for index in range(parts):
        left_spectrum = _log_spectrum(left_windows[index], left.sample_rate, grid)
        right_spectrum = _log_spectrum(right_windows[index], right.sample_rate, grid)
        spectral_scores.append(
            float(np.sqrt(np.mean((left_spectrum - right_spectrum) ** 2)))
        )

    spectral = float(np.mean(spectral_scores))
    periodicity = abs(
        _periodicity(left.samples, left.sample_rate)
        - _periodicity(right.samples, right.sample_rate)
    )
    return spectral, periodicity


def _edge_samples(segment: AudioSegment, side: str, window_ms: float = 22.0) -> np.ndarray:
    count = max(64, int(round(segment.sample_rate * window_ms / 1000.0)))
    count = min(count, len(segment.samples))
    if count < 64:
        raise AudioReadError("edge window is too short")
    if side == "start":
        return segment.samples[:count]
    return segment.samples[-count:]


def lateral_boundary_penalty(
    donor_core: AudioSegment,
    target_late: AudioSegment,
) -> tuple[float, float]:
    left = _edge_samples(donor_core, "end")
    right = _edge_samples(target_late, "start")
    upper = min(5000.0, donor_core.sample_rate * 0.45, target_late.sample_rate * 0.45)
    if upper <= 500.0:
        raise AudioReadError("sample rate is too low for lateral boundary analysis")
    grid = np.linspace(200.0, upper, 128)
    left_spectrum = _log_spectrum(left, donor_core.sample_rate, grid)
    right_spectrum = _log_spectrum(right, target_late.sample_rate, grid)
    spectral = float(np.sqrt(np.mean((left_spectrum - right_spectrum) ** 2)))
    periodicity = abs(
        _periodicity(left, donor_core.sample_rate)
        - _periodicity(right, target_late.sample_rate)
    )
    return spectral, periodicity


def _paired_permutation_p(
    deltas: np.ndarray,
    permutations: int = 10000,
    seed: int = 13007,
) -> float:
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


def _holm_adjust(values: list[float | None]) -> list[float | None]:
    valid = [(index, value) for index, value in enumerate(values) if value is not None]
    if not valid:
        return [None] * len(values)
    ordered = sorted(valid, key=lambda item: item[1])
    adjusted: dict[int, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return [adjusted.get(index) for index in range(len(values))]


def _build_samples(
    root: Path,
) -> tuple[dict[str, dict[str, dict[str, list[LateralRelevanceSample]]]], int, int]:
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)
    grouped: dict[
        str,
        dict[str, dict[str, list[LateralRelevanceSample]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    skipped = 0
    duplicate_segments = 0
    seen: set[tuple[Path, float, float]] = set()

    for observation in observations:
        structure = structure_for(observation)
        if structure.onset != "l":
            continue
        family = _family(structure.final)
        if family is None:
            continue
        key = _segment_key(observation.entry)
        if key in seen:
            duplicate_segments += 1
            continue
        seen.add(key)
        try:
            whole = consonant_segment(observation.entry, edge_trim=0.04)
            body = slice_segment(whole, 0.18, 0.82)
            core = slice_segment(whole, 0.18, 0.52)
            late = slice_segment(whole, 0.58, 0.92)
        except (AudioReadError, ValueError):
            skipped += 1
            continue

        sample = LateralRelevanceSample(
            oto_set=_oto_set_name(root, observation.entry),
            role=_alias_role(observation.entry.alias, affixes),
            family=family,
            final=structure.final,
            alias=observation.entry.alias,
            wav_path=observation.entry.wav_path,
            segment_key=key,
            body=body,
            core=core,
            late=late,
        )
        grouped[sample.role][sample.oto_set][sample.family].append(sample)

    return grouped, skipped, duplicate_segments


def _median_scores(
    donors: list[LateralRelevanceSample],
    target: LateralRelevanceSample,
) -> tuple[float, float, float, float]:
    body_spectral: list[float] = []
    body_periodicity: list[float] = []
    boundary_spectral: list[float] = []
    boundary_periodicity: list[float] = []
    for donor in donors:
        spectral, periodicity = lateral_body_penalty(donor.body, target.body)
        edge_spectral, edge_periodicity = lateral_boundary_penalty(
            donor.core,
            target.late,
        )
        body_spectral.append(spectral)
        body_periodicity.append(periodicity)
        boundary_spectral.append(edge_spectral)
        boundary_periodicity.append(edge_periodicity)
    return (
        float(np.median(body_spectral)),
        float(np.median(body_periodicity)),
        float(np.median(boundary_spectral)),
        float(np.median(boundary_periodicity)),
    )


def _oto_set_summary(
    oto_set: str,
    penalties: list[LateralTargetPenalty],
) -> LateralRelevanceOtoSetResult:
    return LateralRelevanceOtoSetResult(
        oto_set=oto_set,
        targets=len(penalties),
        mean_body_spectral_delta=float(
            np.mean([item.body_spectral_delta for item in penalties])
        ),
        mean_body_periodicity_delta=float(
            np.mean([item.body_periodicity_delta for item in penalties])
        ),
        mean_boundary_spectral_delta=float(
            np.mean([item.boundary_spectral_delta for item in penalties])
        ),
        mean_boundary_periodicity_delta=float(
            np.mean([item.boundary_periodicity_delta for item in penalties])
        ),
    )


def _comparison(
    grouped: dict[str, dict[str, list[LateralRelevanceSample]]],
    role: str,
    target_family: str,
    substitution_family: str,
    seed: int,
) -> LateralRelevanceComparison:
    penalties: list[LateralTargetPenalty] = []
    oto_results: list[LateralRelevanceOtoSetResult] = []

    for oto_set in sorted(grouped):
        targets = grouped[oto_set].get(target_family, [])
        substitutes = grouped[oto_set].get(substitution_family, [])
        bank_penalties: list[LateralTargetPenalty] = []
        if not substitutes:
            continue

        for target in targets:
            controls = [
                donor
                for donor in targets
                if donor.segment_key != target.segment_key
            ]
            if not controls:
                continue
            try:
                (
                    control_body_spectral,
                    control_body_periodicity,
                    control_boundary_spectral,
                    control_boundary_periodicity,
                ) = _median_scores(controls, target)
                (
                    substitution_body_spectral,
                    substitution_body_periodicity,
                    substitution_boundary_spectral,
                    substitution_boundary_periodicity,
                ) = _median_scores(substitutes, target)
            except (AudioReadError, ValueError):
                continue

            item = LateralTargetPenalty(
                oto_set=oto_set,
                role=role,
                target_family=target_family,
                substitution_family=substitution_family,
                alias=target.alias,
                final=target.final,
                control_body_spectral=control_body_spectral,
                substitution_body_spectral=substitution_body_spectral,
                body_spectral_delta=substitution_body_spectral - control_body_spectral,
                control_body_periodicity=control_body_periodicity,
                substitution_body_periodicity=substitution_body_periodicity,
                body_periodicity_delta=(
                    substitution_body_periodicity - control_body_periodicity
                ),
                control_boundary_spectral=control_boundary_spectral,
                substitution_boundary_spectral=substitution_boundary_spectral,
                boundary_spectral_delta=(
                    substitution_boundary_spectral - control_boundary_spectral
                ),
                control_boundary_periodicity=control_boundary_periodicity,
                substitution_boundary_periodicity=substitution_boundary_periodicity,
                boundary_periodicity_delta=(
                    substitution_boundary_periodicity
                    - control_boundary_periodicity
                ),
            )
            bank_penalties.append(item)
            penalties.append(item)

        if bank_penalties:
            oto_results.append(_oto_set_summary(oto_set, bank_penalties))

    if not penalties:
        return LateralRelevanceComparison(
            role=role,
            target_family=target_family,
            substitution_family=substitution_family,
            targets=0,
            mean_body_spectral_delta=None,
            body_spectral_p=None,
            mean_body_periodicity_delta=None,
            body_periodicity_p=None,
            mean_boundary_spectral_delta=None,
            boundary_spectral_p=None,
            boundary_spectral_p_holm=None,
            mean_boundary_periodicity_delta=None,
            boundary_periodicity_p=None,
            oto_sets=tuple(oto_results),
            target_penalties=(),
        )

    body_spectral = np.array(
        [item.body_spectral_delta for item in penalties],
        dtype=np.float64,
    )
    body_periodicity = np.array(
        [item.body_periodicity_delta for item in penalties],
        dtype=np.float64,
    )
    boundary_spectral = np.array(
        [item.boundary_spectral_delta for item in penalties],
        dtype=np.float64,
    )
    boundary_periodicity = np.array(
        [item.boundary_periodicity_delta for item in penalties],
        dtype=np.float64,
    )

    return LateralRelevanceComparison(
        role=role,
        target_family=target_family,
        substitution_family=substitution_family,
        targets=len(penalties),
        mean_body_spectral_delta=float(np.mean(body_spectral)),
        body_spectral_p=_paired_permutation_p(body_spectral, seed=seed),
        mean_body_periodicity_delta=float(np.mean(body_periodicity)),
        body_periodicity_p=_paired_permutation_p(
            body_periodicity,
            seed=seed + 100,
        ),
        mean_boundary_spectral_delta=float(np.mean(boundary_spectral)),
        boundary_spectral_p=_paired_permutation_p(
            boundary_spectral,
            seed=seed + 200,
        ),
        boundary_spectral_p_holm=None,
        mean_boundary_periodicity_delta=float(np.mean(boundary_periodicity)),
        boundary_periodicity_p=_paired_permutation_p(
            boundary_periodicity,
            seed=seed + 300,
        ),
        oto_sets=tuple(oto_results),
        target_penalties=tuple(penalties),
    )


def lateral_relevance_test(root: Path) -> LateralRelevanceResult:
    root = root.expanduser().resolve()
    grouped, skipped, duplicate_segments = _build_samples(root)
    role_results: list[LateralRelevanceRoleResult] = []

    for role_index, role in enumerate(sorted(grouped)):
        comparisons = [
            _comparison(
                grouped[role],
                role,
                target,
                substitute,
                seed=23011 + role_index * 20000 + index * 1000,
            )
            for index, (target, substitute) in enumerate(_COMPARISONS)
        ]
        adjusted = _holm_adjust(
            [item.boundary_spectral_p for item in comparisons]
        )
        comparisons = [
            replace(item, boundary_spectral_p_holm=adjusted[index])
            for index, item in enumerate(comparisons)
        ]
        role_results.append(
            LateralRelevanceRoleResult(
                role=role,
                comparisons=tuple(comparisons),
            )
        )

    samples = sum(
        len(rows)
        for by_oto_set in grouped.values()
        for by_family in by_oto_set.values()
        for rows in by_family.values()
    )
    return LateralRelevanceResult(
        samples=samples,
        skipped=skipped,
        duplicate_segments=duplicate_segments,
        roles=tuple(role_results),
    )
