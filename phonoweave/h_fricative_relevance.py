from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .audio import AudioReadError, AudioSegment, consonant_segment, slice_segment
from .fh_fricative import (
    _Candidate,
    _family,
    _oto_set_name,
    _resolve_candidates,
    _role,
    analyze_fh_fricative,
)
from .mandarin import collect_observations, structure_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


_DIRECTIONS = (("rounded", "other"), ("other", "rounded"))


@dataclass(frozen=True)
class HFricativeRelevanceSample:
    oto_set: str
    family: str
    final: str
    alias: str
    entry: OtoEntry
    segment_key: tuple[Path, float, float]
    body: AudioSegment
    core: AudioSegment
    late: AudioSegment


@dataclass(frozen=True)
class HFricativeTargetPenalty:
    oto_set: str
    target_family: str
    substitution_family: str
    alias: str
    final: str
    control_boundary_spectral: float
    substitution_boundary_spectral: float
    boundary_spectral_delta: float
    control_body_spectral: float
    substitution_body_spectral: float
    body_spectral_delta: float


@dataclass(frozen=True)
class HFricativeOtoSetResult:
    oto_set: str
    targets: int
    mean_boundary_spectral_delta: float
    mean_body_spectral_delta: float


@dataclass(frozen=True)
class HFricativeRelevanceComparison:
    target_family: str
    substitution_family: str
    targets: int
    mean_boundary_spectral_delta: float | None
    boundary_spectral_p: float | None
    boundary_spectral_p_holm: float | None
    mean_body_spectral_delta: float | None
    body_spectral_p: float | None
    oto_sets: tuple[HFricativeOtoSetResult, ...]
    target_penalties: tuple[HFricativeTargetPenalty, ...]


@dataclass(frozen=True)
class HFricativePairSummary:
    both_boundary_positive: bool
    both_boundary_holm_significant: bool
    all_oto_sets_boundary_positive: bool
    split_supported_under_proxy: bool


@dataclass(frozen=True)
class HFricativeRelevanceResult:
    acoustic_gate_passed: bool
    acoustic_p: float | None
    samples: int
    skipped: int
    duplicate_observations_removed: int
    ambiguous_segments_removed: int
    ambiguous_observations_removed: int
    comparisons: tuple[HFricativeRelevanceComparison, ...]
    pair: HFricativePairSummary | None


def _segment_key(entry: OtoEntry) -> tuple[Path, float, float]:
    return (
        entry.wav_path,
        round(entry.offset, 3),
        round(entry.offset + entry.preutterance, 3),
    )


def _log_spectrum(samples: np.ndarray, sample_rate: int, grid: np.ndarray) -> np.ndarray:
    values = samples.astype(np.float64, copy=False)
    values = values - np.mean(values)
    if len(values) < 32:
        raise AudioReadError("spectral window is too short")
    power = np.abs(np.fft.rfft(values * np.hanning(len(values)))) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(values), d=1.0 / sample_rate)
    interp = np.interp(grid, freqs, power)
    interp = interp / (np.sum(interp) + 1e-18)
    return np.log10(interp + 1e-12)


def _spectral_distance(
    left: np.ndarray,
    left_rate: int,
    right: np.ndarray,
    right_rate: int,
) -> float:
    upper = min(10000.0, left_rate * 0.45, right_rate * 0.45)
    if upper <= 500.0:
        raise AudioReadError("sample rate is too low for h relevance analysis")
    grid = np.linspace(300.0, upper, 160)
    left_spectrum = _log_spectrum(left, left_rate, grid)
    right_spectrum = _log_spectrum(right, right_rate, grid)
    return float(np.sqrt(np.mean((left_spectrum - right_spectrum) ** 2)))


def _edge(segment: AudioSegment, side: str, window_ms: float = 22.0) -> np.ndarray:
    count = max(64, int(round(segment.sample_rate * window_ms / 1000.0)))
    count = min(count, len(segment.samples))
    if count < 64:
        raise AudioReadError("h boundary window is too short")
    return segment.samples[:count] if side == "start" else segment.samples[-count:]


def h_boundary_penalty(donor_core: AudioSegment, target_late: AudioSegment) -> float:
    return _spectral_distance(
        _edge(donor_core, "end"),
        donor_core.sample_rate,
        _edge(target_late, "start"),
        target_late.sample_rate,
    )


def h_body_penalty(left: AudioSegment, right: AudioSegment) -> float:
    return _spectral_distance(
        left.samples,
        left.sample_rate,
        right.samples,
        right.sample_rate,
    )


def _paired_permutation_p(
    deltas: np.ndarray,
    permutations: int = 10000,
    seed: int = 83011,
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
) -> tuple[
    dict[str, dict[str, list[HFricativeRelevanceSample]]],
    int,
    int,
    int,
    int,
]:
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)

    candidates: list[_Candidate] = []
    for observation in observations:
        structure = structure_for(observation)
        if structure.onset != "h":
            continue
        candidates.append(
            _Candidate(
                oto_set=_oto_set_name(root, observation.entry),
                family=_family(structure.final),
                final=structure.final,
                role=_role(observation.entry.alias, affixes),
                observation=observation,
            )
        )

    resolved, duplicate_removed, ambiguous_segments, ambiguous_observations = (
        _resolve_candidates(candidates)
    )
    grouped: dict[str, dict[str, list[HFricativeRelevanceSample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    skipped = 0
    for candidate in resolved:
        if candidate.role != "internal":
            continue
        try:
            whole = consonant_segment(candidate.observation.entry)
            body = slice_segment(whole, 0.20, 0.90)
            core = slice_segment(whole, 0.20, 0.55)
            late = slice_segment(whole, 0.60, 0.90)
        except (AudioReadError, ValueError):
            skipped += 1
            continue
        entry = candidate.observation.entry
        sample = HFricativeRelevanceSample(
            oto_set=candidate.oto_set,
            family=candidate.family,
            final=candidate.final,
            alias=entry.alias,
            entry=entry,
            segment_key=_segment_key(entry),
            body=body,
            core=core,
            late=late,
        )
        grouped[sample.oto_set][sample.family].append(sample)

    return grouped, skipped, duplicate_removed, ambiguous_segments, ambiguous_observations


def _median_scores(
    donors: list[HFricativeRelevanceSample],
    target: HFricativeRelevanceSample,
) -> tuple[float, float]:
    boundary = [h_boundary_penalty(donor.core, target.late) for donor in donors]
    body = [h_body_penalty(donor.body, target.body) for donor in donors]
    return float(np.median(boundary)), float(np.median(body))


def _oto_summary(
    oto_set: str,
    penalties: list[HFricativeTargetPenalty],
) -> HFricativeOtoSetResult:
    return HFricativeOtoSetResult(
        oto_set=oto_set,
        targets=len(penalties),
        mean_boundary_spectral_delta=float(
            np.mean([item.boundary_spectral_delta for item in penalties])
        ),
        mean_body_spectral_delta=float(
            np.mean([item.body_spectral_delta for item in penalties])
        ),
    )


def _comparison(
    grouped: dict[str, dict[str, list[HFricativeRelevanceSample]]],
    target_family: str,
    substitution_family: str,
    seed: int,
) -> HFricativeRelevanceComparison:
    penalties: list[HFricativeTargetPenalty] = []
    oto_results: list[HFricativeOtoSetResult] = []

    for oto_set in sorted(grouped):
        targets = grouped[oto_set].get(target_family, [])
        substitutes = grouped[oto_set].get(substitution_family, [])
        bank: list[HFricativeTargetPenalty] = []
        if len(substitutes) < 2:
            continue

        for target in targets:
            controls = [
                donor
                for donor in targets
                if donor.segment_key != target.segment_key
            ]
            if len(controls) < 2:
                continue
            try:
                control_boundary, control_body = _median_scores(controls, target)
                substitution_boundary, substitution_body = _median_scores(substitutes, target)
            except (AudioReadError, ValueError):
                continue

            item = HFricativeTargetPenalty(
                oto_set=oto_set,
                target_family=target_family,
                substitution_family=substitution_family,
                alias=target.alias,
                final=target.final,
                control_boundary_spectral=control_boundary,
                substitution_boundary_spectral=substitution_boundary,
                boundary_spectral_delta=substitution_boundary - control_boundary,
                control_body_spectral=control_body,
                substitution_body_spectral=substitution_body,
                body_spectral_delta=substitution_body - control_body,
            )
            bank.append(item)
            penalties.append(item)

        if bank:
            oto_results.append(_oto_summary(oto_set, bank))

    if not penalties:
        return HFricativeRelevanceComparison(
            target_family=target_family,
            substitution_family=substitution_family,
            targets=0,
            mean_boundary_spectral_delta=None,
            boundary_spectral_p=None,
            boundary_spectral_p_holm=None,
            mean_body_spectral_delta=None,
            body_spectral_p=None,
            oto_sets=tuple(oto_results),
            target_penalties=(),
        )

    boundary = np.array(
        [item.boundary_spectral_delta for item in penalties],
        dtype=np.float64,
    )
    body = np.array(
        [item.body_spectral_delta for item in penalties],
        dtype=np.float64,
    )
    return HFricativeRelevanceComparison(
        target_family=target_family,
        substitution_family=substitution_family,
        targets=len(penalties),
        mean_boundary_spectral_delta=float(np.mean(boundary)),
        boundary_spectral_p=_paired_permutation_p(boundary, seed=seed),
        boundary_spectral_p_holm=None,
        mean_body_spectral_delta=float(np.mean(body)),
        body_spectral_p=_paired_permutation_p(body, seed=seed + 100),
        oto_sets=tuple(oto_results),
        target_penalties=tuple(penalties),
    )


def _pair_summary(
    comparisons: tuple[HFricativeRelevanceComparison, ...],
) -> HFricativePairSummary:
    lookup = {
        (item.target_family, item.substitution_family): item
        for item in comparisons
    }
    forward = lookup[("rounded", "other")]
    reverse = lookup[("other", "rounded")]
    both_positive = (
        forward.mean_boundary_spectral_delta is not None
        and reverse.mean_boundary_spectral_delta is not None
        and forward.mean_boundary_spectral_delta > 0
        and reverse.mean_boundary_spectral_delta > 0
    )
    both_significant = (
        forward.boundary_spectral_p_holm is not None
        and reverse.boundary_spectral_p_holm is not None
        and forward.boundary_spectral_p_holm < 0.05
        and reverse.boundary_spectral_p_holm < 0.05
    )
    all_oto_sets = bool(forward.oto_sets and reverse.oto_sets) and all(
        item.mean_boundary_spectral_delta > 0
        for item in (*forward.oto_sets, *reverse.oto_sets)
    )
    return HFricativePairSummary(
        both_boundary_positive=both_positive,
        both_boundary_holm_significant=both_significant,
        all_oto_sets_boundary_positive=all_oto_sets,
        split_supported_under_proxy=both_positive and both_significant and all_oto_sets,
    )


def h_fricative_relevance_test(root: Path) -> HFricativeRelevanceResult:
    root = root.expanduser().resolve()
    acoustic = analyze_fh_fricative(root, "h")
    acoustic_p = acoustic.stratified_permutation_p
    gate = acoustic_p is not None and acoustic_p < 0.05
    if not gate:
        return HFricativeRelevanceResult(
            acoustic_gate_passed=False,
            acoustic_p=acoustic_p,
            samples=0,
            skipped=0,
            duplicate_observations_removed=0,
            ambiguous_segments_removed=0,
            ambiguous_observations_removed=0,
            comparisons=(),
            pair=None,
        )

    grouped, skipped, duplicate_removed, ambiguous_segments, ambiguous_observations = (
        _build_samples(root)
    )
    comparisons = tuple(
        _comparison(grouped, target, substitute, seed=83011 + index * 4000)
        for index, (target, substitute) in enumerate(_DIRECTIONS)
    )
    adjusted = _holm_adjust([item.boundary_spectral_p for item in comparisons])
    comparisons = tuple(
        replace(item, boundary_spectral_p_holm=adjusted[index])
        for index, item in enumerate(comparisons)
    )
    samples = sum(
        len(rows)
        for by_family in grouped.values()
        for rows in by_family.values()
    )
    return HFricativeRelevanceResult(
        acoustic_gate_passed=True,
        acoustic_p=acoustic_p,
        samples=samples,
        skipped=skipped,
        duplicate_observations_removed=duplicate_removed,
        ambiguous_segments_removed=ambiguous_segments,
        ambiguous_observations_removed=ambiguous_observations,
        comparisons=comparisons,
        pair=_pair_summary(comparisons),
    )
