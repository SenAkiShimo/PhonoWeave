from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .audio import AudioReadError, AudioSegment, read_wav
from .mandarin import collect_observations, normalize_alias, structure_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .stop import (
    StopCandidate,
    _detect_release,
    _detect_vowel_onset,
    _oto_set_name,
    _resolve_candidates,
)
from .stop_context import analyze_stop_context


_FAMILIES = ("i_series", "u_series", "other")
_SUPPORTED = {"b", "p", "d", "t", "g", "k"}


@dataclass(frozen=True)
class StopRelevanceSample:
    oto_set: str
    family: str
    final: str
    alias: str
    entry: OtoEntry
    segment_key: tuple[Path, float, float]
    release_to_vowel_ms: float
    burst: AudioSegment
    onset_early: AudioSegment
    onset_late: AudioSegment


@dataclass(frozen=True)
class StopTargetPenalty:
    oto_set: str
    target_family: str
    substitution_family: str
    alias: str
    final: str
    onset_spectral_delta: float
    burst_spectral_delta: float
    timing_delta_ms: float


@dataclass(frozen=True)
class StopOtoSetResult:
    oto_set: str
    targets: int
    mean_onset_spectral_delta: float
    mean_burst_spectral_delta: float
    mean_timing_delta_ms: float


@dataclass(frozen=True)
class StopComparisonResult:
    target_family: str
    substitution_family: str
    targets: int
    mean_onset_spectral_delta: float | None
    onset_spectral_p: float | None
    onset_spectral_p_holm: float | None
    mean_burst_spectral_delta: float | None
    burst_spectral_p: float | None
    mean_timing_delta_ms: float | None
    timing_p: float | None
    oto_sets: tuple[StopOtoSetResult, ...]
    target_penalties: tuple[StopTargetPenalty, ...]


@dataclass(frozen=True)
class StopPairSummary:
    left: str
    right: str
    both_onset_positive: bool
    both_onset_holm_significant: bool
    all_oto_sets_onset_positive: bool


@dataclass(frozen=True)
class StopRelevanceResult:
    base_unit: str
    samples: int
    skipped: int
    duplicate_observations_removed: int
    ambiguous_segments_removed: int
    ambiguous_observations_removed: int
    acoustic_supported_pairs: tuple[tuple[str, str], ...]
    comparisons: tuple[StopComparisonResult, ...]
    pairs: tuple[StopPairSummary, ...]


def _role(alias: str, affixes: set[tuple[str, str]]) -> str:
    normalized = normalize_alias(alias, affixes).strip()
    return "initial" if normalized.startswith("-") else "internal"


def _family(final: str) -> str:
    if final.startswith("i"):
        return "i_series"
    if final.startswith("u"):
        return "u_series"
    return "other"


def _segment_key(entry: OtoEntry) -> tuple[Path, float, float]:
    return (
        entry.wav_path,
        round(entry.offset, 3),
        round(entry.offset + entry.preutterance, 3),
    )


def _slice_absolute(
    samples: np.ndarray,
    sample_rate: int,
    start_ms: float,
    end_ms: float,
) -> AudioSegment:
    start_ms = max(0.0, start_ms)
    start = max(0, int(round(start_ms * sample_rate / 1000.0)))
    end = min(len(samples), int(round(end_ms * sample_rate / 1000.0)))
    if end - start < 64:
        raise AudioReadError("stop relevance slice is too short")
    return AudioSegment(
        samples=samples[start:end],
        sample_rate=sample_rate,
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _segment_sample(entry: OtoEntry) -> tuple[float, AudioSegment, AudioSegment, AudioSegment]:
    if entry.preutterance <= 0:
        raise AudioReadError("preutterance is not positive")
    samples, sample_rate = read_wav(entry.wav_path)
    start_ms = max(0.0, entry.offset)
    prior_ms = entry.offset + entry.preutterance
    if prior_ms - start_ms < 16.0:
        raise AudioReadError("stop window is too short")

    release_ms, _ = _detect_release(samples, sample_rate, start_ms, prior_ms)
    vowel_onset_ms, _ = _detect_vowel_onset(
        samples,
        sample_rate,
        prior_ms,
        release_ms,
    )
    burst = _slice_absolute(
        samples,
        sample_rate,
        max(start_ms, release_ms - 2.0),
        release_ms + 5.0,
    )
    onset_early = _slice_absolute(
        samples,
        sample_rate,
        vowel_onset_ms,
        vowel_onset_ms + 12.0,
    )
    onset_late = _slice_absolute(
        samples,
        sample_rate,
        vowel_onset_ms + 12.0,
        vowel_onset_ms + 28.0,
    )
    return (
        max(0.0, vowel_onset_ms - release_ms),
        burst,
        onset_early,
        onset_late,
    )


def _spectrum(samples: np.ndarray, sample_rate: int, grid: np.ndarray) -> np.ndarray:
    samples = samples.astype(np.float64, copy=False)
    samples = samples - np.mean(samples)
    power = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    interp = np.interp(grid, freqs, power)
    interp = interp / (np.sum(interp) + 1e-18)
    return np.log10(interp + 1e-12)


def _spectral_distance(left: AudioSegment, right: AudioSegment) -> float:
    upper = min(8000.0, left.sample_rate * 0.45, right.sample_rate * 0.45)
    if upper <= 800.0:
        raise AudioReadError("sample rate is too low for stop relevance analysis")
    grid = np.linspace(300.0, upper, 160)
    return float(
        np.sqrt(
            np.mean(
                (
                    _spectrum(left.samples, left.sample_rate, grid)
                    - _spectrum(right.samples, right.sample_rate, grid)
                ) ** 2
            )
        )
    )


def _onset_penalty(donor: StopRelevanceSample, target: StopRelevanceSample) -> float:
    early = _spectral_distance(donor.onset_early, target.onset_early)
    late = _spectral_distance(donor.onset_late, target.onset_late)
    return float((early + late) / 2.0)


def _median_scores(
    donors: list[StopRelevanceSample],
    target: StopRelevanceSample,
) -> tuple[float, float, float]:
    onset: list[float] = []
    burst: list[float] = []
    timing: list[float] = []
    for donor in donors:
        onset.append(_onset_penalty(donor, target))
        burst.append(_spectral_distance(donor.burst, target.burst))
        timing.append(abs(donor.release_to_vowel_ms - target.release_to_vowel_ms))
    return (
        float(np.median(onset)),
        float(np.median(burst)),
        float(np.median(timing)),
    )


def _paired_permutation_p(
    deltas: np.ndarray,
    permutations: int = 10000,
    seed: int = 65011,
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


def _acoustic_supported_pairs(root: Path, base_unit: str) -> tuple[tuple[str, str], ...]:
    acoustic = analyze_stop_context(root, base_unit)
    return tuple(
        (item.left, item.right)
        for item in acoustic.pairwise
        if item.stratified_p_holm is not None and item.stratified_p_holm < 0.05
    )


def _build_samples(
    root: Path,
    base_unit: str,
) -> tuple[
    dict[str, dict[str, list[StopRelevanceSample]]],
    int,
    int,
    int,
    int,
]:
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)

    candidates: list[StopCandidate] = []
    for observation in observations:
        structure = structure_for(observation)
        if structure.onset != base_unit:
            continue
        candidates.append(
            StopCandidate(
                observation=observation,
                final=structure.final,
                role=_role(observation.entry.alias, affixes),
                oto_set=_oto_set_name(root, observation.entry),
            )
        )

    resolved, duplicate_removed, ambiguous_segments, ambiguous_observations = (
        _resolve_candidates(candidates)
    )
    grouped: dict[str, dict[str, list[StopRelevanceSample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    skipped = 0
    for candidate in resolved:
        if candidate.role != "internal":
            continue
        try:
            release_to_vowel_ms, burst, onset_early, onset_late = _segment_sample(
                candidate.observation.entry
            )
        except (AudioReadError, ValueError, FloatingPointError):
            skipped += 1
            continue
        family = _family(candidate.final)
        sample = StopRelevanceSample(
            oto_set=candidate.oto_set,
            family=family,
            final=candidate.final,
            alias=candidate.observation.entry.alias,
            entry=candidate.observation.entry,
            segment_key=_segment_key(candidate.observation.entry),
            release_to_vowel_ms=release_to_vowel_ms,
            burst=burst,
            onset_early=onset_early,
            onset_late=onset_late,
        )
        grouped[sample.oto_set][sample.family].append(sample)
    return (
        grouped,
        skipped,
        duplicate_removed,
        ambiguous_segments,
        ambiguous_observations,
    )


def _comparison(
    grouped: dict[str, dict[str, list[StopRelevanceSample]]],
    target_family: str,
    substitution_family: str,
    seed: int,
) -> StopComparisonResult:
    penalties: list[StopTargetPenalty] = []
    oto_results: list[StopOtoSetResult] = []
    for oto_set in sorted(grouped):
        targets = grouped[oto_set].get(target_family, [])
        substitutes = grouped[oto_set].get(substitution_family, [])
        if len(substitutes) < 2:
            continue
        bank: list[StopTargetPenalty] = []
        for target in targets:
            controls = [
                donor
                for donor in targets
                if donor.segment_key != target.segment_key
            ]
            if len(controls) < 2:
                continue
            try:
                control = _median_scores(controls, target)
                substitution = _median_scores(substitutes, target)
            except (AudioReadError, ValueError):
                continue
            item = StopTargetPenalty(
                oto_set=oto_set,
                target_family=target_family,
                substitution_family=substitution_family,
                alias=target.alias,
                final=target.final,
                onset_spectral_delta=substitution[0] - control[0],
                burst_spectral_delta=substitution[1] - control[1],
                timing_delta_ms=substitution[2] - control[2],
            )
            bank.append(item)
            penalties.append(item)
        if bank:
            oto_results.append(
                StopOtoSetResult(
                    oto_set=oto_set,
                    targets=len(bank),
                    mean_onset_spectral_delta=float(
                        np.mean([item.onset_spectral_delta for item in bank])
                    ),
                    mean_burst_spectral_delta=float(
                        np.mean([item.burst_spectral_delta for item in bank])
                    ),
                    mean_timing_delta_ms=float(
                        np.mean([item.timing_delta_ms for item in bank])
                    ),
                )
            )

    if not penalties:
        return StopComparisonResult(
            target_family=target_family,
            substitution_family=substitution_family,
            targets=0,
            mean_onset_spectral_delta=None,
            onset_spectral_p=None,
            onset_spectral_p_holm=None,
            mean_burst_spectral_delta=None,
            burst_spectral_p=None,
            mean_timing_delta_ms=None,
            timing_p=None,
            oto_sets=tuple(oto_results),
            target_penalties=(),
        )

    def values(name: str) -> np.ndarray:
        return np.array([getattr(item, name) for item in penalties], dtype=np.float64)

    onset = values("onset_spectral_delta")
    burst = values("burst_spectral_delta")
    timing = values("timing_delta_ms")
    return StopComparisonResult(
        target_family=target_family,
        substitution_family=substitution_family,
        targets=len(penalties),
        mean_onset_spectral_delta=float(np.mean(onset)),
        onset_spectral_p=_paired_permutation_p(onset, seed=seed),
        onset_spectral_p_holm=None,
        mean_burst_spectral_delta=float(np.mean(burst)),
        burst_spectral_p=_paired_permutation_p(burst, seed=seed + 100),
        mean_timing_delta_ms=float(np.mean(timing)),
        timing_p=_paired_permutation_p(timing, seed=seed + 200),
        oto_sets=tuple(oto_results),
        target_penalties=tuple(penalties),
    )


def _pair_summaries(
    supported_pairs: tuple[tuple[str, str], ...],
    comparisons: tuple[StopComparisonResult, ...],
) -> tuple[StopPairSummary, ...]:
    lookup = {
        (item.target_family, item.substitution_family): item
        for item in comparisons
    }
    results: list[StopPairSummary] = []
    for left, right in supported_pairs:
        forward = lookup[(left, right)]
        reverse = lookup[(right, left)]
        both_positive = (
            forward.mean_onset_spectral_delta is not None
            and reverse.mean_onset_spectral_delta is not None
            and forward.mean_onset_spectral_delta > 0.0
            and reverse.mean_onset_spectral_delta > 0.0
        )
        both_sig = (
            forward.onset_spectral_p_holm is not None
            and reverse.onset_spectral_p_holm is not None
            and forward.onset_spectral_p_holm < 0.05
            and reverse.onset_spectral_p_holm < 0.05
        )
        all_oto = bool(forward.oto_sets and reverse.oto_sets) and all(
            item.mean_onset_spectral_delta > 0.0
            for item in (*forward.oto_sets, *reverse.oto_sets)
        )
        results.append(
            StopPairSummary(
                left=left,
                right=right,
                both_onset_positive=both_positive,
                both_onset_holm_significant=both_sig,
                all_oto_sets_onset_positive=all_oto,
            )
        )
    return tuple(results)


def stop_relevance_test(root: Path, base_unit: str) -> StopRelevanceResult:
    if base_unit not in _SUPPORTED:
        raise ValueError("stop relevance supports b, p, d, t, g, k")
    root = root.expanduser().resolve()
    supported_pairs = _acoustic_supported_pairs(root, base_unit)
    grouped, skipped, duplicate_removed, ambiguous_segments, ambiguous_observations = (
        _build_samples(root, base_unit)
    )

    directed = tuple(
        direction
        for left, right in supported_pairs
        for direction in ((left, right), (right, left))
    )
    comparisons = tuple(
        _comparison(
            grouped,
            target,
            substitute,
            seed=65011 + index * 2000 + sum(ord(char) for char in base_unit),
        )
        for index, (target, substitute) in enumerate(directed)
    )
    adjusted = _holm_adjust([item.onset_spectral_p for item in comparisons])
    comparisons = tuple(
        replace(item, onset_spectral_p_holm=adjusted[index])
        for index, item in enumerate(comparisons)
    )
    samples = sum(
        len(rows)
        for by_family in grouped.values()
        for rows in by_family.values()
    )
    return StopRelevanceResult(
        base_unit=base_unit,
        samples=samples,
        skipped=skipped,
        duplicate_observations_removed=duplicate_removed,
        ambiguous_segments_removed=ambiguous_segments,
        ambiguous_observations_removed=ambiguous_observations,
        acoustic_supported_pairs=supported_pairs,
        comparisons=comparisons,
        pairs=_pair_summaries(supported_pairs, comparisons),
    )
