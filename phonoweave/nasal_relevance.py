from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
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
class NasalRelevanceSample:
    oto_set: str
    role: str
    family: str
    final: str
    alias: str
    entry: OtoEntry
    segment_key: tuple[Path, float, float]
    body: AudioSegment
    core: AudioSegment
    late: AudioSegment


@dataclass(frozen=True)
class NasalTargetPenalty:
    oto_set: str
    target_family: str
    substitution_family: str
    alias: str
    final: str
    body_spectral_delta: float
    body_periodicity_delta: float
    body_balance_delta: float
    transition_spectral_delta: float
    transition_periodicity_delta: float
    transition_balance_delta: float


@dataclass(frozen=True)
class NasalOtoSetResult:
    oto_set: str
    targets: int
    mean_body_spectral_delta: float
    mean_transition_spectral_delta: float
    mean_transition_periodicity_delta: float
    mean_transition_balance_delta: float


@dataclass(frozen=True)
class NasalComparisonResult:
    target_family: str
    substitution_family: str
    targets: int
    mean_body_spectral_delta: float | None
    body_spectral_p: float | None
    body_spectral_p_holm: float | None
    mean_body_periodicity_delta: float | None
    body_periodicity_p: float | None
    mean_body_balance_delta: float | None
    body_balance_p: float | None
    mean_transition_spectral_delta: float | None
    transition_spectral_p: float | None
    transition_spectral_p_holm: float | None
    mean_transition_periodicity_delta: float | None
    transition_periodicity_p: float | None
    mean_transition_balance_delta: float | None
    transition_balance_p: float | None
    oto_sets: tuple[NasalOtoSetResult, ...]
    target_penalties: tuple[NasalTargetPenalty, ...]


@dataclass(frozen=True)
class NasalPairSummary:
    left: str
    right: str
    both_transition_positive: bool
    both_transition_holm_significant: bool
    both_body_positive: bool
    both_body_holm_significant: bool
    all_oto_sets_transition_positive: bool


@dataclass(frozen=True)
class NasalRelevanceResult:
    base_unit: str
    samples: int
    skipped: int
    duplicate_observations_removed: int
    ambiguous_segments_removed: int
    comparisons: tuple[NasalComparisonResult, ...]
    pairs: tuple[NasalPairSummary, ...]


@dataclass(frozen=True)
class _Candidate:
    oto_set: str
    role: str
    family: str
    final: str
    alias: str
    entry: OtoEntry


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


def _resolve_candidates(
    candidates: list[_Candidate],
) -> tuple[list[_Candidate], int, int]:
    by_segment: dict[tuple[Path, float, float], list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_segment[_segment_key(candidate.entry)].append(candidate)

    resolved: list[_Candidate] = []
    duplicate_observations_removed = 0
    ambiguous_segments_removed = 0
    for rows in by_segment.values():
        if len(rows) == 1:
            resolved.append(rows[0])
            continue
        labels = {(row.role, row.family, row.final) for row in rows}
        if len(labels) != 1:
            ambiguous_segments_removed += 1
            duplicate_observations_removed += len(rows)
            continue
        resolved.append(rows[0])
        duplicate_observations_removed += len(rows) - 1
    return resolved, duplicate_observations_removed, ambiguous_segments_removed


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


def _spectrum(samples: np.ndarray, sample_rate: int, grid: np.ndarray) -> np.ndarray:
    samples = samples.astype(np.float64, copy=False)
    samples = samples - np.mean(samples)
    window = np.hanning(len(samples))
    power = np.abs(np.fft.rfft(samples * window)) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    interp = np.interp(grid, freqs, power)
    interp = interp / (np.sum(interp) + 1e-18)
    return np.log10(interp + 1e-12)


def _balance(samples: np.ndarray, sample_rate: int) -> float:
    samples = samples.astype(np.float64, copy=False)
    samples = samples - np.mean(samples)
    power = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    low = float(np.sum(power[(freqs >= 100.0) & (freqs < 600.0)])) + 1e-18
    mid = float(np.sum(power[(freqs >= 600.0) & (freqs < 2000.0)])) + 1e-18
    return float(10.0 * np.log10(low / mid))


def _body_penalty(left: AudioSegment, right: AudioSegment) -> tuple[float, float, float]:
    upper = min(4000.0, left.sample_rate * 0.45, right.sample_rate * 0.45)
    if upper <= 600.0:
        raise AudioReadError("sample rate is too low for nasal body analysis")
    grid = np.linspace(100.0, upper, 128)
    spectral = float(
        np.sqrt(
            np.mean(
                (_spectrum(left.samples, left.sample_rate, grid)
                 - _spectrum(right.samples, right.sample_rate, grid)) ** 2
            )
        )
    )
    periodicity = abs(
        _periodicity(left.samples, left.sample_rate)
        - _periodicity(right.samples, right.sample_rate)
    )
    balance = abs(
        _balance(left.samples, left.sample_rate)
        - _balance(right.samples, right.sample_rate)
    )
    return spectral, periodicity, balance


def _edge(segment: AudioSegment, side: str, window_ms: float = 25.0) -> np.ndarray:
    count = max(64, int(round(segment.sample_rate * window_ms / 1000.0)))
    count = min(count, len(segment.samples))
    if count < 64:
        raise AudioReadError("nasal transition edge is too short")
    return segment.samples[:count] if side == "start" else segment.samples[-count:]


def _transition_penalty(
    donor_core: AudioSegment,
    target_late: AudioSegment,
) -> tuple[float, float, float]:
    left = _edge(donor_core, "end")
    right = _edge(target_late, "start")
    upper = min(4000.0, donor_core.sample_rate * 0.45, target_late.sample_rate * 0.45)
    if upper <= 600.0:
        raise AudioReadError("sample rate is too low for nasal transition analysis")
    grid = np.linspace(100.0, upper, 128)
    spectral = float(
        np.sqrt(
            np.mean(
                (_spectrum(left, donor_core.sample_rate, grid)
                 - _spectrum(right, target_late.sample_rate, grid)) ** 2
            )
        )
    )
    periodicity = abs(
        _periodicity(left, donor_core.sample_rate)
        - _periodicity(right, target_late.sample_rate)
    )
    balance = abs(
        _balance(left, donor_core.sample_rate)
        - _balance(right, target_late.sample_rate)
    )
    return spectral, periodicity, balance


def _paired_permutation_p(
    deltas: np.ndarray,
    permutations: int = 10000,
    seed: int = 57011,
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
    base_unit: str,
) -> tuple[dict[str, dict[str, list[NasalRelevanceSample]]], int, int, int]:
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)

    candidates: list[_Candidate] = []
    for observation in observations:
        structure = structure_for(observation)
        if structure.onset != base_unit:
            continue
        family = _family(structure.final)
        if family is None:
            continue
        role = _alias_role(observation.entry.alias, affixes)
        if role != "internal":
            continue
        candidates.append(
            _Candidate(
                oto_set=_oto_set_name(root, observation.entry),
                role=role,
                family=family,
                final=structure.final,
                alias=observation.entry.alias,
                entry=observation.entry,
            )
        )

    resolved, duplicate_removed, ambiguous_removed = _resolve_candidates(candidates)
    grouped: dict[str, dict[str, list[NasalRelevanceSample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    skipped = 0
    for candidate in resolved:
        try:
            whole = consonant_segment(candidate.entry, edge_trim=0.04)
            body = slice_segment(whole, 0.18, 0.82)
            core = slice_segment(whole, 0.18, 0.52)
            late = slice_segment(whole, 0.58, 0.92)
        except (AudioReadError, ValueError):
            skipped += 1
            continue
        sample = NasalRelevanceSample(
            oto_set=candidate.oto_set,
            role=candidate.role,
            family=candidate.family,
            final=candidate.final,
            alias=candidate.alias,
            entry=candidate.entry,
            segment_key=_segment_key(candidate.entry),
            body=body,
            core=core,
            late=late,
        )
        grouped[sample.oto_set][sample.family].append(sample)

    return grouped, skipped, duplicate_removed, ambiguous_removed


def _median_scores(
    donors: list[NasalRelevanceSample],
    target: NasalRelevanceSample,
) -> tuple[float, float, float, float, float, float]:
    body_spectral: list[float] = []
    body_periodicity: list[float] = []
    body_balance: list[float] = []
    transition_spectral: list[float] = []
    transition_periodicity: list[float] = []
    transition_balance: list[float] = []
    for donor in donors:
        bs, bp, bb = _body_penalty(donor.body, target.body)
        ts, tp, tb = _transition_penalty(donor.core, target.late)
        body_spectral.append(bs)
        body_periodicity.append(bp)
        body_balance.append(bb)
        transition_spectral.append(ts)
        transition_periodicity.append(tp)
        transition_balance.append(tb)
    return (
        float(np.median(body_spectral)),
        float(np.median(body_periodicity)),
        float(np.median(body_balance)),
        float(np.median(transition_spectral)),
        float(np.median(transition_periodicity)),
        float(np.median(transition_balance)),
    )


def _comparison(
    grouped: dict[str, dict[str, list[NasalRelevanceSample]]],
    target_family: str,
    substitution_family: str,
    seed: int,
) -> NasalComparisonResult:
    penalties: list[NasalTargetPenalty] = []
    oto_results: list[NasalOtoSetResult] = []

    for oto_set in sorted(grouped):
        targets = grouped[oto_set].get(target_family, [])
        substitutes = grouped[oto_set].get(substitution_family, [])
        bank: list[NasalTargetPenalty] = []
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
                control = _median_scores(controls, target)
                substitution = _median_scores(substitutes, target)
            except (AudioReadError, ValueError):
                continue
            item = NasalTargetPenalty(
                oto_set=oto_set,
                target_family=target_family,
                substitution_family=substitution_family,
                alias=target.alias,
                final=target.final,
                body_spectral_delta=substitution[0] - control[0],
                body_periodicity_delta=substitution[1] - control[1],
                body_balance_delta=substitution[2] - control[2],
                transition_spectral_delta=substitution[3] - control[3],
                transition_periodicity_delta=substitution[4] - control[4],
                transition_balance_delta=substitution[5] - control[5],
            )
            bank.append(item)
            penalties.append(item)
        if bank:
            oto_results.append(
                NasalOtoSetResult(
                    oto_set=oto_set,
                    targets=len(bank),
                    mean_body_spectral_delta=float(
                        np.mean([item.body_spectral_delta for item in bank])
                    ),
                    mean_transition_spectral_delta=float(
                        np.mean([item.transition_spectral_delta for item in bank])
                    ),
                    mean_transition_periodicity_delta=float(
                        np.mean([item.transition_periodicity_delta for item in bank])
                    ),
                    mean_transition_balance_delta=float(
                        np.mean([item.transition_balance_delta for item in bank])
                    ),
                )
            )

    if not penalties:
        return NasalComparisonResult(
            target_family=target_family,
            substitution_family=substitution_family,
            targets=0,
            mean_body_spectral_delta=None,
            body_spectral_p=None,
            body_spectral_p_holm=None,
            mean_body_periodicity_delta=None,
            body_periodicity_p=None,
            mean_body_balance_delta=None,
            body_balance_p=None,
            mean_transition_spectral_delta=None,
            transition_spectral_p=None,
            transition_spectral_p_holm=None,
            mean_transition_periodicity_delta=None,
            transition_periodicity_p=None,
            mean_transition_balance_delta=None,
            transition_balance_p=None,
            oto_sets=tuple(oto_results),
            target_penalties=(),
        )

    def values(name: str) -> np.ndarray:
        return np.array([getattr(item, name) for item in penalties], dtype=np.float64)

    body_spectral = values("body_spectral_delta")
    body_periodicity = values("body_periodicity_delta")
    body_balance = values("body_balance_delta")
    transition_spectral = values("transition_spectral_delta")
    transition_periodicity = values("transition_periodicity_delta")
    transition_balance = values("transition_balance_delta")

    return NasalComparisonResult(
        target_family=target_family,
        substitution_family=substitution_family,
        targets=len(penalties),
        mean_body_spectral_delta=float(np.mean(body_spectral)),
        body_spectral_p=_paired_permutation_p(body_spectral, seed=seed),
        body_spectral_p_holm=None,
        mean_body_periodicity_delta=float(np.mean(body_periodicity)),
        body_periodicity_p=_paired_permutation_p(body_periodicity, seed=seed + 100),
        mean_body_balance_delta=float(np.mean(body_balance)),
        body_balance_p=_paired_permutation_p(body_balance, seed=seed + 200),
        mean_transition_spectral_delta=float(np.mean(transition_spectral)),
        transition_spectral_p=_paired_permutation_p(transition_spectral, seed=seed + 300),
        transition_spectral_p_holm=None,
        mean_transition_periodicity_delta=float(np.mean(transition_periodicity)),
        transition_periodicity_p=_paired_permutation_p(transition_periodicity, seed=seed + 400),
        mean_transition_balance_delta=float(np.mean(transition_balance)),
        transition_balance_p=_paired_permutation_p(transition_balance, seed=seed + 500),
        oto_sets=tuple(oto_results),
        target_penalties=tuple(penalties),
    )


def _pair_summaries(
    comparisons: tuple[NasalComparisonResult, ...],
) -> tuple[NasalPairSummary, ...]:
    lookup = {
        (item.target_family, item.substitution_family): item
        for item in comparisons
    }
    results: list[NasalPairSummary] = []
    for left, right in combinations(_FAMILIES, 2):
        forward = lookup[(left, right)]
        reverse = lookup[(right, left)]
        transition_positive = (
            forward.mean_transition_spectral_delta is not None
            and reverse.mean_transition_spectral_delta is not None
            and forward.mean_transition_spectral_delta > 0
            and reverse.mean_transition_spectral_delta > 0
        )
        transition_sig = (
            forward.transition_spectral_p_holm is not None
            and reverse.transition_spectral_p_holm is not None
            and forward.transition_spectral_p_holm < 0.05
            and reverse.transition_spectral_p_holm < 0.05
        )
        body_positive = (
            forward.mean_body_spectral_delta is not None
            and reverse.mean_body_spectral_delta is not None
            and forward.mean_body_spectral_delta > 0
            and reverse.mean_body_spectral_delta > 0
        )
        body_sig = (
            forward.body_spectral_p_holm is not None
            and reverse.body_spectral_p_holm is not None
            and forward.body_spectral_p_holm < 0.05
            and reverse.body_spectral_p_holm < 0.05
        )
        all_oto_sets = bool(forward.oto_sets and reverse.oto_sets) and all(
            item.mean_transition_spectral_delta > 0
            for item in (*forward.oto_sets, *reverse.oto_sets)
        )
        results.append(
            NasalPairSummary(
                left=left,
                right=right,
                both_transition_positive=transition_positive,
                both_transition_holm_significant=transition_sig,
                both_body_positive=body_positive,
                both_body_holm_significant=body_sig,
                all_oto_sets_transition_positive=all_oto_sets,
            )
        )
    return tuple(results)


def nasal_relevance_test(
    root: Path,
    base_unit: str = "n",
) -> NasalRelevanceResult:
    if base_unit not in {"m", "n"}:
        raise ValueError("nasal relevance supports only m and n")
    root = root.expanduser().resolve()
    grouped, skipped, duplicate_removed, ambiguous_removed = _build_samples(
        root,
        base_unit,
    )
    comparisons = tuple(
        _comparison(
            grouped,
            target,
            substitute,
            seed=57011 + index * 2000 + (0 if base_unit == "m" else 20000),
        )
        for index, (target, substitute) in enumerate(_COMPARISONS)
    )
    body_adjusted = _holm_adjust([item.body_spectral_p for item in comparisons])
    transition_adjusted = _holm_adjust(
        [item.transition_spectral_p for item in comparisons]
    )
    comparisons = tuple(
        replace(
            item,
            body_spectral_p_holm=body_adjusted[index],
            transition_spectral_p_holm=transition_adjusted[index],
        )
        for index, item in enumerate(comparisons)
    )
    samples = sum(
        len(rows)
        for by_family in grouped.values()
        for rows in by_family.values()
    )
    return NasalRelevanceResult(
        base_unit=base_unit,
        samples=samples,
        skipped=skipped,
        duplicate_observations_removed=duplicate_removed,
        ambiguous_segments_removed=ambiguous_removed,
        comparisons=comparisons,
        pairs=_pair_summaries(comparisons),
    )
