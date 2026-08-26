from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, consonant_segment, slice_segment
from .contrast import balanced_accuracy, nearest_centroid_predict
from .features import extract_fricative_features
from .mandarin import collect_observations, normalize_alias, structure_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


_SUPPORTED = {"f", "h"}
_FAMILIES = ("rounded", "other")
_FEATURES = (
    "core_centroid_hz",
    "core_spread_hz",
    "core_skewness",
    "core_kurtosis",
    "core_slope",
    "core_high_band_ratio",
    "late_centroid_hz",
    "late_spread_hz",
    "late_skewness",
    "late_kurtosis",
    "late_slope",
    "late_high_band_ratio",
)


@dataclass(frozen=True)
class FHFricativeSample:
    oto_set: str
    family: str
    final: str
    alias: str
    entry: OtoEntry
    vector: np.ndarray


@dataclass(frozen=True)
class FHFricativeOtoSetResult:
    oto_set: str
    rounded_count: int
    other_count: int
    distance: float
    effects: dict[str, float]


@dataclass(frozen=True)
class FHFricativeAnalysis:
    base_unit: str
    samples: int
    skipped: int
    duplicate_observations_removed: int
    ambiguous_segments_removed: int
    ambiguous_observations_removed: int
    counts: dict[str, int]
    cross_oto_set_balanced_accuracy: float | None
    cross_by_oto_set: dict[str, float]
    stratified_distance: float | None
    stratified_permutation_p: float | None
    stratified_effects: dict[str, float]
    effect_sign_agreement: dict[str, int]
    oto_sets: tuple[FHFricativeOtoSetResult, ...]


@dataclass(frozen=True)
class _Candidate:
    oto_set: str
    family: str
    final: str
    role: str
    observation: object


def _oto_set_name(root: Path, entry: OtoEntry) -> str:
    directory = entry.oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def _segment_key(entry: OtoEntry) -> tuple[Path, float, float]:
    return (
        entry.wav_path,
        round(entry.offset, 3),
        round(entry.offset + entry.preutterance, 3),
    )


def _role(alias: str, affixes: set[tuple[str, str]]) -> str:
    normalized = normalize_alias(alias, affixes).strip()
    return "initial" if normalized.startswith("-") else "internal"


def _family(final: str) -> str:
    if final.startswith("u") or final in {"o", "ou", "ong"}:
        return "rounded"
    return "other"


def _resolve_candidates(
    candidates: list[_Candidate],
) -> tuple[list[_Candidate], int, int, int]:
    grouped: dict[tuple[Path, float, float], list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[_segment_key(candidate.observation.entry)].append(candidate)

    resolved: list[_Candidate] = []
    duplicate_removed = 0
    ambiguous_segments = 0
    ambiguous_observations = 0
    for rows in grouped.values():
        if len(rows) == 1:
            resolved.append(rows[0])
            continue
        labels = {(row.family, row.final, row.role) for row in rows}
        if len(labels) != 1:
            ambiguous_segments += 1
            ambiguous_observations += len(rows)
            continue
        resolved.append(rows[0])
        duplicate_removed += len(rows) - 1
    return resolved, duplicate_removed, ambiguous_segments, ambiguous_observations


def _feature_vector(entry: OtoEntry) -> np.ndarray:
    whole = consonant_segment(entry)
    core = extract_fricative_features(slice_segment(whole, 0.20, 0.55))
    late = extract_fricative_features(slice_segment(whole, 0.60, 0.90))
    return np.array(
        [
            core.centroid_hz,
            core.spread_hz,
            core.skewness,
            core.kurtosis,
            core.slope,
            core.high_band_ratio,
            late.centroid_hz,
            late.spread_hz,
            late.skewness,
            late.kurtosis,
            late.slope,
            late.high_band_ratio,
        ],
        dtype=np.float64,
    )


def _matrix(samples: list[FHFricativeSample]) -> np.ndarray:
    return np.vstack([sample.vector for sample in samples])


def _cross_oto_set(
    grouped: dict[str, list[FHFricativeSample]],
) -> tuple[float | None, dict[str, float]]:
    scores: dict[str, float] = {}
    names = sorted(grouped)
    for held_out in names:
        train = [
            sample
            for name in names
            if name != held_out
            for sample in grouped[name]
        ]
        test = list(grouped[held_out])
        if not train or not test:
            continue
        train_labels = np.array(
            [0 if sample.family == "other" else 1 for sample in train],
            dtype=np.int8,
        )
        test_labels = np.array(
            [0 if sample.family == "other" else 1 for sample in test],
            dtype=np.int8,
        )
        if len(np.unique(train_labels)) < 2 or len(np.unique(test_labels)) < 2:
            continue
        predicted = nearest_centroid_predict(
            _matrix(train),
            train_labels,
            _matrix(test),
        )
        scores[held_out] = balanced_accuracy(test_labels, predicted)
    if not scores:
        return None, scores
    return float(np.mean(list(scores.values()))), scores


def _standardized_stratum(
    samples: list[FHFricativeSample],
) -> tuple[np.ndarray, int] | None:
    rounded = [sample for sample in samples if sample.family == "rounded"]
    other = [sample for sample in samples if sample.family == "other"]
    if len(rounded) < 2 or len(other) < 2:
        return None
    combined = other + rounded
    matrix = _matrix(combined)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=1)
    scale = np.where(scale < 1e-9, 1.0, scale)
    return (matrix - mean) / scale, len(other)


def _stratified_summary(
    grouped: dict[str, list[FHFricativeSample]],
    seed: int,
    permutations: int = 5000,
) -> tuple[
    float | None,
    float | None,
    dict[str, float],
    dict[str, int],
    tuple[FHFricativeOtoSetResult, ...],
]:
    strata: list[tuple[np.ndarray, int]] = []
    effects: list[np.ndarray] = []
    oto_results: list[FHFricativeOtoSetResult] = []

    for oto_set in sorted(grouped):
        stratum = _standardized_stratum(grouped[oto_set])
        if stratum is None:
            continue
        matrix, other_count = stratum
        effect = np.mean(matrix[other_count:], axis=0) - np.mean(
            matrix[:other_count], axis=0
        )
        strata.append(stratum)
        effects.append(effect)
        oto_results.append(
            FHFricativeOtoSetResult(
                oto_set=oto_set,
                rounded_count=len(matrix) - other_count,
                other_count=other_count,
                distance=float(np.linalg.norm(effect) / np.sqrt(len(_FEATURES))),
                effects={
                    name: float(value)
                    for name, value in zip(_FEATURES, effect, strict=True)
                },
            )
        )

    if len(strata) < 2:
        return None, None, {}, {}, tuple(oto_results)

    effect_matrix = np.vstack(effects)
    pooled = np.mean(effect_matrix, axis=0)
    observed = float(np.linalg.norm(pooled) / np.sqrt(len(_FEATURES)))
    effect_map = {
        name: float(value)
        for name, value in zip(_FEATURES, pooled, strict=True)
    }
    agreement: dict[str, int] = {}
    for index, name in enumerate(_FEATURES):
        target = pooled[index]
        if abs(target) < 1e-12:
            agreement[name] = 0
        else:
            agreement[name] = int(
                np.sum(np.sign(effect_matrix[:, index]) == np.sign(target))
            )

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        perm_effects: list[np.ndarray] = []
        for matrix, other_count in strata:
            order = rng.permutation(len(matrix))
            other_rows = matrix[order[:other_count]]
            rounded_rows = matrix[order[other_count:]]
            perm_effects.append(
                np.mean(rounded_rows, axis=0) - np.mean(other_rows, axis=0)
            )
        perm = np.mean(np.vstack(perm_effects), axis=0)
        distance = float(np.linalg.norm(perm) / np.sqrt(len(_FEATURES)))
        if distance >= observed:
            exceed += 1

    return (
        observed,
        float((exceed + 1) / (permutations + 1)),
        effect_map,
        agreement,
        tuple(oto_results),
    )


def analyze_fh_fricative(root: Path, base_unit: str) -> FHFricativeAnalysis:
    if base_unit not in _SUPPORTED:
        raise ValueError("f/h fricative analyzer supports only f and h")

    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)

    candidates: list[_Candidate] = []
    for observation in observations:
        structure = structure_for(observation)
        if structure.onset != base_unit:
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

    grouped: dict[str, list[FHFricativeSample]] = defaultdict(list)
    skipped = 0
    for candidate in resolved:
        if candidate.role != "internal":
            continue
        try:
            vector = _feature_vector(candidate.observation.entry)
        except (AudioReadError, ValueError, RuntimeError, FloatingPointError):
            skipped += 1
            continue
        grouped[candidate.oto_set].append(
            FHFricativeSample(
                oto_set=candidate.oto_set,
                family=candidate.family,
                final=candidate.final,
                alias=candidate.observation.entry.alias,
                entry=candidate.observation.entry,
                vector=vector,
            )
        )

    samples = [sample for rows in grouped.values() for sample in rows]
    counts = {
        family: sum(sample.family == family for sample in samples)
        for family in _FAMILIES
    }
    cross, cross_by = _cross_oto_set(grouped)
    distance, p_value, effects, agreement, oto_sets = _stratified_summary(
        grouped,
        seed=71011 + sum(ord(char) for char in base_unit),
    )

    return FHFricativeAnalysis(
        base_unit=base_unit,
        samples=len(samples),
        skipped=skipped,
        duplicate_observations_removed=duplicate_removed,
        ambiguous_segments_removed=ambiguous_segments,
        ambiguous_observations_removed=ambiguous_observations,
        counts=counts,
        cross_oto_set_balanced_accuracy=cross,
        cross_by_oto_set=cross_by,
        stratified_distance=distance,
        stratified_permutation_p=p_value,
        stratified_effects=effects,
        effect_sign_agreement=agreement,
        oto_sets=oto_sets,
    )
