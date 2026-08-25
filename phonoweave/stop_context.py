from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import numpy as np

from .contrast import balanced_accuracy, nearest_centroid_predict
from .mandarin import collect_observations, normalize_alias, structure_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .stop import StopCandidate, StopFeatures, _extract, _oto_set_name, _resolve_candidates


_FAMILIES = ("i_series", "u_series", "other")
_FEATURES = (
    "release_to_vowel_ms",
    "release_strength",
    "vowel_periodicity",
    "burst_centroid_hz",
    "burst_high_ratio",
)
_SUPPORTED = {"b", "p", "d", "t", "g", "k"}


@dataclass(frozen=True)
class StopContextSample:
    oto_set: str
    family: str
    final: str
    alias: str
    entry: OtoEntry
    features: StopFeatures

    def vector(self) -> np.ndarray:
        return np.array(
            [getattr(self.features, name) for name in _FEATURES],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class StopContextPair:
    left: str
    right: str
    cross_oto_set_balanced_accuracy: float | None
    cross_by_oto_set: dict[str, float]
    stratified_distance: float | None
    stratified_permutation_p: float | None
    stratified_p_holm: float | None
    stratified_effects: dict[str, float]
    effect_sign_agreement: dict[str, int]
    oto_sets: int


@dataclass(frozen=True)
class StopContextAnalysis:
    base_unit: str
    samples: int
    skipped: int
    duplicate_observations_removed: int
    ambiguous_segments_removed: int
    ambiguous_observations_removed: int
    counts: dict[str, int]
    pairwise: tuple[StopContextPair, ...]


def _role(alias: str, affixes: set[tuple[str, str]]) -> str:
    normalized = normalize_alias(alias, affixes).strip()
    return "initial" if normalized.startswith("-") else "internal"


def _family(final: str) -> str:
    if final.startswith("i"):
        return "i_series"
    if final.startswith("u"):
        return "u_series"
    return "other"


def _matrix(samples: list[StopContextSample]) -> np.ndarray:
    return np.vstack([sample.vector() for sample in samples])


def _cross_oto_set(
    grouped: dict[str, list[StopContextSample]],
    left: str,
    right: str,
) -> tuple[float | None, dict[str, float]]:
    scores: dict[str, float] = {}
    names = sorted(grouped)
    for held_out in names:
        train = [
            sample
            for name in names
            if name != held_out
            for sample in grouped[name]
            if sample.family in {left, right}
        ]
        test = [
            sample
            for sample in grouped[held_out]
            if sample.family in {left, right}
        ]
        if not train or not test:
            continue
        train_labels = np.array(
            [0 if sample.family == left else 1 for sample in train],
            dtype=np.int8,
        )
        test_labels = np.array(
            [0 if sample.family == left else 1 for sample in test],
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


def _stratum(
    samples: list[StopContextSample],
    left: str,
    right: str,
) -> tuple[np.ndarray, int] | None:
    left_samples = [sample for sample in samples if sample.family == left]
    right_samples = [sample for sample in samples if sample.family == right]
    if len(left_samples) < 2 or len(right_samples) < 2:
        return None
    combined = left_samples + right_samples
    matrix = _matrix(combined)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=1)
    scale = np.where(scale < 1e-9, 1.0, scale)
    return (matrix - mean) / scale, len(left_samples)


def _stratified_summary(
    grouped: dict[str, list[StopContextSample]],
    left: str,
    right: str,
    seed: int,
    permutations: int = 5000,
) -> tuple[float | None, float | None, dict[str, float], dict[str, int], int]:
    strata: list[tuple[np.ndarray, int]] = []
    effects: list[np.ndarray] = []
    for oto_set in sorted(grouped):
        stratum = _stratum(grouped[oto_set], left, right)
        if stratum is None:
            continue
        matrix, left_count = stratum
        strata.append(stratum)
        effects.append(
            np.mean(matrix[left_count:], axis=0)
            - np.mean(matrix[:left_count], axis=0)
        )
    if len(strata) < 2:
        return None, None, {}, {}, len(strata)

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
        for matrix, left_count in strata:
            order = rng.permutation(len(matrix))
            left_rows = matrix[order[:left_count]]
            right_rows = matrix[order[left_count:]]
            perm_effects.append(
                np.mean(right_rows, axis=0) - np.mean(left_rows, axis=0)
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
        len(strata),
    )


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


def analyze_stop_context(root: Path, base_unit: str) -> StopContextAnalysis:
    if base_unit not in _SUPPORTED:
        raise ValueError("stop context analyzer supports b, p, d, t, g, k")

    root = root.expanduser().resolve()
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
    grouped: dict[str, list[StopContextSample]] = defaultdict(list)
    skipped = 0
    for candidate in resolved:
        if candidate.role != "internal":
            continue
        try:
            features = _extract(candidate.observation.entry)
        except (ValueError, RuntimeError, FloatingPointError):
            skipped += 1
            continue
        grouped[candidate.oto_set].append(
            StopContextSample(
                oto_set=candidate.oto_set,
                family=_family(candidate.final),
                final=candidate.final,
                alias=candidate.observation.entry.alias,
                entry=candidate.observation.entry,
                features=features,
            )
        )

    samples = [sample for rows in grouped.values() for sample in rows]
    counts = {
        family: sum(sample.family == family for sample in samples)
        for family in _FAMILIES
    }

    results: list[StopContextPair] = []
    for pair_index, (left, right) in enumerate(combinations(_FAMILIES, 2)):
        cross, cross_by = _cross_oto_set(grouped, left, right)
        distance, p_value, effects, agreement, oto_sets = _stratified_summary(
            grouped,
            left,
            right,
            seed=53011 + pair_index * 137 + sum(ord(char) for char in base_unit),
        )
        if cross is None and oto_sets == 0:
            continue
        results.append(
            StopContextPair(
                left=left,
                right=right,
                cross_oto_set_balanced_accuracy=cross,
                cross_by_oto_set=cross_by,
                stratified_distance=distance,
                stratified_permutation_p=p_value,
                stratified_p_holm=None,
                stratified_effects=effects,
                effect_sign_agreement=agreement,
                oto_sets=oto_sets,
            )
        )

    adjusted = _holm_adjust([item.stratified_permutation_p for item in results])
    results = [
        replace(item, stratified_p_holm=adjusted[index])
        for index, item in enumerate(results)
    ]
    return StopContextAnalysis(
        base_unit=base_unit,
        samples=len(samples),
        skipped=skipped,
        duplicate_observations_removed=duplicate_removed,
        ambiguous_segments_removed=ambiguous_segments,
        ambiguous_observations_removed=ambiguous_observations,
        counts=counts,
        pairwise=tuple(results),
    )
