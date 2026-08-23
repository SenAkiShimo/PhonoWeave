from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import numpy as np

from .audio import AudioReadError, consonant_segment, slice_segment
from .contrast import (
    balanced_accuracy,
    loo_balanced_accuracy,
    nearest_centroid_predict,
    permutation_p,
    standardized_distance,
)
from .mandarin import collect_observations, normalize_alias, structure_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


_FAMILIES = ("i_series", "u_series", "v_series", "other")
_FEATURES = (
    "low_ratio",
    "mid_ratio",
    "high_ratio",
    "centroid_hz",
    "spectral_flatness",
    "periodicity",
    "peak_1_hz",
    "peak_2_hz",
    "peak_gap_hz",
)


@dataclass(frozen=True)
class LateralFeatures:
    low_ratio: float
    mid_ratio: float
    high_ratio: float
    centroid_hz: float
    spectral_flatness: float
    periodicity: float
    peak_1_hz: float
    peak_2_hz: float
    peak_gap_hz: float

    def vector(self) -> np.ndarray:
        return np.array(
            [getattr(self, name) for name in _FEATURES],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class LateralSample:
    subbank: str
    role: str
    family: str
    final: str
    alias: str
    entry: OtoEntry
    features: LateralFeatures


@dataclass(frozen=True)
class LateralPairSubbankResult:
    subbank: str
    left_count: int
    right_count: int
    distance: float
    loo_balanced_accuracy: float
    permutation_p: float


@dataclass(frozen=True)
class LateralPairResult:
    left: str
    right: str
    cross_subbank_balanced_accuracy: float | None
    cross_by_subbank: dict[str, float]
    mean_distance: float | None
    stratified_distance: float | None
    stratified_permutation_p: float | None
    stratified_p_holm: float | None
    stratified_effects: dict[str, float]
    effect_sign_agreement: dict[str, int]
    stratified_subbanks: int
    subbanks: tuple[LateralPairSubbankResult, ...]


@dataclass(frozen=True)
class LateralRoleResult:
    role: str
    counts: dict[str, int]
    cross_subbank_balanced_accuracy: float | None
    cross_by_subbank: dict[str, float]
    pairwise: tuple[LateralPairResult, ...]


@dataclass(frozen=True)
class LateralAnalysis:
    samples: int
    skipped: int
    duplicate_segments: int
    roles: tuple[LateralRoleResult, ...]


def _subbank_name(root: Path, entry: OtoEntry) -> str:
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


def _family(final: str) -> str:
    if final.startswith("i"):
        return "i_series"
    if final.startswith("u"):
        return "u_series"
    if final.startswith("v") or final == "ü":
        return "v_series"
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


def _peak_frequency(freqs: np.ndarray, power: np.ndarray, lower: float, upper: float) -> float:
    mask = (freqs >= lower) & (freqs <= upper)
    if not np.any(mask):
        return 0.0
    band_freqs = freqs[mask]
    band_power = power[mask]
    return float(band_freqs[int(np.argmax(band_power))])


def _extract_features(entry: OtoEntry) -> LateralFeatures:
    whole = consonant_segment(entry, edge_trim=0.04)
    segment = slice_segment(whole, 0.30, 0.90)
    samples = segment.samples.astype(np.float64, copy=False)
    samples = samples - np.mean(samples)
    window = np.hanning(len(samples))
    power = np.abs(np.fft.rfft(samples * window)) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / segment.sample_rate)

    mask = (freqs >= 150.0) & (freqs <= min(8000.0, segment.sample_rate * 0.45))
    band_freqs = freqs[mask]
    band_power = power[mask]
    total = float(np.sum(band_power))
    if total <= 1e-16:
        raise AudioReadError("lateral segment has no usable spectral energy")

    weights = band_power / total
    centroid = float(np.sum(band_freqs * weights))
    flatness = float(np.exp(np.mean(np.log(band_power))) / np.mean(band_power))

    low = float(np.sum(band_power[band_freqs < 1000.0]) / total)
    mid = float(np.sum(band_power[(band_freqs >= 1000.0) & (band_freqs < 3000.0)]) / total)
    high = float(np.sum(band_power[band_freqs >= 3000.0]) / total)
    peak_1 = _peak_frequency(freqs, power, 250.0, 1300.0)
    peak_2 = _peak_frequency(freqs, power, max(900.0, peak_1 + 180.0), 3300.0)

    return LateralFeatures(
        low_ratio=low,
        mid_ratio=mid,
        high_ratio=high,
        centroid_hz=centroid,
        spectral_flatness=flatness,
        periodicity=_periodicity(segment.samples, segment.sample_rate),
        peak_1_hz=peak_1,
        peak_2_hz=peak_2,
        peak_gap_hz=max(0.0, peak_2 - peak_1),
    )


def _labels(samples: list[LateralSample]) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(_FAMILIES)}
    return np.array([mapping[sample.family] for sample in samples], dtype=np.int8)


def _matrix(samples: list[LateralSample]) -> np.ndarray:
    return np.vstack([sample.features.vector() for sample in samples])


def _cross_subbank(grouped: dict[str, list[LateralSample]]) -> tuple[float | None, dict[str, float]]:
    names = sorted(grouped)
    scores: dict[str, float] = {}
    if len(names) < 2:
        return None, scores

    for held_out in names:
        train = [sample for name in names if name != held_out for sample in grouped[name]]
        test = list(grouped[held_out])
        if not train or not test:
            continue
        train_labels = _labels(train)
        test_labels = _labels(test)
        classes = np.unique(np.concatenate([train_labels, test_labels]))
        if len(classes) < 2:
            continue
        if any(not np.any(train_labels == label) or not np.any(test_labels == label) for label in classes):
            continue
        predicted = nearest_centroid_predict(_matrix(train), train_labels, _matrix(test))
        scores[held_out] = balanced_accuracy(test_labels, predicted)

    if not scores:
        return None, scores
    return float(np.mean(list(scores.values()))), scores


def _pair_cross_subbank(
    grouped: dict[str, list[LateralSample]],
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
        train_labels = np.array([0 if sample.family == left else 1 for sample in train], dtype=np.int8)
        test_labels = np.array([0 if sample.family == left else 1 for sample in test], dtype=np.int8)
        if len(np.unique(train_labels)) < 2 or len(np.unique(test_labels)) < 2:
            continue
        predicted = nearest_centroid_predict(_matrix(train), train_labels, _matrix(test))
        scores[held_out] = balanced_accuracy(test_labels, predicted)

    if not scores:
        return None, scores
    return float(np.mean(list(scores.values()))), scores


def _stratum(
    samples: list[LateralSample],
    left: str,
    right: str,
) -> tuple[np.ndarray, int] | None:
    left_samples = [sample for sample in samples if sample.family == left]
    right_samples = [sample for sample in samples if sample.family == right]
    if not left_samples or not right_samples:
        return None

    combined = left_samples + right_samples
    matrix = _matrix(combined)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=1)
    scale = np.where(scale < 1e-9, 1.0, scale)
    normalized = (matrix - mean) / scale
    return normalized, len(left_samples)


def _stratified_summary(
    grouped: dict[str, list[LateralSample]],
    left: str,
    right: str,
    seed: int,
    permutations: int = 5000,
) -> tuple[float | None, float | None, dict[str, float], dict[str, int], int]:
    strata: list[tuple[np.ndarray, int]] = []
    effects: list[np.ndarray] = []

    for subbank in sorted(grouped):
        stratum = _stratum(grouped[subbank], left, right)
        if stratum is None:
            continue
        matrix, left_count = stratum
        effect = np.mean(matrix[left_count:], axis=0) - np.mean(matrix[:left_count], axis=0)
        strata.append(stratum)
        effects.append(effect)

    if len(strata) < 2:
        return None, None, {}, {}, len(strata)

    effect_matrix = np.vstack(effects)
    pooled_effect = np.mean(effect_matrix, axis=0)
    observed = float(np.linalg.norm(pooled_effect) / np.sqrt(len(_FEATURES)))

    effect_map = {
        name: float(value)
        for name, value in zip(_FEATURES, pooled_effect, strict=True)
    }
    sign_agreement: dict[str, int] = {}
    for index, name in enumerate(_FEATURES):
        target = pooled_effect[index]
        if abs(target) < 1e-12:
            sign_agreement[name] = 0
            continue
        sign_agreement[name] = int(np.sum(np.sign(effect_matrix[:, index]) == np.sign(target)))

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        perm_effects: list[np.ndarray] = []
        for matrix, left_count in strata:
            order = rng.permutation(len(matrix))
            left_rows = matrix[order[:left_count]]
            right_rows = matrix[order[left_count:]]
            perm_effects.append(np.mean(right_rows, axis=0) - np.mean(left_rows, axis=0))
        perm_effect = np.mean(np.vstack(perm_effects), axis=0)
        perm_distance = float(np.linalg.norm(perm_effect) / np.sqrt(len(_FEATURES)))
        if perm_distance >= observed:
            exceed += 1

    p_value = float((exceed + 1) / (permutations + 1))
    return observed, p_value, effect_map, sign_agreement, len(strata)


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


def _pairwise(
    grouped: dict[str, list[LateralSample]],
    seed_base: int,
) -> tuple[LateralPairResult, ...]:
    results: list[LateralPairResult] = []

    for pair_index, (left, right) in enumerate(combinations(_FAMILIES, 2)):
        subbanks: list[LateralPairSubbankResult] = []
        for subbank_index, subbank in enumerate(sorted(grouped)):
            left_samples = [sample for sample in grouped[subbank] if sample.family == left]
            right_samples = [sample for sample in grouped[subbank] if sample.family == right]
            if len(left_samples) < 2 or len(right_samples) < 2:
                continue
            left_matrix = _matrix(left_samples)
            right_matrix = _matrix(right_samples)
            subbanks.append(
                LateralPairSubbankResult(
                    subbank=subbank,
                    left_count=len(left_samples),
                    right_count=len(right_samples),
                    distance=standardized_distance(left_matrix, right_matrix, _FEATURES),
                    loo_balanced_accuracy=loo_balanced_accuracy(left_matrix, right_matrix),
                    permutation_p=permutation_p(
                        left_matrix,
                        right_matrix,
                        _FEATURES,
                        seed=seed_base + pair_index * 100 + subbank_index,
                    ),
                )
            )

        cross, cross_by = _pair_cross_subbank(grouped, left, right)
        mean_distance = None
        if subbanks:
            mean_distance = float(np.mean([item.distance for item in subbanks]))

        stratified_distance, stratified_p, effects, sign_agreement, strata_count = _stratified_summary(
            grouped,
            left,
            right,
            seed=seed_base + 5000 + pair_index,
        )
        results.append(
            LateralPairResult(
                left=left,
                right=right,
                cross_subbank_balanced_accuracy=cross,
                cross_by_subbank=cross_by,
                mean_distance=mean_distance,
                stratified_distance=stratified_distance,
                stratified_permutation_p=stratified_p,
                stratified_p_holm=None,
                stratified_effects=effects,
                effect_sign_agreement=sign_agreement,
                stratified_subbanks=strata_count,
                subbanks=tuple(subbanks),
            )
        )

    adjusted = _holm_adjust([item.stratified_permutation_p for item in results])
    return tuple(
        replace(item, stratified_p_holm=adjusted[index])
        for index, item in enumerate(results)
    )


def analyze_lateral(root: Path) -> LateralAnalysis:
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)

    seen_segments: set[tuple[Path, float, float]] = set()
    duplicate_segments = 0
    skipped = 0
    grouped: dict[str, dict[str, list[LateralSample]]] = defaultdict(lambda: defaultdict(list))

    for observation in observations:
        structure = structure_for(observation)
        if structure.onset != "l":
            continue
        key = _segment_key(observation.entry)
        if key in seen_segments:
            duplicate_segments += 1
            continue
        seen_segments.add(key)
        try:
            features = _extract_features(observation.entry)
        except (AudioReadError, ValueError):
            skipped += 1
            continue
        sample = LateralSample(
            subbank=_subbank_name(root, observation.entry),
            role=_alias_role(observation.entry.alias, affixes),
            family=_family(structure.final),
            final=structure.final,
            alias=observation.entry.alias,
            entry=observation.entry,
            features=features,
        )
        grouped[sample.role][sample.subbank].append(sample)

    roles: list[LateralRoleResult] = []
    for role_index, role in enumerate(sorted(grouped)):
        samples = [sample for rows in grouped[role].values() for sample in rows]
        counts = {family: sum(sample.family == family for sample in samples) for family in _FAMILIES}
        cross, cross_by = _cross_subbank(grouped[role])
        roles.append(
            LateralRoleResult(
                role=role,
                counts=counts,
                cross_subbank_balanced_accuracy=cross,
                cross_by_subbank=cross_by,
                pairwise=_pairwise(grouped[role], seed_base=7301 + role_index * 1000),
            )
        )

    return LateralAnalysis(
        samples=sum(sum(len(rows) for rows in by_subbank.values()) for by_subbank in grouped.values()),
        skipped=skipped,
        duplicate_segments=duplicate_segments,
        roles=tuple(roles),
    )
