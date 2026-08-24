from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import numpy as np

from .audio import AudioReadError, AudioSegment, consonant_segment, slice_segment
from .contrast import balanced_accuracy, nearest_centroid_predict
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
_WINDOWS = (
    ("core", 0.18, 0.48),
    ("late", 0.60, 0.90),
)


@dataclass(frozen=True)
class TimecourseFeatures:
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
class LateralTimeSample:
    oto_set: str
    role: str
    family: str
    final: str
    alias: str
    entry: OtoEntry
    core: TimecourseFeatures
    late: TimecourseFeatures


@dataclass(frozen=True)
class LateralWindowPairResult:
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
class LateralWindowResult:
    name: str
    start_fraction: float
    end_fraction: float
    pairwise: tuple[LateralWindowPairResult, ...]


@dataclass(frozen=True)
class LateralTimeRoleResult:
    role: str
    counts: dict[str, int]
    windows: tuple[LateralWindowResult, ...]


@dataclass(frozen=True)
class LateralTimecourseAnalysis:
    samples: int
    skipped: int
    duplicate_segments: int
    roles: tuple[LateralTimeRoleResult, ...]


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


def _peak_frequency(
    freqs: np.ndarray,
    power: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    mask = (freqs >= lower) & (freqs <= upper)
    if not np.any(mask):
        return 0.0
    band_freqs = freqs[mask]
    band_power = power[mask]
    return float(band_freqs[int(np.argmax(band_power))])


def _features(segment: AudioSegment) -> TimecourseFeatures:
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
        raise AudioReadError("lateral window has no usable spectral energy")

    weights = band_power / total
    centroid = float(np.sum(band_freqs * weights))
    flatness = float(np.exp(np.mean(np.log(band_power))) / np.mean(band_power))
    low = float(np.sum(band_power[band_freqs < 1000.0]) / total)
    mid = float(
        np.sum(band_power[(band_freqs >= 1000.0) & (band_freqs < 3000.0)])
        / total
    )
    high = float(np.sum(band_power[band_freqs >= 3000.0]) / total)
    peak_1 = _peak_frequency(freqs, power, 250.0, 1300.0)
    peak_2 = _peak_frequency(
        freqs,
        power,
        max(900.0, peak_1 + 180.0),
        3300.0,
    )

    return TimecourseFeatures(
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


def _window_features(whole: AudioSegment, start: float, end: float) -> TimecourseFeatures:
    return _features(slice_segment(whole, start, end))


def _matrix(samples: list[LateralTimeSample], window: str) -> np.ndarray:
    return np.vstack([getattr(sample, window).vector() for sample in samples])


def _cross_oto_set(
    grouped: dict[str, list[LateralTimeSample]],
    left: str,
    right: str,
    window: str,
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
            _matrix(train, window),
            train_labels,
            _matrix(test, window),
        )
        scores[held_out] = balanced_accuracy(test_labels, predicted)

    if not scores:
        return None, scores
    return float(np.mean(list(scores.values()))), scores


def _stratum(
    samples: list[LateralTimeSample],
    left: str,
    right: str,
    window: str,
) -> tuple[np.ndarray, int] | None:
    left_samples = [sample for sample in samples if sample.family == left]
    right_samples = [sample for sample in samples if sample.family == right]
    if not left_samples or not right_samples:
        return None

    combined = left_samples + right_samples
    matrix = _matrix(combined, window)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=1)
    scale = np.where(scale < 1e-9, 1.0, scale)
    return (matrix - mean) / scale, len(left_samples)


def _stratified_summary(
    grouped: dict[str, list[LateralTimeSample]],
    left: str,
    right: str,
    window: str,
    seed: int,
    permutations: int = 5000,
) -> tuple[float | None, float | None, dict[str, float], dict[str, int], int]:
    strata: list[tuple[np.ndarray, int]] = []
    effects: list[np.ndarray] = []

    for oto_set in sorted(grouped):
        stratum = _stratum(grouped[oto_set], left, right, window)
        if stratum is None:
            continue
        matrix, left_count = stratum
        effects.append(
            np.mean(matrix[left_count:], axis=0)
            - np.mean(matrix[:left_count], axis=0)
        )
        strata.append(stratum)

    if len(strata) < 2:
        return None, None, {}, {}, len(strata)

    effect_matrix = np.vstack(effects)
    pooled_effect = np.mean(effect_matrix, axis=0)
    observed = float(np.linalg.norm(pooled_effect) / np.sqrt(len(_FEATURES)))
    effect_map = {
        name: float(value)
        for name, value in zip(_FEATURES, pooled_effect, strict=True)
    }

    agreement: dict[str, int] = {}
    for index, name in enumerate(_FEATURES):
        target = pooled_effect[index]
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
        perm_effect = np.mean(np.vstack(perm_effects), axis=0)
        perm_distance = float(
            np.linalg.norm(perm_effect) / np.sqrt(len(_FEATURES))
        )
        if perm_distance >= observed:
            exceed += 1

    p_value = float((exceed + 1) / (permutations + 1))
    return observed, p_value, effect_map, agreement, len(strata)


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


def _window_result(
    grouped: dict[str, list[LateralTimeSample]],
    name: str,
    start: float,
    end: float,
    seed_base: int,
) -> LateralWindowResult:
    pairwise: list[LateralWindowPairResult] = []

    for pair_index, (left, right) in enumerate(combinations(_FAMILIES, 2)):
        cross, cross_by = _cross_oto_set(grouped, left, right, name)
        distance, p_value, effects, agreement, oto_sets = _stratified_summary(
            grouped,
            left,
            right,
            name,
            seed=seed_base + pair_index,
        )
        pairwise.append(
            LateralWindowPairResult(
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

    adjusted = _holm_adjust(
        [item.stratified_permutation_p for item in pairwise]
    )
    pairwise = [
        replace(item, stratified_p_holm=adjusted[index])
        for index, item in enumerate(pairwise)
    ]
    return LateralWindowResult(
        name=name,
        start_fraction=start,
        end_fraction=end,
        pairwise=tuple(pairwise),
    )


def analyze_lateral_timecourse(root: Path) -> LateralTimecourseAnalysis:
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)

    seen_segments: set[tuple[Path, float, float]] = set()
    duplicate_segments = 0
    skipped = 0
    grouped: dict[str, dict[str, list[LateralTimeSample]]] = defaultdict(
        lambda: defaultdict(list)
    )

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
            whole = consonant_segment(observation.entry, edge_trim=0.04)
            core = _window_features(whole, 0.18, 0.48)
            late = _window_features(whole, 0.60, 0.90)
        except (AudioReadError, ValueError):
            skipped += 1
            continue

        sample = LateralTimeSample(
            oto_set=_oto_set_name(root, observation.entry),
            role=_alias_role(observation.entry.alias, affixes),
            family=_family(structure.final),
            final=structure.final,
            alias=observation.entry.alias,
            entry=observation.entry,
            core=core,
            late=late,
        )
        grouped[sample.role][sample.oto_set].append(sample)

    roles: list[LateralTimeRoleResult] = []
    for role_index, role in enumerate(sorted(grouped)):
        samples = [
            sample
            for rows in grouped[role].values()
            for sample in rows
        ]
        counts = {
            family: sum(sample.family == family for sample in samples)
            for family in _FAMILIES
        }
        windows = tuple(
            _window_result(
                grouped[role],
                name,
                start,
                end,
                seed_base=17011 + role_index * 1000 + window_index * 100,
            )
            for window_index, (name, start, end) in enumerate(_WINDOWS)
        )
        roles.append(
            LateralTimeRoleResult(
                role=role,
                counts=counts,
                windows=windows,
            )
        )

    return LateralTimecourseAnalysis(
        samples=sum(
            sum(len(rows) for rows in by_oto_set.values())
            for by_oto_set in grouped.values()
        ),
        skipped=skipped,
        duplicate_segments=duplicate_segments,
        roles=tuple(roles),
    )
