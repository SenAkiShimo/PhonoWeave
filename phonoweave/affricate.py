from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError
from .contrast import balanced_accuracy, loo_balanced_accuracy, nearest_centroid_predict, permutation_p, standardized_distance, standardized_effects
from .features import FricativeFeatures, extract_fricative_features
from .mandarin import collect_observations, context_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .segmentation import detect_affricate_frication


_FEATURES = (
    "centroid_hz",
    "spread_hz",
    "skewness",
    "kurtosis",
    "slope",
    "high_band_ratio",
    "frication_duration_ms",
)

_SUPPORTED_BASES = {"zh", "ch", "z", "c"}


@dataclass(frozen=True)
class AffricateFeatures:
    centroid_hz: float
    spread_hz: float
    skewness: float
    kurtosis: float
    slope: float
    high_band_ratio: float
    frication_duration_ms: float

    def vector(self) -> np.ndarray:
        return np.array([
            self.centroid_hz,
            self.spread_hz,
            self.skewness,
            self.kurtosis,
            self.slope,
            self.high_band_ratio,
            self.frication_duration_ms,
        ], dtype=np.float64)


@dataclass(frozen=True)
class AffricateSample:
    subbank: str
    context: str
    final: str
    alias: str
    entry: OtoEntry
    features: AffricateFeatures


@dataclass(frozen=True)
class AffricateSubbankContrast:
    subbank: str
    plain_count: int
    rounded_count: int
    distance: float
    loo_balanced_accuracy: float
    permutation_p: float
    mean_plain: dict[str, float]
    mean_rounded: dict[str, float]
    effects: dict[str, float]


@dataclass(frozen=True)
class AffricateAnalysis:
    base_unit: str
    samples: int
    skipped: int
    mean_distance: float | None
    distance_cv: float | None
    cross_subbank_balanced_accuracy: float | None
    cross_by_subbank: dict[str, float]
    subbanks: list[AffricateSubbankContrast]


def _subbank_name(root: Path, entry: OtoEntry) -> str:
    directory = entry.oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def _extract_features(entry: OtoEntry) -> AffricateFeatures:
    segmentation = detect_affricate_frication(entry)
    spectral: FricativeFeatures = extract_fricative_features(segmentation.frication)
    return AffricateFeatures(
        centroid_hz=spectral.centroid_hz,
        spread_hz=spectral.spread_hz,
        skewness=spectral.skewness,
        kurtosis=spectral.kurtosis,
        slope=spectral.slope,
        high_band_ratio=spectral.high_band_ratio,
        frication_duration_ms=segmentation.frication.end_ms - segmentation.frication.start_ms,
    )


def _matrix(samples: list[AffricateSample]) -> np.ndarray:
    return np.vstack([sample.features.vector() for sample in samples])


def _means(matrix: np.ndarray) -> dict[str, float]:
    values = np.mean(matrix, axis=0)
    return {name: float(value) for name, value in zip(_FEATURES, values, strict=True)}


def _cross_subbank(grouped: dict[str, dict[str, list[AffricateSample]]]) -> tuple[float | None, dict[str, float]]:
    names = sorted(grouped)
    scores: dict[str, float] = {}
    if len(names) < 2:
        return None, scores

    for held_out in names:
        train: list[AffricateSample] = []
        test: list[AffricateSample] = []
        for name in names:
            rows = grouped[name].get("plain", []) + grouped[name].get("rounded", [])
            (test if name == held_out else train).extend(rows)
        if not train or not test:
            continue
        train_labels = np.array([0 if row.context == "plain" else 1 for row in train], dtype=np.int8)
        test_labels = np.array([0 if row.context == "plain" else 1 for row in test], dtype=np.int8)
        if len(np.unique(train_labels)) < 2 or len(np.unique(test_labels)) < 2:
            continue
        predicted = nearest_centroid_predict(_matrix(train), train_labels, _matrix(test))
        scores[held_out] = balanced_accuracy(test_labels, predicted)

    if not scores:
        return None, scores
    return float(np.mean(list(scores.values()))), scores


def analyze_affricate_contrast(root: Path, base_unit: str) -> AffricateAnalysis:
    if base_unit not in _SUPPORTED_BASES:
        supported = ", ".join(sorted(_SUPPORTED_BASES))
        raise ValueError(f"affricate analyzer supports {supported}")

    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)
    grouped: dict[str, dict[str, list[AffricateSample]]] = defaultdict(lambda: defaultdict(list))
    skipped = 0

    for observation in observations:
        if observation.base_unit != base_unit:
            continue
        context = context_for(observation.base_unit, observation.final)
        if context not in {"plain", "rounded"}:
            continue
        try:
            features = _extract_features(observation.entry)
        except (AudioReadError, ValueError):
            skipped += 1
            continue
        sample = AffricateSample(
            subbank=_subbank_name(root, observation.entry),
            context=context,
            final=observation.final,
            alias=observation.entry.alias,
            entry=observation.entry,
            features=features,
        )
        grouped[sample.subbank][context].append(sample)

    subbanks: list[AffricateSubbankContrast] = []
    for index, subbank in enumerate(sorted(grouped)):
        plain_samples = grouped[subbank].get("plain", [])
        rounded_samples = grouped[subbank].get("rounded", [])
        if len(plain_samples) < 2 or len(rounded_samples) < 2:
            continue
        plain = _matrix(plain_samples)
        rounded = _matrix(rounded_samples)
        subbanks.append(AffricateSubbankContrast(
            subbank=subbank,
            plain_count=len(plain_samples),
            rounded_count=len(rounded_samples),
            distance=standardized_distance(plain, rounded, _FEATURES),
            loo_balanced_accuracy=loo_balanced_accuracy(plain, rounded),
            permutation_p=permutation_p(plain, rounded, _FEATURES, seed=3811 + index),
            mean_plain=_means(plain),
            mean_rounded=_means(rounded),
            effects=standardized_effects(plain, rounded, _FEATURES),
        ))

    distances = np.array([item.distance for item in subbanks], dtype=np.float64)
    mean_distance = float(np.mean(distances)) if len(distances) else None
    distance_cv = (
        float(np.std(distances) / mean_distance)
        if mean_distance is not None and mean_distance > 1e-9
        else None
    )
    cross, cross_by_subbank = _cross_subbank(grouped)
    return AffricateAnalysis(
        base_unit=base_unit,
        samples=sum(item.plain_count + item.rounded_count for item in subbanks),
        skipped=skipped,
        mean_distance=mean_distance,
        distance_cv=distance_cv,
        cross_subbank_balanced_accuracy=cross,
        cross_by_subbank=cross_by_subbank,
        subbanks=subbanks,
    )
