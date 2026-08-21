from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, consonant_segment, slice_segment
from .features import FricativeFeatures, extract_fricative_features
from .mandarin import collect_observations, context_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


_ACOUSTIC_FEATURES = (
    "centroid_hz",
    "spread_hz",
    "skewness",
    "kurtosis",
    "slope",
    "high_band_ratio",
)


@dataclass(frozen=True)
class SampleFeatures:
    subbank: str
    context: str
    final: str
    alias: str
    wav_path: Path
    core: FricativeFeatures
    late: FricativeFeatures


@dataclass(frozen=True)
class RegionContrast:
    standardized_distance: float
    loo_accuracy: float
    mean_plain: dict[str, float]
    mean_rounded: dict[str, float]


@dataclass(frozen=True)
class SubbankContrast:
    subbank: str
    plain_count: int
    rounded_count: int
    core: RegionContrast
    late: RegionContrast


@dataclass(frozen=True)
class ContrastAnalysis:
    base_unit: str
    samples: int
    skipped: int
    subbanks: list[SubbankContrast]
    mean_core_distance: float | None
    core_distance_cv: float | None
    mean_late_distance: float | None
    late_distance_cv: float | None


def _subbank_name(root: Path, entry: OtoEntry) -> str:
    directory = entry.oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def _matrix(features: list[FricativeFeatures]) -> np.ndarray:
    return np.array(
        [
            [
                f.centroid_hz,
                f.spread_hz,
                f.skewness,
                f.kurtosis,
                f.slope,
                f.high_band_ratio,
            ]
            for f in features
        ],
        dtype=np.float64,
    )


def _means(matrix: np.ndarray) -> dict[str, float]:
    values = np.mean(matrix, axis=0)
    return {name: float(value) for name, value in zip(_ACOUSTIC_FEATURES, values, strict=True)}


def _standardized_distance(plain: np.ndarray, rounded: np.ndarray) -> float:
    combined = np.vstack([plain, rounded])
    scale = np.std(combined, axis=0, ddof=1)
    scale = np.where(scale < 1e-9, 1.0, scale)
    delta = (np.mean(plain, axis=0) - np.mean(rounded, axis=0)) / scale
    return float(np.linalg.norm(delta) / np.sqrt(len(delta)))


def _loo_accuracy(plain: np.ndarray, rounded: np.ndarray) -> float:
    combined = np.vstack([plain, rounded])
    labels = np.array([0] * len(plain) + [1] * len(rounded), dtype=np.int8)
    correct = 0

    for index in range(len(combined)):
        train_mask = np.ones(len(combined), dtype=bool)
        train_mask[index] = False
        train = combined[train_mask]
        train_labels = labels[train_mask]

        scale = np.std(train, axis=0, ddof=1)
        scale = np.where(scale < 1e-9, 1.0, scale)
        train_norm = train / scale
        test_norm = combined[index] / scale

        plain_center = np.mean(train_norm[train_labels == 0], axis=0)
        rounded_center = np.mean(train_norm[train_labels == 1], axis=0)
        d_plain = np.linalg.norm(test_norm - plain_center)
        d_rounded = np.linalg.norm(test_norm - rounded_center)
        predicted = 1 if d_rounded < d_plain else 0
        correct += int(predicted == labels[index])

    return float(correct / len(combined))


def _region_contrast(plain: np.ndarray, rounded: np.ndarray) -> RegionContrast:
    return RegionContrast(
        standardized_distance=_standardized_distance(plain, rounded),
        loo_accuracy=_loo_accuracy(plain, rounded),
        mean_plain=_means(plain),
        mean_rounded=_means(rounded),
    )


def _summary(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    cv = float(np.std(array) / mean) if mean > 1e-9 else 0.0
    return mean, cv


def analyze_fricative_contrast(root: Path, base_unit: str = "sh") -> ContrastAnalysis:
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)

    grouped: dict[str, dict[str, list[SampleFeatures]]] = defaultdict(lambda: defaultdict(list))
    skipped = 0

    for observation in observations:
        if observation.base_unit != base_unit:
            continue
        context = context_for(observation.base_unit, observation.final)
        if context not in {"plain", "rounded"}:
            continue
        try:
            whole = consonant_segment(observation.entry)
            core_segment = slice_segment(whole, 0.20, 0.60)
            late_segment = slice_segment(whole, 0.60, 0.92)
            core_features = extract_fricative_features(core_segment)
            late_features = extract_fricative_features(late_segment)
        except (AudioReadError, ValueError):
            skipped += 1
            continue

        sample = SampleFeatures(
            subbank=_subbank_name(root, observation.entry),
            context=context,
            final=observation.final,
            alias=observation.entry.alias,
            wav_path=observation.entry.wav_path,
            core=core_features,
            late=late_features,
        )
        grouped[sample.subbank][context].append(sample)

    subbanks: list[SubbankContrast] = []
    for subbank in sorted(grouped):
        plain_samples = grouped[subbank].get("plain", [])
        rounded_samples = grouped[subbank].get("rounded", [])
        if len(plain_samples) < 2 or len(rounded_samples) < 2:
            continue

        plain_core = _matrix([sample.core for sample in plain_samples])
        rounded_core = _matrix([sample.core for sample in rounded_samples])
        plain_late = _matrix([sample.late for sample in plain_samples])
        rounded_late = _matrix([sample.late for sample in rounded_samples])

        subbanks.append(
            SubbankContrast(
                subbank=subbank,
                plain_count=len(plain_samples),
                rounded_count=len(rounded_samples),
                core=_region_contrast(plain_core, rounded_core),
                late=_region_contrast(plain_late, rounded_late),
            )
        )

    mean_core, core_cv = _summary([item.core.standardized_distance for item in subbanks])
    mean_late, late_cv = _summary([item.late.standardized_distance for item in subbanks])

    return ContrastAnalysis(
        base_unit=base_unit,
        samples=sum(s.plain_count + s.rounded_count for s in subbanks),
        skipped=skipped,
        subbanks=subbanks,
        mean_core_distance=mean_core,
        core_distance_cv=core_cv,
        mean_late_distance=mean_late,
        late_distance_cv=late_cv,
    )
