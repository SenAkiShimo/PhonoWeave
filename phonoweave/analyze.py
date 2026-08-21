from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, consonant_segment
from .features import FricativeFeatures, extract_fricative_features
from .mandarin import collect_observations, context_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


_FEATURE_NAMES = (
    "centroid_hz",
    "spread_hz",
    "skewness",
    "kurtosis",
    "slope",
    "high_band_ratio",
    "duration_ms",
)


@dataclass(frozen=True)
class SampleFeatures:
    subbank: str
    context: str
    final: str
    alias: str
    wav_path: Path
    features: FricativeFeatures


@dataclass(frozen=True)
class SubbankContrast:
    subbank: str
    plain_count: int
    rounded_count: int
    standardized_distance: float
    centroid_accuracy: float
    mean_plain: dict[str, float]
    mean_rounded: dict[str, float]


@dataclass(frozen=True)
class ContrastAnalysis:
    base_unit: str
    samples: int
    skipped: int
    subbanks: list[SubbankContrast]
    mean_distance: float | None
    distance_cv: float | None


def _subbank_name(root: Path, entry: OtoEntry) -> str:
    directory = entry.oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def _means(matrix: np.ndarray) -> dict[str, float]:
    values = np.mean(matrix, axis=0)
    return {name: float(value) for name, value in zip(_FEATURE_NAMES, values, strict=True)}


def _standardized_distance(plain: np.ndarray, rounded: np.ndarray) -> float:
    combined = np.vstack([plain, rounded])
    scale = np.std(combined, axis=0, ddof=1)
    scale = np.where(scale < 1e-9, 1.0, scale)
    delta = (np.mean(plain, axis=0) - np.mean(rounded, axis=0)) / scale
    return float(np.linalg.norm(delta) / np.sqrt(len(delta)))


def _centroid_accuracy(plain: np.ndarray, rounded: np.ndarray) -> float:
    combined = np.vstack([plain, rounded])
    labels = np.array([0] * len(plain) + [1] * len(rounded), dtype=np.int8)
    scale = np.std(combined, axis=0, ddof=1)
    scale = np.where(scale < 1e-9, 1.0, scale)
    normalized = combined / scale
    plain_center = np.mean(normalized[: len(plain)], axis=0)
    rounded_center = np.mean(normalized[len(plain) :], axis=0)
    d_plain = np.linalg.norm(normalized - plain_center, axis=1)
    d_rounded = np.linalg.norm(normalized - rounded_center, axis=1)
    predicted = (d_rounded < d_plain).astype(np.int8)
    return float(np.mean(predicted == labels))


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
            segment = consonant_segment(observation.entry)
            features = extract_fricative_features(segment)
        except (AudioReadError, ValueError):
            skipped += 1
            continue

        sample = SampleFeatures(
            subbank=_subbank_name(root, observation.entry),
            context=context,
            final=observation.final,
            alias=observation.entry.alias,
            wav_path=observation.entry.wav_path,
            features=features,
        )
        grouped[sample.subbank][context].append(sample)

    subbanks: list[SubbankContrast] = []
    for subbank in sorted(grouped):
        plain_samples = grouped[subbank].get("plain", [])
        rounded_samples = grouped[subbank].get("rounded", [])
        if len(plain_samples) < 2 or len(rounded_samples) < 2:
            continue

        plain = np.vstack([sample.features.vector() for sample in plain_samples])
        rounded = np.vstack([sample.features.vector() for sample in rounded_samples])
        subbanks.append(
            SubbankContrast(
                subbank=subbank,
                plain_count=len(plain_samples),
                rounded_count=len(rounded_samples),
                standardized_distance=_standardized_distance(plain, rounded),
                centroid_accuracy=_centroid_accuracy(plain, rounded),
                mean_plain=_means(plain),
                mean_rounded=_means(rounded),
            )
        )

    distances = np.array([subbank.standardized_distance for subbank in subbanks], dtype=np.float64)
    if len(distances):
        mean_distance = float(np.mean(distances))
        distance_cv = float(np.std(distances) / mean_distance) if mean_distance > 1e-9 else 0.0
    else:
        mean_distance = None
        distance_cv = None

    return ContrastAnalysis(
        base_unit=base_unit,
        samples=sum(s.plain_count + s.rounded_count for s in subbanks),
        skipped=skipped,
        subbanks=subbanks,
        mean_distance=mean_distance,
        distance_cv=distance_cv,
    )
