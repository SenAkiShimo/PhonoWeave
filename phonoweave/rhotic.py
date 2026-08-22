from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, consonant_segment, slice_segment
from .contrast import balanced_accuracy, loo_multiclass_balanced_accuracy, nearest_centroid_predict
from .mandarin import collect_observations, context_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


_CONTEXTS = ("plain", "front", "rounded")
_FEATURES = (
    "centroid_hz",
    "spread_hz",
    "spectral_flatness",
    "high_band_ratio",
    "periodicity",
    "f2_hz",
    "f3_hz",
    "f3_minus_f2_hz",
)


@dataclass(frozen=True)
class RhoticFeatures:
    centroid_hz: float
    spread_hz: float
    spectral_flatness: float
    high_band_ratio: float
    periodicity: float
    f2_hz: float
    f3_hz: float
    f3_minus_f2_hz: float

    def vector(self) -> np.ndarray:
        return np.array([
            self.centroid_hz,
            self.spread_hz,
            self.spectral_flatness,
            self.high_band_ratio,
            self.periodicity,
            self.f2_hz,
            self.f3_hz,
            self.f3_minus_f2_hz,
        ], dtype=np.float64)


@dataclass(frozen=True)
class RhoticSample:
    subbank: str
    context: str
    alias: str
    final: str
    entry: OtoEntry
    features: RhoticFeatures


@dataclass(frozen=True)
class RhoticSubbankResult:
    subbank: str
    counts: dict[str, int]
    loo_balanced_accuracy: float
    means: dict[str, dict[str, float]]


@dataclass(frozen=True)
class RhoticAnalysis:
    samples: int
    skipped: int
    subbanks: list[RhoticSubbankResult]
    cross_subbank_balanced_accuracy: float | None
    cross_by_subbank: dict[str, float]


def _subbank_name(root: Path, entry: OtoEntry) -> str:
    directory = entry.oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def _spectrum(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    samples = samples.astype(np.float64, copy=False)
    samples = samples - np.mean(samples)
    window = np.hanning(len(samples))
    power = np.abs(np.fft.rfft(samples * window)) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    return freqs, power


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


def _extract_features(entry: OtoEntry) -> RhoticFeatures:
    whole = consonant_segment(entry, edge_trim=0.04)
    segment = slice_segment(whole, 0.30, 0.92)
    freqs, power = _spectrum(segment.samples, segment.sample_rate)

    mask = freqs >= 300.0
    band_freqs = freqs[mask]
    band_power = power[mask]
    total = float(np.sum(band_power))
    if total <= 1e-16:
        raise AudioReadError("rhotic segment has no usable spectral energy")

    weights = band_power / total
    centroid = float(np.sum(band_freqs * weights))
    spread = float(np.sqrt(np.sum(((band_freqs - centroid) ** 2) * weights)))
    flatness = float(np.exp(np.mean(np.log(band_power))) / np.mean(band_power))
    high_ratio = float(np.sum(band_power[band_freqs >= min(6000.0, segment.sample_rate * 0.22)]) / total)
    periodicity = _periodicity(segment.samples, segment.sample_rate)

    f2 = _peak_frequency(freqs, power, 700.0, 2200.0)
    f3 = _peak_frequency(freqs, power, max(1800.0, f2 + 250.0), 3800.0)
    gap = max(0.0, f3 - f2)

    return RhoticFeatures(
        centroid_hz=centroid,
        spread_hz=spread,
        spectral_flatness=flatness,
        high_band_ratio=high_ratio,
        periodicity=periodicity,
        f2_hz=f2,
        f3_hz=f3,
        f3_minus_f2_hz=gap,
    )


def _matrix(samples: list[RhoticSample]) -> np.ndarray:
    return np.vstack([sample.features.vector() for sample in samples])


def _labels(samples: list[RhoticSample]) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(_CONTEXTS)}
    return np.array([mapping[sample.context] for sample in samples], dtype=np.int8)


def _means(samples: list[RhoticSample]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for context in _CONTEXTS:
        rows = [sample.features.vector() for sample in samples if sample.context == context]
        if not rows:
            continue
        values = np.mean(np.vstack(rows), axis=0)
        output[context] = {
            name: float(value)
            for name, value in zip(_FEATURES, values, strict=True)
        }
    return output


def _cross_subbank(grouped: dict[str, list[RhoticSample]]) -> tuple[float | None, dict[str, float]]:
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
        if len(np.unique(train_labels)) < 3 or len(np.unique(test_labels)) < 3:
            continue
        predicted = nearest_centroid_predict(_matrix(train), train_labels, _matrix(test))
        scores[held_out] = balanced_accuracy(test_labels, predicted)

    if not scores:
        return None, scores
    return float(np.mean(list(scores.values()))), scores


def analyze_rhotic_contrast(root: Path) -> RhoticAnalysis:
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)
    grouped: dict[str, list[RhoticSample]] = defaultdict(list)
    skipped = 0

    for observation in observations:
        if observation.base_unit != "r":
            continue
        context = context_for("r", observation.final)
        if context not in _CONTEXTS:
            continue
        try:
            features = _extract_features(observation.entry)
        except (AudioReadError, ValueError):
            skipped += 1
            continue
        grouped[_subbank_name(root, observation.entry)].append(
            RhoticSample(
                subbank=_subbank_name(root, observation.entry),
                context=context,
                alias=observation.entry.alias,
                final=observation.final,
                entry=observation.entry,
                features=features,
            )
        )

    subbanks: list[RhoticSubbankResult] = []
    for subbank in sorted(grouped):
        samples = grouped[subbank]
        counts = {context: sum(sample.context == context for sample in samples) for context in _CONTEXTS}
        if any(counts[context] < 2 for context in _CONTEXTS):
            continue
        subbanks.append(
            RhoticSubbankResult(
                subbank=subbank,
                counts=counts,
                loo_balanced_accuracy=loo_multiclass_balanced_accuracy(_matrix(samples), _labels(samples)),
                means=_means(samples),
            )
        )

    cross, cross_by_subbank = _cross_subbank(grouped)
    return RhoticAnalysis(
        samples=sum(sum(item.counts.values()) for item in subbanks),
        skipped=skipped,
        subbanks=subbanks,
        cross_subbank_balanced_accuracy=cross,
        cross_by_subbank=cross_by_subbank,
    )
