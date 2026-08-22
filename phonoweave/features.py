from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .audio import AudioSegment


@dataclass(frozen=True)
class FricativeFeatures:
    centroid_hz: float
    spread_hz: float
    skewness: float
    kurtosis: float
    slope: float
    high_band_ratio: float
    duration_ms: float

    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.centroid_hz,
                self.spread_hz,
                self.skewness,
                self.kurtosis,
                self.slope,
                self.high_band_ratio,
                self.duration_ms,
            ],
            dtype=np.float64,
        )


def _spectrum(segment: AudioSegment) -> tuple[np.ndarray, np.ndarray]:
    samples = segment.samples.astype(np.float64, copy=False)
    samples = samples - np.mean(samples)
    if len(samples) < 32:
        raise ValueError("segment is too short")

    window = np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(samples * window)) ** 2
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / segment.sample_rate)

    mask = freqs >= 300.0
    return freqs[mask], spectrum[mask]


def extract_fricative_features(segment: AudioSegment) -> FricativeFeatures:
    freqs, power = _spectrum(segment)
    total = float(np.sum(power))
    if total <= 1e-16:
        raise ValueError("segment has no usable spectral energy")

    weights = power / total
    centroid = float(np.sum(freqs * weights))
    centered = freqs - centroid
    variance = float(np.sum((centered ** 2) * weights))
    spread = math.sqrt(max(variance, 0.0))

    if spread > 1e-9:
        skewness = float(np.sum((centered ** 3) * weights) / (spread ** 3))
        kurtosis = float(np.sum((centered ** 4) * weights) / (spread ** 4))
    else:
        skewness = 0.0
        kurtosis = 0.0

    fit_mask = (freqs >= 1000.0) & (freqs <= min(12000.0, segment.sample_rate / 2.0))
    if np.count_nonzero(fit_mask) >= 8:
        x = freqs[fit_mask] / 1000.0
        y = 10.0 * np.log10(power[fit_mask] + 1e-18)
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0

    high_threshold = min(6000.0, segment.sample_rate * 0.22)
    high_band_ratio = float(np.sum(power[freqs >= high_threshold]) / total)
    duration_ms = 1000.0 * len(segment.samples) / segment.sample_rate

    return FricativeFeatures(
        centroid_hz=centroid,
        spread_hz=spread,
        skewness=skewness,
        kurtosis=kurtosis,
        slope=slope,
        high_band_ratio=high_band_ratio,
        duration_ms=duration_ms,
    )
