"""Periodicity estimator with a minimum overlap, used for calibration experiments.

Unlike the legacy 14 ms lag search, this refuses near-empty autocorrelation
overlap. It uses a longer independent window so low-F0 voiced frames can still
be identified. It does not modify the historic v0.6 feature extractor.
"""
from __future__ import annotations

import numpy as np


def overlap_safe_periodicity(
    samples: np.ndarray,
    sample_rate: int,
    *,
    min_f0_hz: float = 70.0,
    max_f0_hz: float = 500.0,
    min_overlap_fraction: float = 0.5,
) -> float:
    """Peak normalized positive autocorrelation with substantial overlap.

    Returns 1.0 for silent / unusable frames as a *conservative veto*, NOT
    evidence of periodic voice. The classifier also requires nontrivial
    noise energy and other independent acoustic conditions.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or sample_rate <= 0:
        raise ValueError("expected mono frame and positive sample rate")
    if x.size < 32:
        return 1.0
    x = x - np.mean(x)
    if float(np.dot(x, x)) < 1e-10:
        return 1.0

    minimum = max(1, int(sample_rate / max_f0_hz))
    # At a huge lag, correlating 2-3 points is statistically meaningless.
    maximum = min(
        int(sample_rate / min_f0_hz),
        x.size - max(32, int(np.ceil(x.size * min_overlap_fraction))),
    )
    if maximum <= minimum:
        return 1.0

    best = 0.0
    for lag in range(minimum, maximum + 1):
        first, second = x[:-lag], x[lag:]
        denom = float(np.sqrt(np.dot(first, first) * np.dot(second, second)))
        if denom > 1e-12:
            best = max(best, float(np.dot(first, second) / denom))
    return float(np.clip(best, 0.0, 1.0))


def periodicity_track(
    samples: np.ndarray,
    sample_rate: int,
    centers_ms: np.ndarray,
    *,
    window_ms: float = 30.0,
) -> np.ndarray:
    """Conservative periodicity gate at the centers of the short feature frames."""
    length = max(32, int(round(window_ms * sample_rate / 1000.0)))
    result = np.ones(len(centers_ms), dtype=np.float64)
    for i, center_ms in enumerate(centers_ms):
        middle = int(round(float(center_ms) * sample_rate / 1000.0))
        start = middle - length // 2
        end = start + length
        if start < 0 or end > len(samples):
            continue
        result[i] = overlap_safe_periodicity(samples[start:end], sample_rate)
    return result
