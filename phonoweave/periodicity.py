from __future__ import annotations

import numpy as np


def normalized_periodicity(
    samples: np.ndarray,
    sample_rate: int,
    min_hz: float = 70.0,
    max_hz: float = 500.0,
) -> float:
    values = samples.astype(np.float64, copy=False)
    values = values - np.mean(values)
    count = len(values)
    min_lag = max(1, int(sample_rate / max_hz))
    max_lag = min(count - 2, int(sample_rate / min_hz))
    if max_lag <= min_lag:
        return 0.0

    fft_size = 1 << (2 * count - 1).bit_length()
    spectrum = np.fft.rfft(values, n=fft_size)
    autocorrelation = np.fft.irfft(
        spectrum * np.conj(spectrum),
        n=fft_size,
    )[: max_lag + 1]

    squared = values * values
    cumulative = np.empty(count + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(squared, out=cumulative[1:])

    lags = np.arange(min_lag, max_lag + 1, dtype=np.int64)
    left_energy = cumulative[count - lags]
    right_energy = cumulative[count] - cumulative[lags]
    denominator = np.sqrt(np.maximum(left_energy * right_energy, 0.0))

    valid = denominator > 1e-12
    if not np.any(valid):
        return 0.0

    scores = autocorrelation[lags][valid] / denominator[valid]
    best = float(np.max(scores))
    return max(0.0, min(best, 1.0))
