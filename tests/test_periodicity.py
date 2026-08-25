import numpy as np

from phonoweave.periodicity import normalized_periodicity


def _reference(samples: np.ndarray, sample_rate: int) -> float:
    values = samples.astype(np.float64, copy=False)
    values = values - np.mean(values)
    min_lag = max(1, int(sample_rate / 500.0))
    max_lag = min(len(values) - 2, int(sample_rate / 70.0))
    if max_lag <= min_lag:
        return 0.0

    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        left = values[:-lag]
        right = values[lag:]
        denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
        if denominator > 1e-12:
            best = max(best, float(np.dot(left, right) / denominator))
    return max(0.0, min(best, 1.0))


def test_fft_periodicity_matches_reference() -> None:
    rng = np.random.default_rng(41021)
    for length in (96, 257, 1024, 2048):
        samples = rng.normal(size=length)
        expected = _reference(samples, 48000)
        actual = normalized_periodicity(samples, 48000)
        assert abs(actual - expected) < 1e-10
