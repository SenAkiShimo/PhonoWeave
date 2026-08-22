import numpy as np

from phonoweave.audio import AudioSegment
from phonoweave.features import extract_fricative_features


def test_fricative_features_are_finite() -> None:
    sample_rate = 48000
    time = np.arange(int(sample_rate * 0.08)) / sample_rate
    samples = 0.4 * np.sin(2 * np.pi * 3500 * time) + 0.2 * np.sin(2 * np.pi * 8000 * time)
    segment = AudioSegment(samples=samples, sample_rate=sample_rate, start_ms=0.0, end_ms=80.0)

    features = extract_fricative_features(segment)
    assert np.all(np.isfinite(features.vector()))
    assert features.centroid_hz > 1000
    assert 0 <= features.high_band_ratio <= 1
