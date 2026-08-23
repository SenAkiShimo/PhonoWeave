import numpy as np

from phonoweave.audio import AudioSegment
from phonoweave.rhotic_relevance import rhotic_boundary_penalty


def _segment(samples: np.ndarray, sample_rate: int = 48000) -> AudioSegment:
    return AudioSegment(
        samples=samples.astype(np.float64),
        sample_rate=sample_rate,
        start_ms=0.0,
        end_ms=1000.0 * len(samples) / sample_rate,
    )


def test_identical_rhotic_edges_have_near_zero_penalty():
    sample_rate = 48000
    time = np.arange(2400) / sample_rate
    samples = 0.5 * np.sin(2.0 * np.pi * 180.0 * time)
    segment = _segment(samples, sample_rate)
    assert rhotic_boundary_penalty(segment, segment) < 1e-10


def test_different_rhotic_edges_increase_penalty():
    sample_rate = 48000
    time = np.arange(2400) / sample_rate
    left = _segment(0.5 * np.sin(2.0 * np.pi * 180.0 * time), sample_rate)
    right = _segment(
        0.35 * np.sin(2.0 * np.pi * 900.0 * time)
        + 0.15 * np.sin(2.0 * np.pi * 1700.0 * time),
        sample_rate,
    )
    assert rhotic_boundary_penalty(left, right) > 0.1
