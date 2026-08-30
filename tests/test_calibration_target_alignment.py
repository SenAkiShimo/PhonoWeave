from __future__ import annotations

import numpy as np

from phonoweave.calibration_target_alignment import (
    _detect_noise_anchor,
    _detect_sonorant_anchor,
    _suppress_beeps,
)
from phonoweave.calibration_beep_alignment import BeepEvent


def _sine(sample_rate: int, duration_ms: float, frequency: float, amplitude: float) -> np.ndarray:
    n = int(round(sample_rate * duration_ms / 1000.0))
    t = np.arange(n, dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * frequency * t)


def test_beep_suppression_reduces_target_tone() -> None:
    sample_rate = 16000
    samples = np.zeros(sample_rate, dtype=np.float64)
    start = int(round(0.500 * sample_rate))
    tone = _sine(sample_rate, 100.0, 700.0, 0.4)
    samples[start : start + len(tone)] += tone
    event = BeepEvent(
        index=0,
        role="token_0",
        expected_frequency_hz=700.0,
        time_ms=500.0,
        peak_time_ms=530.0,
        tone_score=1.0,
        competing_score=0.0,
    )
    cleaned = _suppress_beeps(samples, sample_rate, (event,))
    before = float(np.sqrt(np.mean(samples[start : start + len(tone)] ** 2)))
    after = float(np.sqrt(np.mean(cleaned[start : start + len(tone)] ** 2)))
    assert after < before * 0.45


def test_noise_anchor_finds_sustained_frication_after_cue() -> None:
    rng = np.random.default_rng(5)
    sample_rate = 16000
    samples = np.zeros(int(sample_rate * 1.3), dtype=np.float64)
    cue_ms = 250.0
    onset_ms = 430.0
    start = int(round(onset_ms * sample_rate / 1000.0))
    noise = rng.normal(0.0, 0.08, int(round(0.24 * sample_rate)))
    noise = np.diff(np.concatenate([[0.0], noise]))
    samples[start : start + len(noise)] += noise
    anchor_ms, strength = _detect_noise_anchor(samples, sample_rate, cue_ms, "fricative")
    assert abs(anchor_ms - onset_ms) < 45.0
    assert np.isfinite(strength)


def test_sonorant_anchor_finds_voiced_region_after_cue() -> None:
    sample_rate = 16000
    samples = np.zeros(int(sample_rate * 1.3), dtype=np.float64)
    cue_ms = 250.0
    onset_ms = 455.0
    start = int(round(onset_ms * sample_rate / 1000.0))
    voiced = _sine(sample_rate, 250.0, 180.0, 0.14)
    voiced += 0.04 * _sine(sample_rate, 250.0, 360.0, 1.0)
    samples[start : start + len(voiced)] += voiced
    anchor_ms, strength = _detect_sonorant_anchor(samples, sample_rate, cue_ms, "nasal")
    assert abs(anchor_ms - onset_ms) < 55.0
    assert np.isfinite(strength)
