import numpy as np

from phonoweave.calibration_periodicity import overlap_safe_periodicity, periodicity_track
from phonoweave.calibration_periodicity_repair import assess_frame_gates


def test_unvoiced_white_noise_not_fake_periodic() -> None:
    rng = np.random.default_rng(42)
    sr = 16000
    noise = rng.standard_normal(int(0.03 * sr))
    assert overlap_safe_periodicity(noise, sr) < .5


def test_voiced_periodic_sine_retains_high_correlation() -> None:
    sr = 16000
    t = np.arange(int(0.03 * sr)) / sr
    voiced = np.sin(2 * np.pi * 200 * t)
    assert overlap_safe_periodicity(voiced, sr) > .9


def test_silence_conservatively_vetoed() -> None:
    assert overlap_safe_periodicity(np.zeros(480), 16000) == 1.0


def test_track_is_finite() -> None:
    sr = 16000
    rng = np.random.default_rng(7)
    audio = rng.standard_normal(16000)
    centers = np.arange(100, 500, 12, dtype=float)
    scores = periodicity_track(audio, sr, centers)
    assert len(scores) == len(centers)
    assert np.all(np.isfinite(scores))


def test_report_keeps_rms_ablation_out_of_acceptance() -> None:
    n = 70
    f = {
        "centers": np.arange(n, dtype=float) * 4 + 7,
        "high_db": np.full(n, -85.0),
        "rms_db": np.full(n, -65.0),
        "flatness": np.full(n, .01),
        "zcr": np.full(n, .01),
        "periodicity": np.full(n, .9),
    }
    f["high_db"][30:50] = -50
    f["flatness"][30:50] = .35
    f["zcr"][30:50] = .23
    f["periodicity"][30:50] = .17
    v = assess_frame_gates(f, 0.)
    assert v["strict_longest_run"] == 0
    assert v["no_rms_longest_run_diagnostic_only"] >= 3
    assert v["onset_ms_after_cue"] is None
