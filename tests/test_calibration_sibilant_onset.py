from __future__ import annotations

import numpy as np

from phonoweave.calibration_sibilant_onset import select_sibilant_onset


def _tracks() -> dict[str, np.ndarray]:
    length = 160
    values = {
        "centers": np.arange(length, dtype=float) * 4.0 - 140.0 + 7.0,
        "high_db": np.full(length, -85.0),
        "rms_db": np.full(length, -65.0),
        "flatness": np.full(length, 0.12),
        "zcr": np.full(length, 0.02),
        "periodicity": np.full(length, 0.95),
    }
    return values


def test_sustained_frication_onset() -> None:
    f = _tracks()
    start = 65  # onset at +120 ms
    stop = 105
    for key, level in {
        "high_db": -50.0,
        "rms_db": -32.0,
        "flatness": 0.40,
        "zcr": 0.20,
        "periodicity": 0.10,
    }.items():
        f[key][start:stop] = level
    onset = select_sibilant_onset(f, 0.0)
    assert onset.status == "candidate"
    assert abs(onset.onset_ms_after_cue - 120.0) <= 4.0


def test_tonal_beep_and_vowel_not_mistaken_as_sibilant() -> None:
    f = _tracks()
    for key, level in {
        "high_db": -44.0,
        "rms_db": -22.0,
        "flatness": 0.015,
        "zcr": 0.03,
        "periodicity": 0.98,
    }.items():
        f[key][40:53] = level  # tonal cue residual
    for key, level in {
        "high_db": -62.0,
        "rms_db": -18.0,
        "flatness": 0.035,
        "zcr": 0.04,
        "periodicity": 0.92,
    }.items():
        f[key][90:135] = level  # voiced nucleus
    guess = select_sibilant_onset(f, 0.0)
    assert guess.status == "unresolved"
    assert guess.onset_ms_after_cue is None


def test_near_boundary_returns_review() -> None:
    f = _tracks()
    for key, level in {
        "high_db": -51.0,
        "rms_db": -30.0,
        "flatness": 0.37,
        "zcr": 0.20,
        "periodicity": 0.08,
    }.items():
        f[key][2:22] = level
    assert select_sibilant_onset(f, 0.0).status == "review"


def test_nonfinite_features_fail_closed() -> None:
    f = _tracks()
    f["high_db"][10] = np.nan
    assert select_sibilant_onset(f, 0.0).status == "unresolved"
