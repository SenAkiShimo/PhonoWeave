from __future__ import annotations

import numpy as np
from phonoweave.calibration_cv_transition import cv_tracks, detect_cv_boundary
from phonoweave.calibration_dual_eval import compare_dual_rows


def test_cv_transition_from_broadband_noise_to_voiced_vowel() -> None:
    sr = 16000
    rng = np.random.default_rng(42)
    y = rng.normal(0, 1e-6, sr*3)
    begin, end = int(1.1*sr), int(1.28*sr)
    y[begin:end] += rng.normal(0, .08, end-begin)
    t = np.arange(int(.4*sr)) / sr
    y[end:end+len(t)] += .25*np.sin(2*np.pi*200*t)
    features = cv_tracks(y, sr)
    result = detect_cv_boundary(features, 1090.0)
    assert result.status == "candidate"
    assert 1250 <= result.absolute_ms <= 1310


def test_no_vowel_transition_returns_unresolved() -> None:
    sr = 16000
    rng = np.random.default_rng(9)
    y = rng.normal(0, 1e-6, sr*3)
    y[int(1.1*sr):int(1.5*sr)] += rng.normal(0, .05, int(.4*sr))
    result = detect_cv_boundary(cv_tracks(y, sr), 1090.0)
    assert result.absolute_ms is None


def test_metrics_preserve_provenance_and_anchor_kind() -> None:
    rows = [
        {"anchor_type":"consonant_onset","manual_status":"ok","outside_expected_slot":False,
         "manual_source":"manual_v0.3","error_ms":6.0},
        {"anchor_type":"cv_boundary","manual_status":"ok","outside_expected_slot":False,
         "manual_source":"legacy_reviewed_copy","error_ms":-10.0},
        {"anchor_type":"cv_boundary","manual_status":"uncertain","outside_expected_slot":False,
         "manual_source":"manual_v0.3","error_ms":None},
    ]
    stats = compare_dual_rows(rows)
    assert stats["consonant_onset"]["mae_ms"] == 6.0
    assert stats["cv_boundary_legacy_import"]["mae_ms"] == 10.0
    assert stats["cv_boundary_manually_placed"]["eligible"] == 0
    assert stats["cv_boundary_all_provisional"]["matched"] == 1
