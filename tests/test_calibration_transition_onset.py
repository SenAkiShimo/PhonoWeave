from __future__ import annotations
import numpy as np
from phonoweave.calibration_transition_onset import acoustic_tracks, detect_transition, nominal_target_cue_ms

def test_nominal_schedule():
    assert abs(nominal_target_cue_ms(0) - 1083.333333) < .001
    assert abs(nominal_target_cue_ms(2) - 2750.) < .001

def test_synthetic_sustained_frication():
    rng = np.random.default_rng(21)
    sr = 16000
    y = rng.normal(0, 1e-5, sr*3)
    y[int(1.2*sr):int(1.4*sr)] += rng.normal(0,.05,int(.2*sr))
    f = acoustic_tracks(y,sr)
    r = detect_transition(f,1083.333)
    assert r.status == "candidate"
    assert 1170 < r.onset_ms < 1230

def test_short_click_is_not_accepted():
    rng = np.random.default_rng(22)
    sr = 16000
    y = rng.normal(0,1e-5,sr*3)
    y[int(1.2*sr):int(1.203*sr)] += rng.normal(0,.08,int(.003*sr))
    r = detect_transition(acoustic_tracks(y,sr),1083.333)
    assert r.onset_ms is None
