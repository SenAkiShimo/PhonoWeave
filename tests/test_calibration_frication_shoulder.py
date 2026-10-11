from __future__ import annotations
import numpy as np
from phonoweave.calibration_transition_onset import acoustic_tracks
from phonoweave.calibration_frication_shoulder import detect_frication_shoulder

def test_weak_shoulder_precedes_sustained_core():
    rng=np.random.default_rng(24)
    sr=16000
    y=rng.normal(0,1e-6,sr*3)
    # Quiet lead-in onsets before the louder frication core.
    y[int(1.17*sr):int(1.21*sr)] += rng.normal(0,0.003,int(.04*sr))
    y[int(1.21*sr):int(1.39*sr)] += rng.normal(0,0.08,int(.18*sr))
    t=acoustic_tracks(y,sr)
    r=detect_frication_shoulder(t,1083.333)
    assert r.core_onset_ms is not None
    assert r.shoulder_onset_ms is not None
    assert r.shoulder_onset_ms <= r.core_onset_ms

def test_silence_never_produces_onset():
    r=detect_frication_shoulder(acoustic_tracks(np.zeros(48000),16000),1083.333)
    assert r.shoulder_onset_ms is None
