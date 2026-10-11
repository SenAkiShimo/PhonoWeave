import numpy as np
from phonoweave.calibration_cv_consistency import classify_syllable
from phonoweave.calibration_transition_onset import acoustic_tracks
from phonoweave.calibration_cv_transition import cv_tracks

def test_breath_like_noise_requires_cv_confirmation():
    sr=16000; rng=np.random.default_rng(8)
    y=rng.normal(0,1e-6,3*sr)
    y[int(1.35*sr):int(1.95*sr)]+=rng.normal(0,.06,int(.6*sr))
    r=classify_syllable(acoustic_tracks(y,sr),cv_tracks(y,sr),1250.)
    assert r.status!="paired_candidate"

def test_sustained_frication_then_vowel_can_be_paired():
    sr=16000;rng=np.random.default_rng(42)
    y=rng.normal(0,1e-6,3*sr)
    i,j=int(1.2*sr),int(1.38*sr)
    y[i:j]+=rng.normal(0,.08,j-i)
    t=np.arange(int(.45*sr))/sr
    y[j:j+len(t)]+=.25*np.sin(2*np.pi*200*t)
    r=classify_syllable(acoustic_tracks(y,sr),cv_tracks(y,sr),1083.333)
    assert r.status=="paired_candidate"
    assert r.consonant_onset_ms < r.cv_boundary_ms
