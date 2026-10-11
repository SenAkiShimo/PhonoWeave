import numpy as np
from phonoweave.calibration_excess_fallback import detect_excess_fallback
from phonoweave.calibration_transition_onset import acoustic_tracks
from phonoweave.calibration_cv_transition import cv_tracks

def test_no_frication_or_vowel_does_not_create_paired_candidate():
    y = np.zeros(48000)
    r = detect_excess_fallback(acoustic_tracks(y,16000),cv_tracks(y,16000),1250.0)
    assert r.status != "paired_candidate"

def test_noise_without_vowel_cannot_form_valid_paired_candidate():
    rng=np.random.default_rng(17)
    y=rng.normal(0,.02,48000)
    r=detect_excess_fallback(acoustic_tracks(y,16000),cv_tracks(y,16000),1250.0)
    assert r.status != "paired_candidate"

def test_strict_success_is_preserved():
    rng = np.random.default_rng(39)
    sr=16000
    y=rng.normal(0,1.e-6,3*sr)
    lo,mid=int(sr*1.2),int(sr*1.38)
    y[lo:mid]+=rng.normal(0,.08,mid-lo)
    t=np.arange(int(.4*sr))/sr
    y[mid:mid+len(t)]+=.24*np.sin(2*np.pi*180*t)
    t1=acoustic_tracks(y,sr)
    t2=cv_tracks(y,sr)
    r=detect_excess_fallback(t1,t2,1083.333)
    assert r.status=="paired_candidate"
    assert r.method=="strict_v2.1"
