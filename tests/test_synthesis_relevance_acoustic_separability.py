"""Objective fricative context study regression and independence tests."""
import numpy as np
from phonoweave.synthesis_relevance_acoustic_separability import (
    Observation, _features, grouped_cv, paired_wav_control
)

def data(n=36):
    rows=[]
    for i in range(n):
        wav=f"wav{i:03d}"
        for context in ("plain","rounded"):
            base=2. if context=="rounded" else -2.
            feat=tuple(base+0.01*(i%5)+0.03*j for j in range(9))
            rows.append(Observation(wav,"D4",context,feat))
    return rows

def test_features_finite_for_valid_noise():
    rng=np.random.default_rng(31)
    x=rng.normal(0,0.1,44100)
    features=_features(x,44100,100.,300.)
    assert len(features)==9
    assert all(np.isfinite(features))

def test_grouped_cv_has_no_wav_reuse():
    result=grouped_cv(data())
    assert result["status"]=="evaluated"
    assert result["balanced_accuracy"]>.9
    assert sum(fold["test_wavs"] for fold in result["folds"])==36

def test_paired_signflip_detects_constructed_effect():
    result=paired_wav_control(data(),permutations=199)
    assert result["status"]=="evaluated"
    assert result["paired_wavs"]==36
    assert result["paired_sign_flip_p"]<=0.02

def test_paired_control_does_not_count_aliases_as_independent_wavs():
    rows=data()
    dup=rows[0]
    rows.extend([dup]*10)
    result=paired_wav_control(rows,permutations=99)
    assert result["paired_wavs"]==36

def test_sparse_evidence_explicitly_unresolved():
    assert grouped_cv(data(3))["status"]=="insufficient"
    assert paired_wav_control(data(3))["status"]=="insufficient"
