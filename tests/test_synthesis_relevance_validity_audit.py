"""Regression for extraction diagnostics and group-preserving negative controls."""
from pathlib import Path
import numpy as np
from phonoweave.synthesis_relevance_acoustic_separability import Observation, grouped_cv
from phonoweave.synthesis_relevance_validity_audit import (
    _feature_failure,_duration_bin,_folds,_score_labels,
    wav_group_label_null, leave_one_subbank_out, audit_extraction,
)

def examples(n=36,separation=2.0):
    out=[]
    for i in range(n):
        for ctx in ("plain","rounded"):
            z=(separation if ctx=="rounded" else -separation)+0.02*(i%5)
            out.append(Observation(f"wav-{i:03d}","D4" if i%2 else "D5",ctx,tuple(z+0.04*k for k in range(9))))
    return out

def test_exact_grouped_baseline_replay():
    rows=examples()
    x=np.asarray([row.features for row in rows])
    y=np.asarray([row.context=="rounded" for row in rows],dtype=int)
    from_audit=_score_labels(x,y,_folds(rows))
    baseline=grouped_cv(rows)["balanced_accuracy"]
    assert abs(from_audit-baseline)<1e-6

def test_group_label_null_separates_true_signal():
    out=wav_group_label_null(examples(),permutations=79)
    assert out["status"]=="evaluated"
    assert out["observed_balanced_accuracy"]>=0.95
    assert out["permutation_p_upper_tail"]<=0.05
    assert out["valid_null_trials"]==79

def test_short_interval_is_explicitly_categorized():
    x=np.zeros(44100)
    assert _duration_bin(35)=="0_to_40ms"
    assert _feature_failure(x,44100,0.,70.,0.)=="oto_interval_shorter_than_110ms_after_trim"
    assert _feature_failure(x,44100,0.,150.,50.)=="oto_interval_shorter_than_110ms_after_trim"

def test_pitch_subbank_holdout_and_empty_bank(tmp_path:Path):
    result=leave_one_subbank_out(examples())
    assert result["subbanks_evaluable"]==2
    report=audit_extraction(tmp_path)
    assert report["v05_replay"]["matches"]
    assert report["v05_replay"]["valid_events"]==0
