"""Deterministic and leakage-resistant benchmark invariants."""
from dataclasses import replace
import pytest

from phonoweave.recording_budget_benchmark import (
    Example, RecordingUnit, acoustic_distance, evaluate_heldout, metadata,
    run_budget_benchmark, select_sequence, split_groups,
)

def fixture_units():
    result=[]
    for i in range(24):
        base=("s","sh","x")[i%3]
        context=("plain","rounded")[(i//3)%2]
        group=(i//6)%4
        vector=tuple((0.1*(i%4)+0.02*group) for _ in range(7))
        result.append(RecordingUnit(f"hash-{i:03d}",
                      (Example(base,context,"a",vector),)))
    return tuple(result)

def test_split_is_wav_disjoint_and_repeatable():
    units=fixture_units()
    t,h=split_groups(units,1731)
    t2,h2=split_groups(units,1731)
    assert [x.digest for x in h]==[x.digest for x in h2]
    assert len(h)==6
    assert set(x.digest for x in t).isdisjoint(x.digest for x in h)

def test_selection_only_observes_acquired_features():
    units=fixture_units()[:10]
    candidates=metadata(units)
    observed_a=[]
    observed_b=[]
    d={u.digest:u.examples for u in units}
    def a(k):
        observed_a.append(k)
        return d[k]
    def b(k):
        observed_b.append(k)
        if k in observed_b[:-1]:
            raise AssertionError("candidate acquired twice")
        return d[k]
    x=select_sequence(candidates,a,policy="adaptive_observed",seed=123)
    y=select_sequence(candidates,b,policy="adaptive_observed",seed=123)
    assert x==y
    assert observed_a==list(x)==observed_b

def test_no_heldout_features_passed_into_selection():
    units=fixture_units()
    train,held=split_groups(units,1731)
    ids={x.digest:x.examples for x in train}
    seq=select_sequence(metadata(train),ids.__getitem__,policy="adaptive_observed",seed=42)
    assert set(seq)==set(ids)
    assert not set(seq).intersection(x.digest for x in held)

def test_penalty_is_separate_from_matched_only():
    selected=(RecordingUnit("a",(Example("x","plain","i",(0.,)*7),)),)
    held=(RecordingUnit("b",(Example("x","rounded","v",(1.,)*7),)),)
    m=evaluate_heldout(selected,held)
    assert m["cell_coverage"]==0.
    assert m["matched_only_acoustic_distance"] is None
    assert m["penalized_acoustic_distance"]==2.

def test_full_budget_and_reproducible_report():
    units=fixture_units()
    r=run_budget_benchmark(units,split_seeds=(1731,2087),budgets=(0.25,1.0))
    again=run_budget_benchmark(units,split_seeds=(1731,2087),budgets=(0.25,1.0))
    assert r==again
    assert len(r["rows"])==12
    assert len(r["splits"])==2
    for row in r["rows"]:
        assert 0<=row["cell_coverage"]<=1
        assert row["budget_wav"]<=row["train_available_wav"]

def test_duplicate_wav_group_rejected():
    u=fixture_units()
    with pytest.raises(ValueError,match="duplicate"):
        split_groups(u+(u[0],),1731)
