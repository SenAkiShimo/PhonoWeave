"""Disjoint fold coverage, repeatability and selection-budget invariants."""
import pytest
from phonoweave.recording_budget_benchmark import Example, RecordingUnit
from phonoweave.recording_budget_disjoint_cv import disjoint_folds, run_disjoint_benchmark

def corpus(n=68):
    units=[]
    for i in range(n):
        base=("s","sh","x")[i%3]
        context=("plain","rounded")[(i//3)%2]
        units.append(RecordingUnit(
            digest=f"wav-{i:04d}",
            examples=(Example(base,context,"a",tuple(float(i%7)*0.02 for _ in range(7))),)))
    return tuple(units)

def test_all_wavs_held_out_once_and_balanced():
    data=corpus()
    folds=disjoint_folds(data)
    assert sorted(len(group) for group in folds)==[13,13,14,14,14]
    ids=[u.digest for group in folds for u in group]
    assert len(ids)==len(set(ids))==68
    assert set(ids)=={u.digest for u in data}
    assert folds==disjoint_folds(data)

def test_partition_ignores_acoustic_vectors():
    original=corpus()
    modified=tuple(RecordingUnit(u.digest,tuple(Example(
        e.base,e.context,e.final,tuple(1.5 for _ in e.features)
    ) for e in u.examples)) for u in original)
    assert [[x.digest for x in fold] for fold in disjoint_folds(original)] == [
        [x.digest for x in fold] for fold in disjoint_folds(modified)]

def test_report_equal_wav_budgets_and_zero_test_reuse():
    report=run_disjoint_benchmark(corpus(),budgets=(0.25,0.50,1.0))
    assert report["partition"]["all_unique_wavs_tested_exactly_once"]
    assert report["partition"]["distinct_heldout_wavs"]==68
    assert report["partition"]["test_reuse_count"]==0
    assert len(report["rows"])==45
    for fold in range(5):
        for budget in (0.25,0.50,1.0):
            rows=[row for row in report["rows"] if row["fold"]==fold and row["budget_fraction"]==budget]
            assert len(rows)==3
            assert len({row["budget_wav"] for row in rows})==1
    assert report==run_disjoint_benchmark(corpus(),budgets=(0.25,0.50,1.0))

def test_duplicate_groups_rejected():
    data=corpus()
    with pytest.raises(ValueError,match="duplicate"):
        disjoint_folds(data+(data[0],))
