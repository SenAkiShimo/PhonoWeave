"""Metadata-only coverage-first selection and penalty decomposition."""
from dataclasses import replace
import pytest

from phonoweave.recording_budget_benchmark import Example, RecordingUnit, metadata, evaluate_heldout
from phonoweave.recording_budget_duration_comparison import evaluate_duration_budget
from phonoweave.recording_budget_context_control import (
    coverage_first_random_sequence, decompose_distance, evaluate_context_control,
)

def example(cell, value):
    base,context=cell.split(":")
    return Example(base,context,"a",(value,)*7)

def corpus(n=30):
    cells=("s:plain","s:rounded","sh:plain","sh:rounded","x:plain","x:rounded")
    return tuple(RecordingUnit(f"wav-{i:03d}",(
        example(cells[i%6],(i%7)*0.025),
    )) for i in range(n))

def test_coverage_priority_and_determinism():
    data=corpus(30)
    a=coverage_first_random_sequence(metadata(data),seed=100)
    assert a==coverage_first_random_sequence(metadata(data),seed=100)
    assert set(a)=={u.digest for u in data}
    by_id={u.digest:u for u in data}
    prefix=set()
    for identifier in a:
        if len(prefix)==6:
            break
        prefix.update(by_id[identifier].cells)
    assert len(prefix)==6

def test_decomposition_missing_and_present():
    held=(RecordingUnit("held",(example("s:plain",0.2),example("x:plain",0.3))),)
    sel=(RecordingUnit("sel",(example("s:plain",0.1),)),)
    r=decompose_distance(sel,held)
    assert r["missing_wav_cell_pairs"]==1
    assert r["matched_wav_cell_pairs"]==1
    assert r["full_context_coverage"] is False
    assert abs(r["missing_penalty_contribution"]-1.)<1e-8
    assert abs(r["matched_distance_contribution"]-0.05)<1e-6
    assert r["penalized_distance"]==evaluate_heldout(sel,held)["penalized_acoustic_distance"]

def test_replays_original_v07_and_enforces_ceiling():
    units=corpus()
    durations={u.digest:2.+i%3 for i,u in enumerate(units)}
    reference=evaluate_duration_budget(units,durations,n_random=3)
    result=evaluate_context_control(units,durations,reference,repeats=3)
    assert result["unique_eligible_wavs"]==30
    assert len(result["folds"])==5
    for fold in result["folds"]:
        for row in fold["budgets"]:
            assert len(row["coverage_first_random"])==3
            for trial in row["coverage_first_random"]:
                assert trial["spent_fraction"]<=row["budget_fraction"]+1e-5
    assert result==evaluate_context_control(units,durations,reference,repeats=3)

def test_rejects_drifted_original():
    units=corpus()
    durations={u.digest:2.+i%3 for i,u in enumerate(units)}
    reference=evaluate_duration_budget(units,durations,n_random=3)
    reference["fold_reports"][0]["budgets"][0]["policies"]["adaptive_observed"]["heldout_penalized_distance"] += 0.1
    with pytest.raises(ValueError,match="replay mismatch"):
        evaluate_context_control(units,durations,reference,repeats=3)
