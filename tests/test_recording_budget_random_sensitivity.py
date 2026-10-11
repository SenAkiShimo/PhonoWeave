"""Repeated-null checks: frozen folds, seed zero, and aggregate counts."""
import math
import pytest

from phonoweave.recording_budget_benchmark import Example, RecordingUnit
from phonoweave.recording_budget_disjoint_cv import run_disjoint_benchmark
from phonoweave.recording_budget_random_sensitivity import evaluate_random_sensitivity

def corpus(n=30):
    return tuple(RecordingUnit(f"wav-{i:03d}",(
        Example(("s","sh","x")[i%3],("plain","rounded")[(i//3)%2],"a",
                tuple((i%7)*0.025 for _ in range(7))),
    )) for i in range(n))

def test_exact_v02_seed_zero_reproduction():
    units=corpus()
    a=run_disjoint_benchmark(units,budgets=(0.25,0.5,1.0))
    r=evaluate_random_sensitivity(units,n_random=4,budgets=(0.25,0.5,1.0))
    for fold in r["fold_results"]:
        for row in fold["budgets"]:
            original=next(x for x in a["rows"] if x["fold"]==fold["fold"]
                          and x["budget_fraction"]==row["budget_fraction"]
                          and x["policy"]=="stratified_random")
            assert row["random_seed0_distance"]==original["penalized_acoustic_distance"]
            original_adaptive=next(x for x in a["rows"] if x["fold"]==fold["fold"]
                          and x["budget_fraction"]==row["budget_fraction"]
                          and x["policy"]=="adaptive_observed")
            assert row["adaptive_distance"]==original_adaptive["penalized_acoustic_distance"]

def test_reproducible_and_count_conservation():
    units=corpus()
    a=evaluate_random_sensitivity(units,n_random=5,budgets=(0.25,0.5,1.0))
    assert a==evaluate_random_sensitivity(units,n_random=5,budgets=(0.25,0.5,1.0))
    assert a["unique_heldout_wavs"]==len(units)
    for row in a["summary"]:
        assert sum(row[k] for k in ("random_repeats_better_than_adaptive",
                                  "random_repeats_equal_to_adaptive",
                                  "random_repeats_worse_than_adaptive"))==5
        assert row["random_macro_p05"]<=row["random_macro_median"]<=row["random_macro_p95"]

def test_invalid_repeat_count():
    with pytest.raises(ValueError):
        evaluate_random_sensitivity(corpus(),n_random=1)
