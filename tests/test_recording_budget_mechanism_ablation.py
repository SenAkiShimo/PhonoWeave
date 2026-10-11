"""Mechanism ablation invariants and frozen-selector reproduction."""
import pytest

from phonoweave.recording_budget_benchmark import (
    Example, RecordingUnit, metadata, select_sequence,
)
from phonoweave.recording_budget_disjoint_cv import run_disjoint_benchmark
from phonoweave.recording_budget_mechanism_ablation import (
    ABLATIONS, ablated_sequence, evaluate_mechanism_ablations,
)

def corpus(n=30):
    return tuple(RecordingUnit(
        f"wav-{i:03d}",
        (Example(("s","sh","x")[i%3],("plain","rounded")[(i//3)%2],
                 ("a","u","i")[i%3],
                 tuple((i%7)*0.025 for _ in range(7))),),
    ) for i in range(n))

def test_full_and_metadata_only_exact_original():
    units=corpus()
    lookup={u.digest:u for u in units}
    c=metadata(units)
    acquire=lambda digest: lookup[digest].examples
    for seed in (42, 20261011):
        assert ablated_sequence(c,acquire,variant="full_adaptive",seed=seed) == (
            select_sequence(c,acquire,policy="adaptive_observed",seed=seed))
        assert ablated_sequence(c,acquire,variant="metadata_balancing_only",seed=seed) == (
            select_sequence(c,acquire,policy="fixed_balanced",seed=seed))

def test_v02_folds_and_full_scores_unchanged():
    units=corpus()
    v02=run_disjoint_benchmark(units,budgets=(0.25,0.5,1.0))
    result=evaluate_mechanism_ablations(units,budgets=(0.25,0.5,1.0))
    assert len(result["rows"])==5*3*len(ABLATIONS)
    for row in result["rows"]:
        if row["variant"]!="full_adaptive":
            continue
        original=next(r for r in v02["rows"] if
                      r["fold"]==row["fold"] and
                      r["budget_fraction"]==row["budget_fraction"] and
                      r["policy"]=="adaptive_observed")
        assert row["distance"]==original["penalized_acoustic_distance"]
    assert result==evaluate_mechanism_ablations(units,budgets=(0.25,0.5,1.0))

def test_invalid_variant_rejected():
    with pytest.raises(ValueError,match="unsupported"):
        ablated_sequence(metadata(corpus()),lambda x: (),variant="oracle",seed=1)
