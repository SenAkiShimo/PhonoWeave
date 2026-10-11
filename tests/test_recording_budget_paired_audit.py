"""Paired fold audit invariants with a deterministic synthetic benchmark."""
import copy
import pytest
from phonoweave.recording_budget_benchmark import (
    RecordingUnit, Example, run_budget_benchmark,
)
from phonoweave.recording_budget_paired_audit import analyze_budget_report

def fixture_report():
    units=tuple(
        RecordingUnit(f"hash-{i:03d}",(
            Example(("s","sh","x")[i%3],("plain","rounded")[(i//3)%2],"a",tuple(0.05*i for _ in range(7))),
        )) for i in range(24)
    )
    return run_budget_benchmark(units,split_seeds=(1731,2087),budgets=(0.25,1.0))

def test_paired_folds_and_overlap():
    result=analyze_budget_report(fixture_report())
    assert result["folds"]==2
    assert len(result["comparisons"])==4
    assert len(result["test_overlap"]["pairwise"])==1
    assert all(r["adaptive_wins"]+r["adaptive_losses"]+r["ties"]==2 for r in result["comparisons"])

def test_missing_policy_rejected():
    payload=fixture_report()
    payload["rows"]=[r for r in payload["rows"] if not (r["fold"]==0 and r["policy"]=="fixed_balanced")]
    with pytest.raises(ValueError,match="incomplete policy"):
        analyze_budget_report(payload)

def test_train_test_overlap_rejected():
    payload=fixture_report()
    payload["splits"][0]["heldout_sha256"][0]=payload["splits"][0]["training_sha256"][0]
    with pytest.raises(ValueError,match="leakage"):
        analyze_budget_report(payload)

def test_duplicate_policy_rejected():
    payload=fixture_report()
    payload["rows"].append(copy.deepcopy(payload["rows"][0]))
    with pytest.raises(ValueError,match="duplicate fold"):
        analyze_budget_report(payload)
