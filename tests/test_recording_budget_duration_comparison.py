"""Duration-budget conservation, policy freezing and no-backfill invariants."""
import pytest

from phonoweave.recording_budget_benchmark import (
    Example, RecordingUnit,
)
from phonoweave.recording_budget_duration_comparison import (
    duration_limited_prefix, evaluate_duration_budget,
)

def units(n=25):
    return tuple(
        RecordingUnit(f"unit{i:02d}",(
            Example(("s","sh","x")[i%3],("plain","rounded")[(i//3)%2],
                    "a",tuple((i%7)*0.02 for _ in range(7))),
        )) for i in range(n)
    )

def test_stop_on_first_blocked_wav_and_never_backfill():
    order=("short","long","another-short")
    chosen,seconds,blocked=duration_limited_prefix(
        order,{"short":2.,"long":8.,"another-short":1.},6.)
    assert chosen==("short",)
    assert seconds==2.
    assert blocked==8.

def test_exhaustive_at_full_budget_and_seed_zero_random_replay():
    data=units()
    seconds={u.digest:float(1+i%4) for i,u in enumerate(data)}
    report=evaluate_duration_budget(data,seconds,n_random=3,budgets=(0.25,0.5,1.0))
    assert len(report["fold_reports"])==5
    for fold in report["fold_reports"]:
        for row in fold["budgets"]:
            assert row["random_repetitions"][0]==row["policies"]["stratified_random"]
            for policy in row["policies"].values():
                assert policy["spent_seconds"]<=row["budget_ceiling_seconds"]+1e-5
            if row["budget_fraction"]==1.0:
                assert all(p["acquired_wavs"]==fold["train_unique_wavs"] for p in row["policies"].values())
    assert report==evaluate_duration_budget(data,seconds,n_random=3,budgets=(0.25,0.5,1.0))

def test_variable_durations_change_budget_count_without_overrun():
    data=units()
    seconds={u.digest:(100.0 if i%5==0 else 1.0) for i,u in enumerate(data)}
    report=evaluate_duration_budget(data,seconds,n_random=3,budgets=(0.25,0.5,0.75,1.0))
    for fold in report["fold_reports"]:
        for row in fold["budgets"]:
            for value in row["policies"].values():
                assert value["spent_seconds"]<=row["budget_ceiling_seconds"]+1e-5
                if row["budget_fraction"]<1.0:
                    assert value["acquired_wavs"]<=fold["train_unique_wavs"]

@pytest.mark.parametrize("seconds,limit",[
    ({"a":-1.},1.),({"a":float("inf")},1.),({"a":1.},-1.)
])
def test_invalid_duration(seconds,limit):
    with pytest.raises(ValueError):
        duration_limited_prefix(("a",),seconds,limit)
