"""Frozen WAV-budget cost audit: grouping and count/duration invariants."""
import pytest

from phonoweave.recording_budget_benchmark import Example, RecordingUnit
from phonoweave.recording_budget_cost_audit import audit_recording_costs

def sample_units():
    units=[]
    for i in range(25):
        examples=tuple(
            Example(("s","sh","x")[i%3],("plain","rounded")[(i//3)%2],
                    "a",tuple((i%7)*0.025 for _ in range(7)))
            for _ in range(1+(i%4))
        )
        units.append(RecordingUnit(f"wav-{i:03d}",examples))
    return tuple(units)

def test_cost_audit_is_stable_and_has_unity_at_full_budget():
    units=sample_units()
    seconds={u.digest: float(1+(i%5)) for i,u in enumerate(units)}
    a=audit_recording_costs(units,seconds)
    assert a==audit_recording_costs(units,seconds)
    assert a["eligible_oto_acoustic_segments"]==sum(len(u.examples) for u in units)
    assert a["total_eligible_wav_duration_seconds"]==sum(seconds.values())
    for r in a["summary"]:
        assert 0 < r["mean_duration_fraction"] <= 1
        assert 0 < r["mean_segment_fraction"] <= 1
        if r["budget_fraction"]==1.0:
            assert r["mean_duration_fraction"]==1.
            assert r["mean_segment_fraction"]==1.

def test_rejects_missing_durations():
    units=sample_units()
    durations={unit.digest:1. for unit in units[:-1]}
    with pytest.raises(ValueError,match="duration index"):
        audit_recording_costs(units,durations)
