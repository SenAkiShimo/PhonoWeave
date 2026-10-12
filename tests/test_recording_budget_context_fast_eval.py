"""Numerical equivalence of vectorized exact-distance cache and frozen metric."""
import pytest
from phonoweave.recording_budget_benchmark import Example, RecordingUnit, evaluate_heldout
from phonoweave.recording_budget_context_fast_eval import PreparedHeldoutDistances

def e(base, ctx, value):
    return Example(base, ctx, "a", (value,) * 7)

def test_cached_scores_match_original_for_partial_and_complete_coverage():
    held = (
        RecordingUnit("h1", (e("s","plain",0.1), e("s","plain",0.2), e("x","rounded",0.3))),
        RecordingUnit("h2", (e("sh","plain",0.4), e("s","plain",0.5))),
    )
    pool = (
        RecordingUnit("p1", (e("s","plain",0.12), e("sh","plain",0.45))),
        RecordingUnit("p2", (e("x","rounded",0.29),)),
        RecordingUnit("p3", (e("s","plain",0.41),)),
    )
    cache = PreparedHeldoutDistances({u.digest:u for u in pool}, held)
    for chosen in ((),("p1",),("p2",),("p1","p2"),("p1","p2","p3")):
        expected=evaluate_heldout(tuple(u for u in pool if u.digest in chosen),held)
        actual=cache.score_subset(chosen)
        assert actual["penalized_distance"] == expected["penalized_acoustic_distance"]
        assert actual["matched_only_distance"] == expected["matched_only_acoustic_distance"]
        assert actual["cell_coverage"] == expected["cell_coverage"]
        assert actual["matched_wav_cell_pairs"] == expected["matched_wav_cell_pairs"]
        assert actual["heldout_wav_cell_pairs"] == expected["heldout_wav_cell_pairs"]
        assert abs(
            actual["matched_distance_contribution"] +
            actual["missing_penalty_contribution"] -
            actual["penalized_distance"]
        ) < 0.00002

def test_cached_budget_prefix_matches_direct_scoring():
    held=(RecordingUnit("h",(e("s","plain",0.1),e("x","rounded",0.5))),)
    pool={
        "a": RecordingUnit("a",(e("s","plain",0.2),)),
        "b": RecordingUnit("b",(e("x","rounded",0.48),)),
        "c": RecordingUnit("c",(e("s","plain",0.08),)),
    }
    cache=PreparedHeldoutDistances(pool,held)
    outcomes=cache.score_duration_budgets(
        ("a","b","c"), {"a":1.,"b":8.,"c":1.},
        (0.1,0.5,1.),10.,3
    )
    assert outcomes[0.1]["acquired_wavs"]==1
    assert outcomes[0.5]["acquired_wavs"]==1
    assert outcomes[1.]["acquired_wavs"]==3
    assert outcomes[0.5]["penalized_distance"]==evaluate_heldout((pool["a"],),held)["penalized_acoustic_distance"]
    assert outcomes[1.]["penalized_distance"]==evaluate_heldout(tuple(pool.values()),held)["penalized_acoustic_distance"]
