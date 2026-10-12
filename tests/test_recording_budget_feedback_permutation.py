"""Permutation preserves acoustic marginals; true feedback reproduces frozen order."""
from collections import Counter

from phonoweave.recording_budget_benchmark import (
    Example, RecordingUnit, metadata, select_sequence,
)
from phonoweave.recording_budget_duration_comparison import evaluate_duration_budget
from phonoweave.recording_budget_sham_feedback import (
    incremental_adaptive_sequence, permuted_training_feedback,
)
from phonoweave.recording_budget_context_control import evaluate_context_control
from phonoweave.recording_budget_feedback_permutation import evaluate_sham_feedback

def corpus(n=30):
    cells=(("s","plain"),("s","rounded"),("sh","plain"),
           ("sh","rounded"),("x","plain"),("x","rounded"))
    return tuple(RecordingUnit(
        f"wav-{i:03d}",tuple(Example(
            *cells[i%6],"a",(0.01*(i%7+j),)*7
        ) for j in range(1+(i%3)))
    ) for i in range(n))

def test_feedback_marginals_and_metadata_invariant():
    units=corpus()
    feedback,stats=permuted_training_feedback(units,seed=11)
    original=Counter((e.cell,e.features) for u in units for e in u.examples)
    shuffled=Counter((e.cell,e.features) for values in feedback.values() for e in values)
    assert original==shuffled
    assert stats["changed_feature_assignments"]>0
    for u in units:
        assert len(feedback[u.digest])==len(u.examples)
        assert [(x.cell,x.final) for x in feedback[u.digest]]==[
            (x.cell,x.final) for x in u.examples]

def test_fast_selector_matches_original_for_actual_and_sham_feedback():
    units=corpus()
    meta=metadata(units)
    lookup={u.digest:u.examples for u in units}
    assert incremental_adaptive_sequence(meta,lookup.__getitem__)==select_sequence(
        meta,lookup.__getitem__,policy="adaptive_observed",seed=101)
    sham,_=permuted_training_feedback(units,seed=21)
    assert incremental_adaptive_sequence(meta,sham.__getitem__)==select_sequence(
        meta,sham.__getitem__,policy="adaptive_observed",seed=101)

def test_replay_and_reproducibility():
    units=corpus()
    seconds={u.digest:2.+(i%3) for i,u in enumerate(units)}
    v07=evaluate_duration_budget(units,seconds,n_random=3)
    v08=evaluate_context_control(units,seconds,v07,repeats=3)
    result=evaluate_sham_feedback(units,seconds,v08,repeats=3)
    assert result==evaluate_sham_feedback(units,seconds,v08,repeats=3)
    assert len(result["folds"])==5
    for group in result["summary"]:
        assert sum(group[key] for key in ("permuted_draws_better_than_true",
                                          "permuted_draws_equal_to_true",
                                          "permuted_draws_worse_than_true"))==3
        if group["budget_fraction"]==1.0:
            assert group["permuted_draws_equal_to_true"]==3
