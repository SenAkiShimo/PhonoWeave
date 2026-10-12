"""No-lookahead, metadata ablation, WAV evidence aggregation, determinism."""
import pytest
from phonoweave.recording_budget_benchmark import Example, RecordingUnit, metadata
from phonoweave.recording_budget_acquisition_utility import (
    AcousticEvidence, select_utility_sequence,
)
from phonoweave.recording_budget_utility_eval import evaluate_bank

def corpus(n=30):
    return tuple(RecordingUnit(
        f"wav-{i:03d}",
        (Example(("s","sh","x")[i%3],("plain","rounded")[(i//3)%2],"a",
                 (i%9*0.013,)*7),)
    ) for i in range(n))

def test_coverage_priority_and_deterministic_sequence():
    data=corpus()
    candidates=metadata(data)
    mapping={u.digest:u.examples for u in data}
    order=select_utility_sequence(candidates,mapping.__getitem__)
    assert order==select_utility_sequence(candidates,mapping.__getitem__)
    assert set(order)==set(mapping)
    covered=set()
    for identifier in order:
        covered.update(next(u for u in data if u.digest==identifier).cells)
        if len(covered)==6:
            break
    assert len(covered)==6

def test_reveals_only_already_selected_acoustic_feedback():
    data=corpus()
    mapping={u.digest:u.examples for u in data}
    seen=[]
    def acquire(identifier):
        assert identifier not in seen
        seen.append(identifier)
        return mapping[identifier]
    result=select_utility_sequence(metadata(data),acquire)
    assert result==tuple(seen)

def test_metadata_only_never_acquires_audio():
    def forbidden(identifier):
        raise AssertionError("metadata-only mode must not request samples")
    data=corpus()
    order=select_utility_sequence(metadata(data),forbidden,use_acoustic_feedback=False)
    assert len(order)==len(data)

def test_one_WAV_with_multiple_OTO_segments_counts_once():
    e=Example("s","plain","a",(0.1,)*7)
    evidence=AcousticEvidence()
    evidence.observe((e,e,e))
    assert evidence.observations("s:plain")==1
    evidence.observe((e,))
    assert evidence.observations("s:plain")==2

def test_realized_duration_limits_and_reproducibility():
    data=corpus()
    durations={u.digest:2.0+i%4 for i,u in enumerate(data)}
    output=evaluate_bank(data,durations)
    assert output==evaluate_bank(data,durations)
    for fold in output["folds"]:
        for row in fold["budget_results"]:
            for scores in row["methods"].values():
                assert scores["spent_fraction"]<=row["budget_fraction"]+1e-5
