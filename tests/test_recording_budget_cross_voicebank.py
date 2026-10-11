"""Cross-bank manifest and replay invariants."""
import pytest

from phonoweave.recording_budget_benchmark import Example, RecordingUnit
from phonoweave.recording_budget_disjoint_cv import run_disjoint_benchmark
from phonoweave.recording_budget_random_sensitivity import evaluate_random_sensitivity
from phonoweave.recording_budget_cross_voicebank import (
    _compare_protocols, validate_manifest, evaluate_replication,
)

def manifest():
    return {"schema":"phonoweave.voicebank_replication_manifest.v1",
            "voicebanks":[
                {"bank_id":"bankA","speaker_group_id":"speakerA","path":"/tmp/banka"},
                {"bank_id":"bankB","speaker_group_id":"speakerB","path":"/tmp/bankb"}]}

def test_unique_speakers_required():
    data=manifest()
    data["voicebanks"][1]["speaker_group_id"]="speakerA"
    with pytest.raises(ValueError,match="unique"):
        validate_manifest(data)

def test_duplicate_paths_rejected():
    data=manifest()
    data["voicebanks"][1]["path"]=data["voicebanks"][0]["path"]
    with pytest.raises(ValueError,match="duplicate"):
        validate_manifest(data)

def test_missing_voicebanks_are_visible_failures():
    result=evaluate_replication(manifest(),repeats=3)
    assert result["evaluated_banks"]==0
    assert result["excluded_banks"]==2
    assert not result["cross_bank_replication_adequate"]
    assert all(x["status"]=="excluded" for x in result["banks"])

def test_original_protocol_replay():
    units=tuple(RecordingUnit(f"wav-{i:03d}",(
        Example(("s","sh","x")[i%3],("plain","rounded")[(i//3)%2],
                "a",tuple((i%7)*0.025 for _ in range(7))),
    )) for i in range(30))
    old=run_disjoint_benchmark(units,budgets=(0.25,0.5,1.0))
    new=evaluate_random_sensitivity(units,n_random=3,budgets=(0.25,0.5,1.0))
    _compare_protocols(old,new)
    new["fold_results"][0]["budgets"][0]["random_seed0_distance"] += 0.01
    with pytest.raises(ValueError,match="replay mismatch"):
        _compare_protocols(old,new)
