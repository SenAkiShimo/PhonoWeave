from pathlib import Path
import json
import pytest
from phonoweave.research_manifest_cli import generate

def test_generate_manifest_idempotent_and_private(tmp_path):
    singers=tmp_path/"singers"
    singers.mkdir()
    (singers/"Voice A").mkdir()
    (singers/"Voice B").mkdir()
    cohort=tmp_path/"cohort.json"
    cohort.write_text(json.dumps({"schema":"phonoweave.voicebank_replication_cohort.v1","voicebanks":[
        {"bank_id":"bank01","speaker_group_id":"speaker01","relative_dir":"Voice A"},
        {"bank_id":"bank02","speaker_group_id":"speaker02","relative_dir":"Voice B"},
    ]}),encoding="utf-8")
    output=tmp_path/"config"/"voicebanks.json"
    assert generate(cohort,singers,output)==output
    original=output.read_bytes()
    assert generate(cohort,singers,output)==output
    assert output.read_bytes()==original
    parsed=json.loads(output.read_text("utf-8"))
    assert len(parsed["voicebanks"])==2
    assert parsed["voicebanks"][0]["path"]==str((singers/"Voice A").resolve())

def test_missing_singer_does_not_write_partial_manifest(tmp_path):
    singers=tmp_path/"singers"
    singers.mkdir()
    cohort=tmp_path/"cohort.json"
    cohort.write_text(json.dumps({"schema":"phonoweave.voicebank_replication_cohort.v1","voicebanks":[
        {"bank_id":"bank01","speaker_group_id":"speaker01","relative_dir":"not_installed"},
    ]}),encoding="utf-8")
    output=tmp_path/"config"/"voicebanks.json"
    with pytest.raises(FileNotFoundError,match="not_installed"):
        generate(cohort,singers,output)
    assert not output.exists()
