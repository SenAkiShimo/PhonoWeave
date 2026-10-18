import json
import pytest
from phonoweave.lossless_alias_export import write_alias_records
from phonoweave.recording_scheme_precheck import audit_plan_evidence

def data(alias="- cengD#4", role="initial_cv", pitch="D#4", line=3):
    key=dict(source_oto=f"{pitch}/oto.ini",line_number=line,original_alias=alias)
    src=dict(key,source_wav="record.wav")
    r=dict(key,role=role,pitch_suffix=pitch,pitch_suffix_status="matched",right_token="ceng")
    return src,r,dict(key,compile_status="roundtrip_review_only"),dict(src,cutoff_ms=-12.5)

def test_lossless_alias_jsonl_preserves_formula_prefix_and_negative_numeric(tmp_path):
    s,r,c,t=data()
    path=tmp_path/"alias_records.jsonl"
    assert write_alias_records(path,[s],[r],[c],[t])==1
    obj=json.loads(path.read_text())
    assert obj["source"]["original_alias"]=="- cengD#4"
    assert obj["original_oto_timing"]["cutoff_ms"]==-12.5
    with pytest.raises(FileExistsError):
        write_alias_records(path,[s],[r],[c],[t])

def test_cross_pitch_evidence_is_not_auto_approval():
    s,r,c,t=data()
    plan={"schema":"phonoweave.recording_scheme_action_plan.v0.1","speaker_id":"s",
          "tasks":[{"task_id":"RS-001","review_action":"review_cross_pitch_coverage",
                    "pitch_groups":["D#4"],"suggested_syllables":["ceng"]}]}
    records=[dict(schema="phonoweave.lossless_alias_record.v0.1",source=s,role=r)]
    result=audit_plan_evidence(plan,records)
    assert result["review_tasks_checked"]==1
    assert result["human_resolutions_written"]==0
    assert result["cross_pitch_findings"][0]["observed_pitch_variants"][0]["matching_cv_aliases"]==1
    assert result["recordings_authorized"]==0

def test_mismatched_provenance_refused(tmp_path):
    s,r,c,t=data()
    r["line_number"]=7
    with pytest.raises(ValueError,match="mismatch"):
        write_alias_records(tmp_path/"records.jsonl",[s],[r],[c],[t])
