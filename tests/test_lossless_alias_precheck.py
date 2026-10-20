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


def test_all_review_actions_have_non_authorizing_structural_findings():
    src, role, comp, timing = data(alias="a pD#4", role="vc_transition")
    role["right_token"] = "p"
    src["base_unit"] = "p"
    rec = dict(schema="phonoweave.lossless_alias_record.v0.1",
               source=src, role=role, compilation_preview=comp,
               original_oto_timing=timing)
    example = {"source_oto": src["source_oto"], "line_number": src["line_number"],
               "source_wav": src["source_wav"],
               "original_alias": src["original_alias"],
               "pitch_group": "D#4"}
    actions = [
        ("review_internal_candidate_mapping", [example]),
        ("review_context_mapping", []),
        ("review_role_scoped_mapping", []),
        ("review_provisional_unit_decision", []),
    ]
    plan = {"schema": "phonoweave.recording_scheme_action_plan.v0.1",
            "speaker_id": "s", "tasks": [
                {"task_id": f"RS-{i:03d}", "unit_id": "p_internal",
                 "base_unit": "p", "review_action": action,
                 "pitch_groups": ["D#4"], "source_examples": examples,
                 "suggested_syllables": []}
                for i, (action, examples) in enumerate(actions, 1)
            ]}
    result = audit_plan_evidence(plan, [rec])
    assert result["review_tasks_checked"] == 4
    assert result["human_resolutions_written"] == 0
    assert result["recordings_authorized"] == 0
    assert result["task_findings"][0]["evidence"]["consistent_examples"] == 1


def test_internal_candidate_mismatched_wav_remains_unverified():
    src, role, comp, timing = data(alias="a pD#4", role="vc_transition")
    role["right_token"] = "p"
    rec = dict(schema="phonoweave.lossless_alias_record.v0.1",
               source=src, role=role, compilation_preview=comp)
    plan = {"schema": "phonoweave.recording_scheme_action_plan.v0.1",
            "speaker_id": "s", "tasks": [{
                "task_id": "RS-001", "unit_id": "p_internal",
                "base_unit": "p", "review_action": "review_internal_candidate_mapping",
                "pitch_groups": ["D#4"], "suggested_syllables": [],
                "source_examples": [{
                    "source_oto": src["source_oto"],
                    "line_number": src["line_number"],
                    "original_alias": src["original_alias"],
                    "source_wav": "incorrect.wav", "pitch_group": "D#4",
                }],
            }]}
    result = audit_plan_evidence(plan, [rec])
    assert result["task_findings"][0]["structural_status"] == "source_examples_need_review"
    assert "source_wav_mismatch" in result["task_findings"][0]["evidence"]["source_examples"][0]["issues"]
