"""Validation of artifact integrity, source roles, explicit approvals and safety."""
import copy
from hashlib import sha256
import json

import pytest

from phonoweave.alias_oto_task_compiler import (
    FILES, PACK_SCHEMA, _canonical_sha, build_review_template,
    compile_oto_tasks, load_pack, write_compiled_tasks,
)
from phonoweave.recording_scheme_decisions import _hash_plan


def fixture():
    plan = {
        "schema": "phonoweave.recording_scheme_action_plan.v0.1",
        "speaker_id": "fixture", "source_review_rows": 1,
        "tasks": [{"task_id": "RS-001", "unit_id": "b_internal_other",
                   "base_unit": "b", "review_action": "review_internal_candidate_mapping",
                   "pitch_groups": ["A#3"]}],
    }
    keys = {"source_oto": "A#3/oto.ini", "line_number": 8,
            "original_alias": "a bA#3"}
    alias = {
        "schema": "phonoweave.lossless_alias_record.v0.1",
        "source": {**keys, "source_wav": "A#3/_a_baA#3.wav"},
        "role": {**keys, "role": "vc_transition", "pitch_suffix": "A#3"},
    }
    directive = {
        "schema": "phonoweave.split_decision_directives.v0.1",
        "source_linked_internal_candidates": 1,
        "directives": [{
            "line_id": "RL-001", "pitch_group": "A#3",
            "source_wav": "A#3/_a_baA#3.wav", "syllable_index": 1,
            "syllable": "ba", "evidence_kind": "source_linked_internal_VC_candidate",
            "candidate_realizations": [{
                "unit_id": "b_internal_other", "context": "internal:other",
                "evidence_kind": "source_linked_internal_VC_candidate",
                "source_oto": "A#3/oto.ini", "oto_line_number": "8",
                "original_alias": "a bA#3", "alias_mapping_approved": False,
            }],
        }],
    }
    reviewed = {
        "schema": "phonoweave.reviewed_recording_scheme.v0.1",
        "speaker_id": "fixture", "source_plan_sha256": _hash_plan(plan),
        "reviewer_id": "engineer", "production_alias_export": "disabled",
        "reviewed_pitch_cells": 1, "pending_pitch_cells": 0,
        "resolved_pitch_reviews": [{
            "task_id": "RS-001", "pitch_group": "A#3",
            "unit_id": "b_internal_other", "base_unit": "b",
            "review_action": "review_internal_candidate_mapping",
            "resolution": "candidate_accepted", "reviewer_note": "Verified source",
            "evidence_reference": "A#3/oto.ini:8",
        }], "pending_pitch_reviews": [],
    }
    return plan, directive, [alias], reviewed


def approved_template(plan, directive, aliases):
    template = build_review_template(plan, directive, aliases, "fingerprint")
    template["reviewer_id"] = "engineer"
    template["targets"][0].update({
        "decision": "approve_annotation", "new_alias": "b_internal_A#3",
        "intended_role": "internal_cv", "role_verified": True,
        "boundary_verified": True, "reviewer_note": "Source segmented by ear",
    })
    return template


def test_template_is_pending_and_source_linked():
    plan, directive, aliases, reviewed = fixture()
    result = build_review_template(plan, directive, aliases, "fingerprint")
    assert result["target_count"] == 1
    assert result["targets"][0]["decision"] == "pending"
    assert result["targets"][0]["original_alias"] == "a bA#3"
    assert result["targets"][0]["oto_line_number"] == 8
    output = compile_oto_tasks(result, plan, directive, aliases, "fingerprint", reviewed)
    assert output["approved_manual_annotation_tasks"] == 0
    assert output["oto_ini_export"] == "disabled"


def test_explicit_review_creates_manual_task_not_oto_values(tmp_path):
    plan, directive, aliases, reviewed = fixture()
    template = approved_template(plan, directive, aliases)
    result = compile_oto_tasks(template, plan, directive, aliases, "fingerprint", reviewed)
    assert result["approved_manual_annotation_tasks"] == 1
    assert result["annotation_tasks"][0]["oto_parameters_ms"] is None
    assert result["annotation_tasks"][0]["phonemizer_routing"] is None
    out = tmp_path / "taskpack"
    write_compiled_tasks(result, out)
    assert "b_internal_A#3" in (out / "ANNOTATOR_OTO_TASKS.csv").read_text()
    with pytest.raises(FileExistsError):
        write_compiled_tasks(result, out)


def test_without_independent_candidate_approval_cannot_schedule_annotation():
    plan, directive, aliases, reviewed = fixture()
    reviewed["resolved_pitch_reviews"][0]["resolution"] = "candidate_rejected"
    template = approved_template(plan, directive, aliases)
    with pytest.raises(ValueError, match="accepted source context"):
        compile_oto_tasks(template, plan, directive, aliases, "fingerprint", reviewed)


def test_invalid_role_collision_and_unverified_boundary_blocked():
    plan, directive, aliases, reviewed = fixture()
    template = approved_template(plan, directive, aliases)
    template["targets"][0]["new_alias"] = "a bA#3"
    with pytest.raises(ValueError, match="colliding"):
        compile_oto_tasks(template, plan, directive, aliases, "fingerprint", reviewed)
    template["targets"][0]["new_alias"] = "new_alias_A#3"
    template["targets"][0]["boundary_verified"] = False
    with pytest.raises(ValueError, match="role/boundary"):
        compile_oto_tasks(template, plan, directive, aliases, "fingerprint", reviewed)


def test_forged_template_identity_and_mismatched_alias_source():
    plan, directive, aliases, reviewed = fixture()
    template = approved_template(plan, directive, aliases)
    template["targets"][0]["source_wav"] = "different.wav"
    with pytest.raises(ValueError, match="identity changed"):
        compile_oto_tasks(template, plan, directive, aliases, "fingerprint", reviewed)
    template["targets"][0]["source_wav"] = "A#3/_a_baA#3.wav"
    directive["directives"][0]["candidate_realizations"][0]["source_oto"] = "forged/oto.ini"
    with pytest.raises(ValueError, match="source OTO"):
        compile_oto_tasks(template, plan, directive, aliases, "fingerprint", reviewed)


def test_wrong_or_stale_review_fails():
    plan, directive, aliases, reviewed = fixture()
    template = approved_template(plan, directive, aliases)
    reviewed["source_plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        compile_oto_tasks(template, plan, directive, aliases, "fingerprint", reviewed)
    reviewed["source_plan_sha256"] = _hash_plan(plan)
    with pytest.raises(ValueError, match="stale"):
        compile_oto_tasks(template, plan, directive, aliases, "wrong", reviewed)


def test_review_pack_digest_integrity(tmp_path):
    plan, directive, aliases, reviewed = fixture()
    pack = tmp_path / "pack"
    pack.mkdir()
    payloads = [json.dumps(aliases[0]) + "\n", json.dumps(plan), json.dumps(directive)]
    digests = {}
    for relative, data in zip(FILES, payloads):
        path = pack / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
        digests[relative] = sha256(data.encode()).hexdigest()
    (pack / "review_pack_manifest.json").write_text(json.dumps({
        "schema": PACK_SCHEMA, "artifact_sha256": digests,
    }))
    _, loaded_plan, loaded_dir, loaded_records, fingerprint = load_pack(pack)
    assert loaded_plan == plan and loaded_dir == directive
    assert loaded_records == aliases
    assert fingerprint == _canonical_sha(digests)
    (pack / FILES[2]).write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_pack(pack)


def test_initial_lexical_cue_does_not_create_source_alias_target():
    plan, directive, aliases, reviewed = fixture()
    directive["directives"] = [{
        "line_id": "RL-001", "pitch_group": "A#3", "syllable_index": 0,
        "evidence_kind": "syllable_context_only", "candidate_realizations": [{
            "unit_id": "b_plain", "original_alias": None,
        }],
    }]
    directive["source_linked_internal_candidates"] = 0
    template = build_review_template(plan, directive, aliases, "fingerprint")
    assert template["targets"] == []
