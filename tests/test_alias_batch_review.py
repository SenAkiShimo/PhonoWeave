"""Batch policy identity, no implied approvals, and per-source safety gates."""
import copy
import json
import pytest

from phonoweave.alias_batch_review import (
    _format_alias, expand_batch_review, make_batch_review, main,
)
from phonoweave.alias_oto_task_compiler import compile_oto_tasks
from test_alias_oto_task_compiler import fixture


def batch_fixture():
    action, directives, records, reviewed = fixture()
    batch = make_batch_review(action, directives, records, "fingerprint")
    return action, directives, records, reviewed, batch


def authorize_group(batch):
    batch["reviewer_id"] = "reviewer-one"
    batch["groups"][0].update(
        decision="approve_naming", alias_format="{unit_id}_{pitch_group}_{task_id}",
        reviewer_note="Reviewed internal-CV naming convention")
    return batch


def authorize_target(batch):
    batch["targets"][0].update(
        decision="approve_annotation", role_verified=True,
        boundary_verified=True, reviewer_note="Audited actual waveform and role")
    return batch


def test_batch_template_default_is_never_approved():
    a, d, r, reviewed, batch = batch_fixture()
    assert batch["group_count"] == 1
    assert batch["target_count"] == 1
    assert batch["groups"][0]["decision"] == "pending"
    expanded, preview = expand_batch_review(batch, a, d, r, "fingerprint")
    assert expanded["targets"][0]["decision"] == "pending"
    assert preview["name_proposals"][0]["proposed_alias"] is None
    result = compile_oto_tasks(expanded, a, d, r, "fingerprint", reviewed)
    assert result["approved_manual_annotation_tasks"] == 0


def test_group_naming_is_not_individual_boundary_approval():
    a, d, r, reviewed, batch = batch_fixture()
    authorize_group(batch)
    expanded, preview = expand_batch_review(batch, a, d, r, "fingerprint")
    assert preview["name_proposals"][0]["proposed_alias"] == "b_internal_other_A#3_ALT-0001"
    assert expanded["targets"][0]["decision"] == "pending"
    assert compile_oto_tasks(expanded, a, d, r, "fingerprint", reviewed)[
        "approved_manual_annotation_tasks"] == 0


def test_double_explicit_review_creates_task_with_null_oto():
    a, d, r, reviewed, batch = batch_fixture()
    authorize_target(authorize_group(batch))
    expanded, preview = expand_batch_review(batch, a, d, r, "fingerprint")
    assert preview["explicitly_approved_individual_targets"] == 1
    result = compile_oto_tasks(expanded, a, d, r, "fingerprint", reviewed)
    assert result["approved_manual_annotation_tasks"] == 1
    assert result["annotation_tasks"][0]["oto_parameters_ms"] is None
    assert result["annotation_tasks"][0]["phonemizer_routing"] is None
    assert result["oto_ini_export"] == "disabled"


def test_source_review_independence_remains_required():
    a, d, r, reviewed, batch = batch_fixture()
    authorize_target(authorize_group(batch))
    reviewed["resolved_pitch_reviews"][0]["resolution"] = "candidate_rejected"
    expanded, _ = expand_batch_review(batch, a, d, r, "fingerprint")
    with pytest.raises(ValueError, match="accepted source context"):
        compile_oto_tasks(expanded, a, d, r, "fingerprint", reviewed)


def test_tampered_identity_and_stale_source_fail():
    a, d, r, reviewed, batch = batch_fixture()
    batch["targets"][0]["group_id"] = "AN-999"
    with pytest.raises(ValueError, match="identity changed"):
        expand_batch_review(batch, a, d, r, "fingerprint")
    a, d, r, reviewed, batch = batch_fixture()
    with pytest.raises(ValueError, match="stale"):
        expand_batch_review(batch, a, d, r, "different-fingerprint")


def test_invalid_format_and_existing_alias_collision_rejected():
    a, d, r, reviewed, batch = batch_fixture()
    authorize_group(batch)
    batch["groups"][0]["alias_format"] = "{unit_id.__class__}"
    with pytest.raises(ValueError, match="unsafe alias format field"):
        expand_batch_review(batch, a, d, r, "fingerprint")
    batch["groups"][0]["alias_format"] = "a bA#3_{task_id}"
    # this is a different name; old exact alias collision tested below
    expand_batch_review(batch, a, d, r, "fingerprint")
    batch["groups"][0]["alias_format"] = "a bA#3"
    with pytest.raises(ValueError, match="must reference"):
        expand_batch_review(batch, a, d, r, "fingerprint")
    # Match an existing alias exactly while still using a placeholder.
    r[0]["source"]["original_alias"] = "b_internal_other_A#3_ALT-0001"
    r[0]["role"]["original_alias"] = "b_internal_other_A#3_ALT-0001"
    d["directives"][0]["candidate_realizations"][0]["original_alias"] = "b_internal_other_A#3_ALT-0001"
    batch = make_batch_review(a, d, r, "fingerprint")
    authorize_group(batch)
    with pytest.raises(ValueError, match="collision"):
        expand_batch_review(batch, a, d, r, "fingerprint")


def test_bulk_group_target_count_and_duplicate_alias_detection():
    a, d, r, reviewed = fixture()
    new_source = copy.deepcopy(r[0])
    new_source["source"]["line_number"] = 9
    new_source["role"]["line_number"] = 9
    new_source["source"]["original_alias"] = "a b_newA#3"
    new_source["role"]["original_alias"] = "a b_newA#3"
    r.append(new_source)
    new_directive = copy.deepcopy(d["directives"][0])
    new_directive["line_id"] = "RL-002"
    new_directive["candidate_realizations"][0]["source_oto"] = "A#3/oto.ini"
    new_directive["candidate_realizations"][0]["oto_line_number"] = "9"
    new_directive["candidate_realizations"][0]["original_alias"] = "a b_newA#3"
    d["directives"].append(new_directive)
    d["source_linked_internal_candidates"] = 2
    batch = make_batch_review(a, d, r, "fingerprint")
    assert batch["group_count"] == 1 and batch["target_count"] == 2
    authorize_group(batch)
    batch["groups"][0]["alias_format"] = "{unit_id}_{pitch_group}"
    with pytest.raises(ValueError, match="collision"):
        expand_batch_review(batch, a, d, r, "fingerprint")
    batch["groups"][0]["alias_format"] = "{unit_id}_{pitch_group}_{task_id}"
    expanded, preview = expand_batch_review(batch, a, d, r, "fingerprint")
    assert len(set(x["proposed_alias"] for x in preview["name_proposals"])) == 2
    assert not any(x["decision"] == "approve_annotation" for x in expanded["targets"])


def test_individual_verification_cannot_be_inherited_from_group():
    a, d, r, reviewed, batch = batch_fixture()
    authorize_group(batch)
    batch["targets"][0]["decision"] = "approve_annotation"
    with pytest.raises(ValueError, match="independent role/boundary"):
        expand_batch_review(batch, a, d, r, "fingerprint")
    batch["targets"][0]["decision"] = "pending"
    batch["targets"][0]["role_verified"] = True
    with pytest.raises(ValueError, match="unapproved target"):
        expand_batch_review(batch, a, d, r, "fingerprint")


def test_invalid_alias_pattern_format():
    with pytest.raises(ValueError, match="format"):
        _format_alias("{task_id!r}", {"task_id":"ALT-0001"})
    with pytest.raises(ValueError, match="grammar"):
        _format_alias("{task_id}/bad", {"task_id":"ALT-0001"})


def test_cli_template_preview_compile_without_any_approvals(tmp_path):
    from hashlib import sha256
    from phonoweave.alias_oto_task_compiler import FILES, PACK_SCHEMA
    a, d, r, reviewed = fixture()
    pack = tmp_path / "pack"
    pack.mkdir()
    serialized = [json.dumps(r[0]) + "\n", json.dumps(a), json.dumps(d)]
    digests = {}
    for name, payload in zip(FILES, serialized):
        path = pack / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        digests[name] = sha256(payload.encode()).hexdigest()
    (pack / "review_pack_manifest.json").write_text(json.dumps({
        "schema": PACK_SCHEMA, "artifact_sha256": digests,
    }))
    grouped = tmp_path / "batch.json"
    assert main(["template", "--pack", str(pack), "--output", str(grouped)]) == 0
    assert json.loads(grouped.read_text())["target_count"] == 1
    preview = tmp_path / "preview.json"
    assert main(["preview", "--pack", str(pack), "--batch-reviews", str(grouped),
                 "--output", str(preview)]) == 0
    assert json.loads(preview.read_text())["approved_naming_groups"] == []
    reviewed_path = tmp_path / "reviewed.json"
    reviewed_path.write_text(json.dumps(reviewed))
    output = tmp_path / "compiled"
    assert main(["compile", "--pack", str(pack), "--batch-reviews", str(grouped),
                 "--reviewed-scheme", str(reviewed_path), "--output", str(output)]) == 0
    compiled = json.loads((output / "alias_oto_annotation_tasks.json").read_text())
    assert compiled["approved_manual_annotation_tasks"] == 0
    assert compiled["oto_ini_export"] == "disabled"
    assert compiled["batch_review"]["group_count"] == 1
    with pytest.raises(SystemExit):
        main(["template", "--pack", str(pack), "--output", str(grouped)])
