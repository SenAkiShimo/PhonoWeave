"""Acceptance feedback is a decision constraint, not consent to record."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from phonoweave.native_author_feedback_v27 import (
    FEEDBACK_SCHEMA, SCHEMA, evaluate_author_feedback, attach_author_feedback,
)


def original():
    labels = ["i", "ir", "i0"]
    cases = []
    for num, ctx in enumerate(("a", "ei", "hw"), start=1):
        cases.append({
            "case_number": num, "bank": "lu_B4",
            "following_context": ctx, "source_wav": f"vcsection{num}.wav",
            "source_pcm_sha256": f"{num}" * 64,
            "examples": [
                {"original_label": label, "original_alias": f"B4{label} {ctx}",
                 "source_wav": f"vcsection{num}.wav"}
                for label in labels
            ],
        })
    return {
        "schema": "phonoweave.minimal_native_validation_preview.v2.6",
        "original_user_labels": labels,
        "source_archive_sha256": "a" * 64,
        "selected_review_cases": cases, "selected_count": 3,
        "new_recording_lines": [], "automatic_unit_merges_or_splits": [],
    }


def feedback(p=None):
    p = p or original()
    return {
        "schema": FEEDBACK_SCHEMA,
        "preview_sha256": "b" * 64,
        "alias_mapping_correct": "yes",
        "existing_realizations_should_remain_distinct": "yes",
        "case_recordability": [
            {"case_number": x["case_number"], "status": "uncertain"}
            for x in p["selected_review_cases"]
        ],
    }


def assess(p=None, f=None, **kwargs):
    p = p or original()
    f = f or feedback(p)
    return evaluate_author_feedback(
        p, f, preview_file_sha256="b" * 64, **kwargs
    )


def test_author_acceptance_retains_original_names_but_does_not_order_recordings():
    output = assess()
    assert output["schema"] == SCHEMA
    assert output["author_confirmation"]["original_alias_mapping"] == "yes"
    assert output["author_confirmation"]["desire_to_keep_existing_realization_labels_distinct"] == "yes"
    assert output["author_confirmation"]["scientifically_verified_phonetic_separation"] is False
    assert [x["original_user_labels"] if False else x["following_context"] for x in output["reviewed_contexts"]] == ["a", "ei", "hw"]
    assert {x["author_recordability"] for x in output["reviewed_contexts"]} == {"uncertain"}
    assert output["recording_readiness"] == "blocked_by_unconfirmed_context_recordability"
    assert output["automatic_recording_requests"] == []
    assert output["automatic_unit_merges"] == []
    assert output["automatic_unit_splits"] == []


def test_unknown_is_not_no_and_no_does_not_require_recording():
    report = assess()
    assert "Unknown recording feasibility is not a rejection" in report["warnings"][0]
    f = feedback()
    f["case_recordability"][0]["status"] = "no"
    report2 = assess(f=f)
    assert report2["reviewed_contexts"][0]["author_recordability"] == "no"
    assert report2["automatic_recording_requests"] == []


def test_all_yes_not_automatic_consent():
    f = feedback()
    for item in f["case_recordability"]:
        item["status"] = "yes"
    r = assess(f=f)
    assert r["recording_readiness"] == "still_requires_independent_source_and_explicit_recording_plan"
    assert not any(x["recording_ready"] for x in r["reviewed_contexts"])
    assert r["automatic_recording_requests"] == []


def test_feedback_hash_must_match_original_review():
    f=feedback()
    f["preview_sha256"]="e"*64
    with pytest.raises(ValueError,match="exact v2.6 preview"):
        assess(f=f)


def test_missing_or_duplicate_case_feedback_rejected():
    f=feedback()
    f["case_recordability"][2]=copy.deepcopy(f["case_recordability"][0])
    with pytest.raises(ValueError,match="duplicate author case"):
        assess(f=f)
    f=feedback()
    f["case_recordability"].pop()
    with pytest.raises(ValueError,match="exactly"):
        assess(f=f)


def test_no_new_alias_label_is_invented():
    p=original()
    p["new_recording_lines"]=["new_alias"]
    with pytest.raises(ValueError,match="unexpectedly authorizes"):
        assess(p=p)


def test_alternatives_are_unreviewed_and_come_from_different_contexts():
    p=original()
    source=[]
    for context, h, wav in (("a","1","x.wav"),("m","4","m.wav"),("sh","5","sh.wav"),("u","6","u.wav")):
        for label in p["original_user_labels"]:
            source.append({
                "symbol":label,"bank":"lu_C4","context":context,
                "source_pcm_sha256":h*64,"source_wav":wav,
                "source_alias":"C4"+label+" "+context,"style":"lu"
            })
    r=assess(p=p,source_rows=source,source_archive_sha256="a"*64)
    assert len(r["alternative_archived_contexts_for_future_inspection_only"])==3
    assert {x["following_context"] for x in r["alternative_archived_contexts_for_future_inspection_only"]}=={"m","sh","u"}
    assert all(x["author_recordability"]=="unreviewed" for x in r["alternative_archived_contexts_for_future_inspection_only"])
    assert all(x["review_only_not_recording_instruction"] for x in r["alternative_archived_contexts_for_future_inspection_only"])
    assert r["automatic_recording_requests"] == []


def test_alternative_source_hash_mismatch_rejected():
    with pytest.raises(ValueError,match="candidate archive differs"):
        assess(source_rows=[],source_archive_sha256="9"*64)


def test_invalid_alternative_budget_rejected():
    with pytest.raises(ValueError,match="max_alternatives"):
        assess(max_alternatives=True)


def test_attach_to_existing_recording_plan_never_mutates_reclist(monkeypatch):
    from phonoweave import native_author_feedback_v27 as module
    plan={"recording_plan":{"recording_lines":["existing target"],"review_tasks":["fix oto"]}}
    backup=copy.deepcopy(plan)
    monkeypatch.setattr(module,"run",lambda *args,**kw:assess())
    out=attach_author_feedback(plan,"preview.json","feedback.json")
    assert out["recording_plan"]==backup["recording_plan"]
    assert plan==backup
    assert out["native_author_feedback_review"]["automatic_recording_requests"]==[]
    with pytest.raises(ValueError,match="already contains"):
        attach_author_feedback(out,"preview.json","feedback.json")


def test_real_feedback_file_roundtrip(tmp_path):
    from phonoweave import native_author_feedback_v27 as module
    preview=original()
    preview_path=tmp_path/"preview.json"
    preview_path.write_text(json.dumps(preview),encoding="utf-8")
    f=feedback()
    f["preview_sha256"]=hashlib.sha256(preview_path.read_bytes()).hexdigest()
    feedback_path=tmp_path/"feedback.json"
    feedback_path.write_text(json.dumps(f),encoding="utf-8")
    result=module.run(preview_path,feedback_path)
    assert result["reviewed_contexts"][2]["author_recordability"]=="uncertain"
    assert result["automatic_unit_splits"]==[]
