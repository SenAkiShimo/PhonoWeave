"""v2.4: existing author-native labels, acoustic evidence, and reclist isolation.

These are synthetic decision-interface tests, not phoneme recognition trials.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from phonoweave.native_unit_review_v24 import (
    SCHEMA, SOURCE_SCHEMA, SOURCE_ORIGIN,
    attach_existing_unit_review, summarize_existing_realizations,
)


def evidence(*, errors_for_y=0, context_suffix=""):
    labels = ["字", "字0"]
    banks = [("sA", "G3"), ("sA", "B4"), ("sB", "D4"), ("sB", "G4")]
    original = []
    for style, pitch in banks:
        for context in ("m", "sh"):
            for label in labels:
                original.append({
                    "bank": f"{style}_{pitch}",
                    "style": style,
                    "pitch": pitch,
                    "context": context + context_suffix,
                    "actual_user_symbol": label,
                    "nearest_user_symbol": label,
                    "matched": True,
                })
    d = {
        "schema": SOURCE_SCHEMA,
        "language_id": "user-Lang",
        "original_user_base_symbol": "字",
        "user_realization_labels": labels,
        "pitch_context_origin": SOURCE_ORIGIN,
        "candidate_windows": 16,
        "per_label_count": {k: 8 for k in labels},
        "unit_decision": "insufficient_independent_sources_for_auto_alias_change",
        "recommended_new_user_aliases": [],
        "source_audio_modified": False,
    }
    for field, agg in (
        ("cross_context_predictions", "cross_context_top1"),
        ("leave_pitch_out_predictions", "leave_pitch_out_top1"),
        ("leave_style_out_predictions", "leave_style_out_top1"),
    ):
        rows = copy.deepcopy(original)
        if field == "leave_style_out_predictions":
            indexes = [i for i, x in enumerate(rows) if x["actual_user_symbol"] == "字0"]
            for i in indexes[:errors_for_y]:
                rows[i]["nearest_user_symbol"] = "字"
                rows[i]["matched"] = False
        correct = sum(x["matched"] for x in rows)
        d[field] = rows
        d[agg] = {"correct": correct, "total": len(rows),
                  "accuracy": round(correct / len(rows), 4)}
    return d


def test_two_consistent_reports_preserve_existing_user_labels():
    x = summarize_existing_realizations([evidence(), evidence(errors_for_y=1)])
    assert x["schema"] == SCHEMA
    assert x["original_user_labels"] == ["字", "字0"]
    assert x["same_trial_identities_across_reports"]
    assert [r["review_status"] for r in x["realizations"]] == [
        "existing_label_model_rendered_distinction_consistent",
        "existing_label_model_rendered_distinction_consistent",
    ]
    assert x["automatic_new_synthesis_units"] == []
    assert x["automatic_new_recording_lines"] == []
    assert x["trial_identity_not_pcm_proof"] is True


def test_window_sensitive_existing_label_requires_review():
    x = summarize_existing_realizations([evidence(), evidence(errors_for_y=2)])
    states = {r["original_user_label"]: r["review_status"] for r in x["realizations"]}
    assert states["字"] == "existing_label_model_rendered_distinction_consistent"
    assert states["字0"] == "insufficient_or_window_sensitive_evidence"


def test_single_report_insufficient_even_when_perfect():
    x = summarize_existing_realizations([evidence()])
    assert all(r["review_status"] == "insufficient_or_window_sensitive_evidence"
               for r in x["realizations"])


def test_incomparable_trial_sets_do_not_claim_window_stability():
    x = summarize_existing_realizations([
        evidence(), evidence(context_suffix="_other"),
    ])
    assert x["same_trial_identities_across_reports"] is False
    assert all(r["review_status"] == "insufficient_or_window_sensitive_evidence"
               for r in x["realizations"])


def test_duplicate_trial_is_rejected_not_double_counted():
    d = evidence()
    for field, agg in (
        ("cross_context_predictions", "cross_context_top1"),
        ("leave_pitch_out_predictions", "leave_pitch_out_top1"),
        ("leave_style_out_predictions", "leave_style_out_top1"),
    ):
        d[field].append(copy.deepcopy(d[field][0]))
        d[agg]["total"] += 1
        d[agg]["correct"] += 1
        d[agg]["accuracy"] = round(d[agg]["correct"] / d[agg]["total"], 4)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_existing_realizations([d])


def test_contradictory_aggregate_rejected():
    d = evidence()
    d["leave_style_out_top1"]["correct"] -= 1
    with pytest.raises(ValueError, match="numerator"):
        summarize_existing_realizations([d])


def test_wrong_source_provenance_rejected():
    d = evidence()
    d["pitch_context_origin"] = "independent_human_takes"
    with pytest.raises(ValueError, match="provenance"):
        summarize_existing_realizations([d])


def test_auto_new_alias_in_source_rejected():
    d = evidence()
    d["recommended_new_user_aliases"] = ["字A"]
    with pytest.raises(ValueError, match="provenance"):
        summarize_existing_realizations([d])


def test_context_and_label_identities_preserved_and_no_plan_mutations(tmp_path):
    input_plan = {
        "schema": "phonoweave.user_native_recording_plan.v2.3",
        "recording_plan": {
            "recording_lines": ["其他已缺录项"],
            "selected": [{"line": "其他已缺录项", "missing_audio_banks": ["D4"]}],
            "review_tasks": [{"line": "检查项", "category": "source_verification"}],
        },
    }
    original = copy.deepcopy(input_plan)
    files = []
    for i, d in enumerate((evidence(), evidence(errors_for_y=2))):
        p = tmp_path / f"evidence_{i}.json"
        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        files.append(p)
    result = attach_existing_unit_review(input_plan, files)
    assert input_plan == original
    assert result["recording_plan"] == original["recording_plan"]
    assert result["existing_sound_unit_review"]["input_report_sha256"][0]
    assert len(result["existing_sound_unit_review"]["input_report_sha256"]) == 2
    assert "字0" in result["existing_sound_unit_review"]["original_user_labels"]
    assert result["existing_sound_unit_review"]["automatic_new_recording_lines"] == []


def test_duplicate_input_path_refused(tmp_path):
    p = tmp_path / "evidence.json"
    p.write_text(json.dumps(evidence()), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate evidence path"):
        attach_existing_unit_review({"recording_plan": {}}, [p, p])


def test_report_label_mismatch_refused():
    a = evidence()
    b = evidence()
    b["user_realization_labels"] = ["字0", "字"]
    with pytest.raises(ValueError, match="different languages"):
        summarize_existing_realizations([a, b])


def test_boolean_count_rejected():
    d = evidence()
    d["per_label_count"]["字"] = True
    with pytest.raises(ValueError, match="counts"):
        summarize_existing_realizations([d])


def test_no_independent_phone_claim_even_when_all_green():
    result = summarize_existing_realizations([evidence(), evidence()])
    assert "not unknown phoneme discovery" in result["limitations"][0]
    assert result["automatic_alias_edits"] == []
    assert all(r["new_phoneme_authorized"] is False for r in result["realizations"])


def test_identical_copied_reports_are_not_separate_windows():
    with pytest.raises(ValueError, match="duplicate acoustic evidence content"):
        summarize_existing_realizations([evidence(), evidence()])
