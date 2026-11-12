"""Small source-provenance-linked manual review preview; never a recording list."""
import copy
import json

import pytest

from phonoweave.native_validation_preview_v26 import (
    SCHEMA, build_preview, attach_validation_preview, run_archive
)


def rows():
    result = []
    labels = ("i", "ir", "i0")
    for bank, style in (("mang_D4", "mang"), ("lu_C4", "lu")):
        for context, wav, char in (("a", "vowels.wav", "a"),
                                   ("shw", "sibilants.wav", "b"),
                                   ("m", "nasals.wav", "c")):
            for label in labels:
                result.append({
                    "bank": bank, "style": style, "context": context,
                    "symbol": label, "source_pcm_sha256": char * 64,
                    "source_wav": wav,
                    "source_alias": bank.split("_")[1] + label + " " + context
                })
    return result


def test_selects_three_cases_not_entire_coverage_matrix():
    report = build_preview(rows(), ["i", "ir", "i0"], max_cases=3)
    assert report["schema"] == SCHEMA
    assert report["selected_count"] == 3
    assert len({x["following_context"] for x in report["selected_review_cases"]}) == 3
    assert len({x["source_wav"] for x in report["selected_review_cases"]}) == 3
    assert {x["style"] for x in report["selected_review_cases"]} == {"mang", "lu"}
    for case in report["selected_review_cases"]:
        assert [x["original_label"] for x in case["examples"]] == ["i", "ir", "i0"]
        assert case["new_recording_required_now"] is False
        assert case["verified_phone_boundary"] is False
    assert report["new_recording_lines"] == []
    assert report["automatic_unit_merges_or_splits"] == []


def test_single_case_budget_is_honored():
    report = build_preview(rows(), ["i", "ir"], max_cases=1)
    assert report["selected_count"] == 1
    assert len(report["selected_review_cases"][0]["examples"]) == 2


def test_duplicate_oto_alias_does_not_make_independent_case():
    duplicate = copy.deepcopy(rows()[0])
    result = build_preview(rows() + [duplicate], ["i", "ir", "i0"])
    assert result["selected_count"] == 3
    assert result["new_recording_lines"] == []


def test_conflicting_alias_same_pcm_bank_context_abstains_for_case():
    altered = rows()
    changed = copy.deepcopy(altered[0])
    changed["source_alias"] = "D4not_the_same"
    altered.append(changed)
    result = build_preview(altered, ["i", "ir", "i0"])
    assert result["source_linked_candidate_cases"] == 5
    assert all(not (x["bank"] == "mang_D4" and x["following_context"] == "a")
               for x in result["selected_review_cases"])


def test_different_wavs_cannot_be_faked_as_one_sample():
    altered = rows()[:3]
    altered[1]["source_wav"] = "other.wav"
    result = build_preview(altered, ["i", "ir", "i0"])
    assert result["selected_count"] == 0
    assert result["status"] == "insufficient_co_recorded_original_label_examples"


def test_source_pcm_identity_validation():
    altered = rows()
    altered[0]["source_pcm_sha256"] = "not_sha"
    with pytest.raises(ValueError, match="SHA-256"):
        build_preview(altered, ["i", "ir", "i0"])


@pytest.mark.parametrize("budget", (0, True, 9, -1, 1.5))
def test_invalid_review_size_rejected(budget):
    with pytest.raises(ValueError, match="max_cases"):
        build_preview(rows(), ["i", "ir", "i0"], max_cases=budget)


def test_duplicate_and_unknown_labels_rejected():
    with pytest.raises(ValueError, match="distinct"):
        build_preview(rows(), ["i", "i"])
    with pytest.raises(ValueError, match="2-5"):
        build_preview(rows(), ["i"])


def test_attaching_preview_never_mutates_existing_recording_plan(monkeypatch):
    base = {
        "recording_plan": {
            "recording_lines": ["already verified missing"],
            "review_tasks": [{"line": "fix OTO"}],
        },
        "existing_sound_unit_review": {
            "original_user_labels": ["i", "ir", "i0"]
        },
    }
    original = copy.deepcopy(base)
    monkeypatch.setattr(
        "phonoweave.native_validation_preview_v26.run_archive",
        lambda archive, labels, max_cases: build_preview(rows(), labels, max_cases=max_cases),
    )
    result = attach_validation_preview(base, "not_used.zip", ["i", "ir", "i0"])
    assert base == original
    assert result["recording_plan"] == original["recording_plan"]
    assert result["minimal_unit_validation_preview"]["selected_count"] == 3
    with pytest.raises(ValueError, match="disagree"):
        attach_validation_preview(base, "not_used.zip", ["i", "invented"])


def test_non_pcm_alias_names_remain_opaque():
    data = rows()
    for row in data:
        row["symbol"] = {"i": "阴", "ir": "i¹", "i0": "ζ"}[row["symbol"]]
    out = build_preview(data, ["阴", "i¹", "ζ"])
    assert [x["original_label"] for x in out["selected_review_cases"][0]["examples"]] == [
        "阴", "i¹", "ζ"
    ]
    assert out["automatic_unit_merges_or_splits"] == []
