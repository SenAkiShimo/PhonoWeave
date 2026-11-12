"""v2.9 existing sound-unit alias -> physical PCM asset routing.

These fixtures verify WAV-container reuse, NOT phonetic unit substitution.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from phonoweave.native_asset_routing_v29 import (
    SCHEMA, build_asset_routing_ledger, attach_asset_routing_ledger,
)

LABELS = ["i", "ir", "i0"]


def event(label, ctx, wav, pcm, *, bank="lu_B4", alias=None):
    return {
        "bank": bank, "symbol": label, "context": ctx, "source_wav": wav,
        "source_pcm_sha256": pcm, "source_alias": alias or f"B4{label} {ctx}",
        "features": [0.0] * 12,
    }


def example_rows():
    rows = []
    for bank, prefix in [("lu_B4", "B4"), ("mang_D4", "D4")]:
        for context, wav, pcm in [
            ("a", "vcsection4.wav", "a" if bank == "lu_B4" else "c"),
            ("sh", "vcsection3.wav", "b" if bank == "lu_B4" else "d"),
        ]:
            for label in LABELS:
                rows.append(event(label, context, wav, pcm * 64, bank=bank,
                                  alias=f"{prefix}{label} {context}"))
    # One unit-context edge with two alternative historical source WAVs.
    rows.append(event("i", "a", "eriin.wav", "e"*64))
    return rows


def test_preserves_all_author_labels_and_distinguishes_container_from_unit():
    report = build_asset_routing_ledger(example_rows(), LABELS)
    assert report["schema"] == SCHEMA
    assert report["original_user_labels"] == LABELS
    assert report["total_existing_alias_context_routes"] == 12
    assert report["distinct_existing_bank_PCM_source_containers"] == 5
    assert report["routes_requiring_source_choice_review"] == 1
    assert len(report["same_container_cross_label_examples"]) == 4
    assert report["proposed_new_recording_lines"] == []
    assert report["approved_unit_substitutions"] == []
    assert report["approved_alias_merges"] == []
    for item in report["same_container_cross_label_examples"]:
        assert item["co_container_not_same_phonetic_unit"] is True
        assert item["safe_to_merge_aliases_or_segments"] is False


def test_only_ambiguous_source_option_is_reviewed():
    report = build_asset_routing_ledger(example_rows(), LABELS)
    ambiguous = [r for r in report["original_alias_routes"]
                 if r["routing_status"] == "multiple_observed_source_options_review"]
    assert [(x["bank"], x["original_unit_label"], x["following_context"])
            for x in ambiguous] == [("lu_B4", "i", "a")]
    assert len(ambiguous[0]["observed_source_options"]) == 2
    assert all(r["unit_identity_must_be_preserved"]
               for r in report["original_alias_routes"])


def test_duplicate_oto_occurrence_is_not_new_asset_or_route():
    rows = example_rows()
    rows.append(deepcopy(rows[0]))
    report = build_asset_routing_ledger(rows, LABELS)
    assert report["distinct_existing_bank_PCM_source_containers"] == 5
    assert report["total_existing_alias_context_routes"] == 12


def test_same_pcm_different_wav_paths_is_one_asset_but_flagged():
    rows = example_rows()
    rows.append(event("i", "m", "copy.wav", "a"*64))
    report = build_asset_routing_ledger(rows, LABELS)
    assert report["distinct_existing_bank_PCM_source_containers"] == 5
    container = [a for a in report["existing_source_containers"]
                 if a["bank"] == "lu_B4" and a["source_pcm_sha256"] == "a"*64][0]
    assert container["possible_exact_PCM_duplicate_paths"] is True
    assert container["observed_source_wav_paths"] == ["copy.wav", "vcsection4.wav"]


def test_different_pcm_for_same_bank_wav_refused():
    rows = example_rows()
    rows.append(event("i", "m", "vcsection4.wav", "f"*64))
    with pytest.raises(ValueError, match="incompatible PCM"):
        build_asset_routing_ledger(rows, LABELS)


@pytest.mark.parametrize("labels", [["i"], ["i", "i"], ["i", "not_present"], ["", "ir"]])
def test_invalid_or_absent_author_labels_refused(labels):
    with pytest.raises(ValueError, match="2–5|distinct|do not occur"):
        build_asset_routing_ledger(example_rows(), labels)


def test_no_silent_alias_merge_even_in_one_long_wav():
    report = build_asset_routing_ledger(example_rows(), LABELS)
    a = next(r for r in report["original_alias_routes"]
             if r["bank"] == "lu_B4" and r["original_unit_label"] == "i"
             and r["following_context"] == "sh")
    b = next(r for r in report["original_alias_routes"]
             if r["bank"] == "lu_B4" and r["original_unit_label"] == "ir"
             and r["following_context"] == "sh")
    assert (a["observed_source_options"][0]["source_wav"]
            == b["observed_source_options"][0]["source_wav"])
    assert a["cross_context_audio_substitution_authorized"] is False
    assert b["cross_context_audio_substitution_authorized"] is False


def test_attachment_preserves_existing_plan(monkeypatch):
    from phonoweave import native_asset_routing_v29 as m
    old = {"recording_plan": {"recording_lines": ["record x"], "review_tasks": ["inspect y"]},
           "native_author_feedback_review": {
               "original_user_labels": LABELS,
               "author_confirmation": {
                   "original_alias_mapping": "yes",
                   "desire_to_keep_existing_realization_labels_distinct": "yes",
               }}}
    before = deepcopy(old)
    monkeypatch.setattr(m, "run_archive", lambda archive, labels:
                        build_asset_routing_ledger(example_rows(), labels))
    result = attach_asset_routing_ledger(old, "unused.zip", LABELS)
    assert old == before
    assert result["recording_plan"] == before["recording_plan"]
    assert result["existing_source_asset_routing"]["proposed_new_recording_lines"] == []
    with pytest.raises(ValueError, match="already attached"):
        attach_asset_routing_ledger(result, "unused.zip", LABELS)


def test_author_confirmation_mismatch_blocks_attachment():
    old = {"recording_plan": {},
           "native_author_feedback_review": {
               "original_user_labels": LABELS,
               "author_confirmation": {
                   "original_alias_mapping": "yes",
                   "desire_to_keep_existing_realization_labels_distinct": "uncertain",
               }}}
    with pytest.raises(ValueError, match="author confirmation"):
        attach_asset_routing_ledger(old, "unused.zip", LABELS)


def test_original_unicode_labels_are_not_normalized_or_replaced():
    rows = example_rows()
    mapping = {"i": "阴", "ir": "ir²", "i0": "ζ"}
    for row in rows:
        row["symbol"] = mapping[row["symbol"]]
    report = build_asset_routing_ledger(rows, list(mapping.values()))
    assert report["original_user_labels"] == ["阴", "ir²", "ζ"]
    assert all(route["original_unit_label"] in mapping.values()
               for route in report["original_alias_routes"])
