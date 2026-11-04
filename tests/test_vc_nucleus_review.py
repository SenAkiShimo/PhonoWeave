"""Regression tests for author-confirmed vowel-label identities and VC-shaped OTO review."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from phonoweave.vc_nucleus_review import (
    PROFILE_SCHEMA, SCHEMA, SOURCE_SCHEMA,
    compile_confirmed_nucleus_review, write_confirmed_nucleus_review,
)


def _fixture():
    hash_ = "a" * 64
    profile = {
        "schema": PROFILE_SCHEMA,
        "source_oto_sha256": hash_,
        "review_scope": "nucleus_label_semantics_only",
        "alias_prefix": "D4",
        "labels": [
            {"alias_left_surface": "i", "nucleus_class_id": "li_i",
             "speaker_examples": ["li"], "author_confirmed": True},
            {"alias_left_surface": "ir", "nucleus_class_id": "shi_i",
             "speaker_examples": ["shi", "ri"], "author_confirmed": True},
            {"alias_left_surface": "i0", "nucleus_class_id": "zi_i",
             "speaker_examples": ["zi", "ci"], "author_confirmed": True},
        ],
    }
    rows = [
        {"alias": f"D4{left} {right}", "left_surface": left, "right_surface": right,
         "oto_line": index, "wav": wav,
         "spectral_status": status, "spectral_centroid_hz": index * 20.0}
        for index, (left, right, wav, status) in enumerate([
            ("i", "m", "same.wav", "provisional_oto_relative_only"),
            ("ir", "m", "same.wav", "provisional_oto_relative_only"),
            ("i0", "m", "same.wav", "provisional_oto_relative_only"),
            ("i", "sh", "a.wav", "provisional_oto_relative_only"),
            ("ir", "sh", "b.wav", "provisional_oto_relative_only"),
            ("i0", "sh", "c.wav", "provisional_oto_relative_only"),
            ("i", "z", "d.wav", "too_short_for_fixed_16ms_proxy"),
            ("ir", "z", "d.wav", "provisional_oto_relative_only"),
            ("i0", "z", "d.wav", "provisional_oto_relative_only"),
            ("i", "n", "k.wav", "provisional_oto_relative_only"),
        ], start=1)
    ]
    return {"schema": SOURCE_SCHEMA, "source": {"oto_sha256": hash_, "archive_sha256": "b" * 64}, "candidates": rows}, profile


def test_nucleus_labels_distinct_and_priority():
    evidence, profile = _fixture()
    result = compile_confirmed_nucleus_review(evidence, profile)
    assert result["schema"] == SCHEMA
    assert result["summary"] == {
        "tracked_nucleus_labels": 3,
        "unique_right_surface_labels": 4,
        "complete_surface_groups": 3,
        "three_measurable_proxy_groups": 2,
        "same_wav_complete_groups": 2,
        "same_wav_three_measurable_groups": 1,
        "review_priority_counts": {
            "first_review_same_wav_three_measurable": 1,
            "second_review_three_measurable_different_wavs": 1,
            "needs_better_window_or_manual_boundaries": 1,
            "incomplete_nucleus_family": 1,
        },
        "total_candidates_with_confirmed_left_label": 10,
    }
    m = result["groups"][0]
    assert m["right_surface"] == "m"
    assert m["same_wav_all_measurable"] == ["same.wav"]
    assert {x["author_confirmed_nucleus_class_id"] for x in m["nucleus_candidates"]} == {"li_i", "shi_i", "zi_i"}
    assert m["right_phone_identity"] == "unverified"
    assert result["production_ready"] is False
    assert result == compile_confirmed_nucleus_review(evidence, profile)


def test_old_local_evidence_key_compatible():
    evidence, profile = _fixture()
    evidence["provisional_candidates"] = evidence.pop("candidates")
    for row in evidence["provisional_candidates"]:
        row["spectrum_status"] = row.pop("spectral_status")
        row["centroid_hz"] = row.pop("spectral_centroid_hz")
    report = compile_confirmed_nucleus_review(evidence, profile)
    assert report["summary"]["same_wav_three_measurable_groups"] == 1


def test_wrong_source_fingerprint_fails():
    evidence, profile = _fixture()
    profile["source_oto_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        compile_confirmed_nucleus_review(evidence, profile)


def test_absent_author_confirmation_or_colliding_identity_fails():
    evidence, profile = _fixture()
    profile["labels"][0]["author_confirmed"] = False
    with pytest.raises(ValueError, match="unconfirmed"):
        compile_confirmed_nucleus_review(evidence, profile)
    _, profile = _fixture()
    profile["labels"][1]["nucleus_class_id"] = "li_i"
    with pytest.raises(ValueError, match="duplicate"):
        compile_confirmed_nucleus_review(evidence, profile)


def test_original_alias_must_be_preserved():
    evidence, profile = _fixture()
    evidence["candidates"][0]["alias"] = "D4i0 m"
    with pytest.raises(ValueError, match="inconsistent"):
        compile_confirmed_nucleus_review(evidence, profile)


def test_csv_preserves_boundary_review_unverified(tmp_path):
    evidence, profile = _fixture()
    report = compile_confirmed_nucleus_review(evidence, profile)
    dest = tmp_path / "review"
    write_confirmed_nucleus_review(report, dest)
    data = json.loads((dest / "confirmed_nucleus_groups.json").read_text(encoding="utf-8"))
    assert data["summary"]["complete_surface_groups"] == 3
    csv_text = (dest / "priority_comparison_queue.csv").read_text(encoding="utf-8")
    assert "false,false" in csv_text
    assert "same.wav" in csv_text
    with pytest.raises(FileExistsError):
        write_confirmed_nucleus_review(report, dest)
