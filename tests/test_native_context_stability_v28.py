"""v2.8 source-template isolation and known-label paired-context diagnostics."""
from __future__ import annotations

import copy

import pytest

from phonoweave.native_context_stability_v28 import (
    SCHEMA, evaluate_existing_labels, attach_context_review,
)


LABELS = ["i", "ir", "i0"]


def fixture(*, banks=("D4", "C4"), templates=("x.wav", "y.wav", "z.wav"),
            contexts=("a", "m", "k")):
    data = []
    for bank in banks:
        for t, wav in enumerate(templates):
            for c, context in enumerate(contexts):
                for i, label in enumerate(LABELS):
                    # Context-dependent common offset, plus a stable contrast
                    # that persists under complete source-WAV holdout.
                    x = [5.0*c + 0.3*t + float(i)*8.0,
                         2.0*c - 0.2*t - float(i)*5.0] + [float(i)]*10
                    data.append({
                        "bank": bank, "context": context, "symbol": label,
                        "source_pcm_sha256": (str(t+1) if bank == "D4" else chr(ord("a")+t))*64,
                        "source_wav": wav, "features": x,
                    })
    return data


def test_source_disjoint_with_exact_expected_denominators():
    report = evaluate_existing_labels(fixture(), LABELS, null_repetitions=4)
    assert report["schema"] == SCHEMA
    assert report["complete_paired_source_context_groups"] == 18
    assert report["heldout_template_folds"] == 6
    assert report["eligible_label_decisions"] == 54
    assert report["raw_correct"] == 54
    assert report["paired_context_centered_correct"] == 54
    assert report["training_label_permutation_negative_control"]["repetitions"] == 4
    assert report["automatic_new_recordings"] == []
    assert report["automatic_unit_merges_or_splits"] == []
    for b in report["per_bank"]:
        for f in b["heldout_source_fold_details"]:
            assert f["training_distinct_wav_templates"] == 2
            assert f["heldout_source_pcm_count"] == 1


def test_no_false_replication_when_one_wav_template():
    report = evaluate_existing_labels(
        fixture(templates=("x.wav",)), LABELS, null_repetitions=0)
    assert report["eligible_label_decisions"] == 0
    assert report["raw_accuracy"] is None
    assert report["review_status"] == "insufficient_distinct_wav_template_folds"
    assert report["automatic_new_recordings"] == []


def test_repeated_source_wav_alias_cannot_create_train_source():
    samples = fixture()
    # Four duplicated OTO rows per label/source would be invalid as one
    # unambiguous source-linked triplet: exclude such contexts, not train on
    # fake independent examples.
    samples.extend(copy.deepcopy(samples[:3]))
    report = evaluate_existing_labels(samples, LABELS, null_repetitions=0)
    assert report["ambiguous_or_duplicate_paired_groups"] == 1
    assert report["complete_paired_source_context_groups"] == 17


def test_source_pcm_conflict_in_same_wav_is_rejected():
    data = fixture()
    data[1]["source_pcm_sha256"] = "e"*64
    with pytest.raises(ValueError, match="conflicting PCM"):
        evaluate_existing_labels(data, LABELS)


def test_duplicate_pcm_copied_to_different_wav_is_rejected():
    data = fixture()
    data[27]["source_pcm_sha256"] = data[0]["source_pcm_sha256"]
    with pytest.raises(ValueError, match="conflicting PCM|share decoded PCM"):
        evaluate_existing_labels(data, LABELS)


@pytest.mark.parametrize("value", (True, -1, 0, 1, 21, 2.5))
def test_train_source_requirement_must_be_valid_int(value):
    with pytest.raises(ValueError, match="min_train_source_templates"):
        evaluate_existing_labels(fixture(), LABELS,
                                 min_train_source_templates=value)


@pytest.mark.parametrize("value", (True, -1, 129, 1.2))
def test_null_count_rejected(value):
    with pytest.raises(ValueError, match="null_repetitions"):
        evaluate_existing_labels(fixture(), LABELS, null_repetitions=value)


def test_null_seed_is_deterministic():
    rows = fixture()
    first = evaluate_existing_labels(rows, LABELS,
                                    null_repetitions=8, random_seed=123)
    second = evaluate_existing_labels(rows, LABELS,
                                     null_repetitions=8, random_seed=123)
    assert first["training_label_permutation_negative_control"] == second[
        "training_label_permutation_negative_control"]
    assert first["paired_confusion_actual_rows_predicted_columns"] == second[
        "paired_confusion_actual_rows_predicted_columns"]


def test_context_common_shift_is_removed_by_known_paired_centering():
    rows = fixture(contexts=("a", "m", "k", "z"))
    report = evaluate_existing_labels(rows, LABELS, null_repetitions=0)
    assert report["paired_context_centered_accuracy"] == 1.0


def test_original_unicode_labels_preserved_without_invented_phone():
    rows = fixture()
    replacements = dict(zip(LABELS, ["陰", "i²", "ζ"]))
    for r in rows:
        r["symbol"] = replacements[r["symbol"]]
    report = evaluate_existing_labels(rows, list(replacements.values()),
                                      null_repetitions=0)
    assert report["original_author_labels"] == ["陰", "i²", "ζ"]
    assert report["automatic_unit_merges_or_splits"] == []


def test_diagnostic_attaches_without_recording_plan_mutation(monkeypatch):
    from phonoweave import native_context_stability_v28 as module
    original = {"recording_plan": {"recording_lines": ["a"], "review_tasks": ["fix OTO"]},
                "native_author_feedback_review": {
                    "original_user_labels": LABELS,
                    "author_confirmation": {
                        "desire_to_keep_existing_realization_labels_distinct": "yes"
                    },
                }}
    unchanged = copy.deepcopy(original)
    monkeypatch.setattr(
        module, "run_archive",
        lambda archive, labels, null_repetitions=32:
        evaluate_existing_labels(fixture(), labels, null_repetitions=0),
    )
    result = attach_context_review(original, "ignored.zip", LABELS)
    assert original == unchanged
    assert result["recording_plan"] == unchanged["recording_plan"]
    assert result["existing_native_context_stability"]["automatic_new_recordings"] == []
    with pytest.raises(ValueError, match="must not be overwritten"):
        attach_context_review(result, "ignored.zip", LABELS)
    with pytest.raises(ValueError, match="disagree"):
        attach_context_review(original, "ignored.zip", ["i", "ir", "new"])


def test_user_unconfirmed_distinctions_cannot_get_assumed_as_approved():
    report = {
        "recording_plan": {},
        "native_author_feedback_review": {
            "original_user_labels": LABELS,
            "author_confirmation": {
                "desire_to_keep_existing_realization_labels_distinct": "uncertain"
            },
        },
    }
    with pytest.raises(ValueError, match="not confirmed"):
        attach_context_review(report, "ignored.zip", LABELS)
