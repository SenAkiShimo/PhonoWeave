"""Bridge existing author-native acoustic evidence to *non-destructive* unit reviews.

No IPA conversion, model fitting, auto new aliases, or recording tasks.
Review existing realization labels only; OTO-relative/model-rendered windows
cannot establish independent human phones or synthesis value.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

SCHEMA = "phonoweave.existing_native_unit_review.v2.4"
SOURCE_SCHEMA = "phonoweave.user_native_acoustic_evidence.v1"
SOURCE_ORIGIN = "author_confirmed_diffSinger_renders_not_independent_human_singers"
SPLITS = (
    ("cross_context_predictions", "cross_context_top1"),
    ("leave_pitch_out_predictions", "leave_pitch_out_top1"),
    ("leave_style_out_predictions", "leave_style_out_top1"),
)
# Review heuristics; NOT calibrated phone-identity or synthesis acceptance thresholds.
MIN_REPORTS = 2
MIN_TRIALS = 6
REVIEW_FRACTION = 0.85


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _validate(report: dict[str, Any]) -> dict:
    if not isinstance(report, dict) or report.get("schema") != SOURCE_SCHEMA:
        raise ValueError("expected existing user-native acoustic evidence v1")
    if (report.get("pitch_context_origin") != SOURCE_ORIGIN
            or report.get("unit_decision") !=
            "insufficient_independent_sources_for_auto_alias_change"
            or report.get("recommended_new_user_aliases") != []
            or report.get("source_audio_modified") is not False):
        raise ValueError("unsupported acoustic evidence provenance or unit decision")
    labels = report.get("user_realization_labels")
    if not isinstance(labels, list) or not labels or not all(map(_nonempty, labels)):
        raise ValueError("nonempty opaque original user labels required")
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate user labels")
    if not _nonempty(report.get("language_id")) or not _nonempty(report.get("original_user_base_symbol")):
        raise ValueError("original user base symbol and language required")
    counts = report.get("per_label_count")
    if not isinstance(counts, dict) or set(counts) != set(labels):
        raise ValueError("per-label counts inconsistent with original labels")
    if any(type(counts[x]) is not int or counts[x] < 1 for x in labels):
        raise ValueError("invalid candidate-window counts")
    if type(report.get("candidate_windows")) is not int or sum(counts.values()) != report["candidate_windows"]:
        raise ValueError("candidate-window total inconsistent with label counts")
    records = {}
    for field, aggregate_field in SPLITS:
        predictions = report.get(field)
        aggregate = report.get(aggregate_field)
        if not isinstance(predictions, list) or not isinstance(aggregate, dict):
            raise ValueError("missing original transfer predictions")
        trial_keys, per_label = set(), defaultdict(lambda: [0, 0])
        for row in predictions:
            if not isinstance(row, dict):
                raise ValueError("invalid transfer prediction")
            original, nearest = row.get("actual_user_symbol"), row.get("nearest_user_symbol")
            if original not in labels or nearest not in labels or type(row.get("matched")) is not bool:
                raise ValueError("transfer prediction has invalid original label or match")
            if row["matched"] != (original == nearest):
                raise ValueError("contradictory prediction and match field")
            if not all(_nonempty(row.get(k)) for k in ("bank", "style", "pitch", "context")):
                raise ValueError("missing source/style/context identity")
            key = tuple(row[k] for k in ("bank", "style", "pitch", "context", "actual_user_symbol"))
            if key in trial_keys:
                raise ValueError("duplicate transfer trial identity")
            trial_keys.add(key)
            per_label[original][1] += 1
            per_label[original][0] += int(row["matched"])
        n = len(predictions)
        if type(aggregate.get("total")) is not int or aggregate["total"] != n:
            raise ValueError("transfer denominator mismatch")
        if type(aggregate.get("correct")) is not int or aggregate["correct"] != sum(x[0] for x in per_label.values()):
            raise ValueError("transfer numerator mismatch")
        reported_acc = aggregate.get("accuracy")
        if n:
            if type(reported_acc) not in (float, int) or abs(reported_acc - aggregate["correct"]/n) > 0.00011:
                raise ValueError("transfer accuracy mismatch")
        elif reported_acc is not None:
            raise ValueError("empty transfer must not claim accuracy")
        if any(per_label[label][1] > counts[label] for label in labels):
            raise ValueError("transfer has more trials than source windows")
        records[field] = {"keys": trial_keys, "per_label": dict(per_label), "rows": predictions}
    return records


def summarize_existing_realizations(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize uncertainty for EXISTING labels, never create new units.

    Multiple reports are descriptive sensitivity checks only. The v1 report
    format has no source-PCM hash, so identical trial IDs cannot prove genuine
    independent repetitions or different extraction windows.
    """
    if not isinstance(reports, list) or not 1 <= len(reports) <= 5:
        raise ValueError("supply one to five existing-evidence reports")
    # Repeated copies of one report are not repeated window sensitivity evidence.
    fingerprints = [sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                    for x in reports]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("duplicate acoustic evidence content is not an independent sensitivity check")
    all_records = [_validate(x) for x in reports]
    first = reports[0]
    labels = first["user_realization_labels"]
    for report in reports[1:]:
        if (report["language_id"] != first["language_id"] or
                report["original_user_base_symbol"] != first["original_user_base_symbol"] or
                report["user_realization_labels"] != labels):
            raise ValueError("cannot combine different languages, bases or original labels")
    comparable = all(
        rec[field]["keys"] == all_records[0][field]["keys"]
        for rec in all_records[1:] for field, _ in SPLITS
    )
    rows = []
    for label in labels:
        by_method = {}
        for field, _ in SPLITS:
            scores = []
            for rec in all_records:
                good, total = rec[field]["per_label"].get(label, [0, 0])
                scores.append({"correct": good, "trials": total,
                               "rate": round(good/total, 4) if total else None})
            by_method[field.replace("_predictions", "")] = {
                "per_report": scores,
                "minimum_rate_across_reports": min((x["rate"] for x in scores if x["rate"] is not None), default=None),
            }
        coverage = all_records[0]["cross_context_predictions"]["rows"]
        cases = [x for x in coverage if x["actual_user_symbol"] == label]
        banks = len({x["bank"] for x in cases})
        styles = len({x["style"] for x in cases})
        contexts = len({x["context"] for x in cases})
        has_enough = all(
            all(x["trials"] >= MIN_TRIALS for x in data["per_report"]) and
            data["minimum_rate_across_reports"] is not None and
            data["minimum_rate_across_reports"] >= REVIEW_FRACTION
            for data in by_method.values()
        )
        stable_screen = (len(reports) >= MIN_REPORTS and comparable and
                         len(labels) >= 2 and has_enough and
                         banks >= 2 and styles >= 2 and contexts >= 2)
        rows.append({
            "original_user_label": label,
            "candidate_windows_in_primary_report": first["per_label_count"][label],
            "observed_banks": banks,
            "observed_rendered_styles": styles,
            "observed_following_contexts": contexts,
            "transfer": by_method,
            "review_status": (
                "existing_label_model_rendered_distinction_consistent"
                if stable_screen else "insufficient_or_window_sensitive_evidence"
            ),
            "next_check": (
                "compare_existing_realizations_in_equal_budget_synthesis_and_independent_takes"
                if stable_screen else "inspect_source_context_confound_and_validate_existing_label"
            ),
            "automatic_new_recording_lines": [],
            "new_phoneme_authorized": False,
        })
    return {
        "schema": SCHEMA,
        "language_id": first["language_id"],
        "base_user_symbol": first["original_user_base_symbol"],
        "original_user_labels": labels[:],
        "report_count": len(reports),
        "same_trial_identities_across_reports": comparable,
        "trial_identity_not_pcm_proof": True,
        "evidence_origin": "one_singer_model_rendered_styles_not_independent_human_takes",
        "status": "review_existing_realizations_only",
        "screening_heuristic_not_validated_threshold": {"min_reports": MIN_REPORTS,
                                                        "min_trials_per_label_and_test": MIN_TRIALS,
                                                        "min_descriptive_recall": REVIEW_FRACTION},
        "realizations": rows,
        "automatic_new_synthesis_units": [],
        "automatic_alias_edits": [],
        "automatic_new_recording_lines": [],
        "limitations": [
            "Known author labels supervised classification; not unknown phoneme discovery.",
            "v1 trial keys lack source PCM checksums and cannot prove distinct human takes.",
            "Model-rendered styles and OTO-derived windows cannot prove synthesis usefulness.",
            "No unit merge, split, or extra recording authorized by this evidence.",
        ],
    }


def attach_existing_unit_review(report: dict, paths: list[Path]) -> dict:
    """Attach *separate* review evidence without changing a recording plan."""
    if not isinstance(report, dict) or not isinstance(report.get("recording_plan"), dict):
        raise ValueError("recording report required")
    if not isinstance(paths, list) or not 1 <= len(paths) <= 5:
        raise ValueError("one to five evidence paths required")
    source_files = [Path(p).expanduser().resolve(strict=True) for p in paths]
    if len(set(source_files)) != len(source_files):
        raise ValueError("duplicate evidence path")
    raw = [p.read_bytes() for p in source_files]
    evidence = [json.loads(x) for x in raw]
    review = summarize_existing_realizations(evidence)
    # Preserve native coverage status but never infer a mapping of i, i0, etc.
    # onto arbitrary recording lines / custom WAV names.
    review["input_report_sha256"] = [sha256(x).hexdigest() for x in raw]
    result = {**report, "existing_sound_unit_review": review}
    if result["recording_plan"] != report["recording_plan"]:
        raise AssertionError("review must not modify the recording plan")
    return result
