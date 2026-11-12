"""Apply the author's v2.6 acceptance feedback without scheduling any recording.

A confirmed original alias and desire to keep current realizations separate are
*author intent*, not acoustic boundary or independent-session evidence. Unknown
recordability remains unknown. Other archived contexts are preview-only.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .native_unit_budget_v25 import _read_archive
from .native_validation_preview_v26 import SCHEMA as V26, build_preview

SCHEMA = "phonoweave.native_author_feedback_gate.v2.7"
FEEDBACK_SCHEMA = "phonoweave.native_author_feedback.v2.7"
ALLOWED = {"yes", "no", "uncertain"}


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        c not in "0123456789abcdef" for c in value
    ):
        raise ValueError("invalid SHA-256 in " + field)
    return value


def _validate_preview(preview: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(preview, dict) or preview.get("schema") != V26:
        raise ValueError("expected original v2.6 validation preview")
    labels = preview.get("original_user_labels")
    cases = preview.get("selected_review_cases")
    if (not isinstance(labels, list) or not 2 <= len(labels) <= 5
            or not all(isinstance(s, str) and s.strip() for s in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("invalid original author labels")
    if (not isinstance(cases, list) or not 1 <= len(cases) <= 8
            or type(preview.get("selected_count")) is not int
            or preview["selected_count"] != len(cases)):
        raise ValueError("invalid original review-case count")
    _digest(preview.get("source_archive_sha256"), "preview source archive")
    if preview.get("new_recording_lines") != [] or preview.get("automatic_unit_merges_or_splits") != []:
        raise ValueError("source preview unexpectedly authorizes unit or recording changes")
    seen_cases, seen_sources = set(), set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("invalid original review case")
        number = case.get("case_number")
        if type(number) is not int or number < 1 or number in seen_cases:
            raise ValueError("duplicate or invalid review case number")
        seen_cases.add(number)
        if not all(isinstance(case.get(k), str) and case[k] for k in
                   ("bank", "following_context", "source_wav", "source_pcm_sha256")):
            raise ValueError("missing original source identity")
        _digest(case["source_pcm_sha256"], "preview source PCM")
        source_key = (case["bank"], case["following_context"], case["source_pcm_sha256"])
        if source_key in seen_sources:
            raise ValueError("duplicate physical review source")
        seen_sources.add(source_key)
        examples = case.get("examples")
        if (not isinstance(examples, list) or len(examples) != len(labels) or
                [x.get("original_label") for x in examples if isinstance(x, dict)] != labels):
            raise ValueError("review examples differ from author-original labels")
        for example in examples:
            if (not isinstance(example.get("original_alias"), str)
                    or not example["original_alias"]
                    or example.get("source_wav") != case["source_wav"]):
                raise ValueError("inconsistent original alias/source WAV mapping")
    if seen_cases != set(range(1, len(cases) + 1)):
        raise ValueError("noncontiguous review-case identifiers")
    return cases


def evaluate_author_feedback(
    preview: dict[str, Any],
    feedback: dict[str, Any], *,
    preview_file_sha256: str,
    source_rows: list[dict[str, Any]] | None = None,
    source_archive_sha256: str | None = None,
    max_alternatives: int = 3,
) -> dict[str, Any]:
    """No new recording lines, even if an author marks a context recordable."""
    cases = _validate_preview(preview)
    if not isinstance(feedback, dict) or feedback.get("schema") != FEEDBACK_SCHEMA:
        raise ValueError("invalid author feedback schema")
    if _digest(preview_file_sha256, "preview file") != feedback.get("preview_sha256"):
        raise ValueError("feedback is not tied to this exact v2.6 preview")
    if type(max_alternatives) is not int or not 0 <= max_alternatives <= 8:
        raise ValueError("max_alternatives must be an integer from 0 to 8")
    alias = feedback.get("alias_mapping_correct")
    distinct = feedback.get("existing_realizations_should_remain_distinct")
    if alias not in ALLOWED or distinct not in ALLOWED:
        raise ValueError("author acceptance choices must be yes/no/uncertain")
    opinions = feedback.get("case_recordability")
    if not isinstance(opinions, list) or len(opinions) != len(cases):
        raise ValueError("recordability must cover exactly the reviewed cases")
    statuses = {}
    for x in opinions:
        if not isinstance(x, dict) or set(x) != {"case_number", "status"}:
            raise ValueError("invalid case recordability entry")
        n, status = x["case_number"], x["status"]
        if type(n) is not int or n in statuses or n not in {c["case_number"] for c in cases}:
            raise ValueError("invalid or duplicate author case reference")
        if status not in ALLOWED:
            raise ValueError("invalid context recordability status")
        statuses[n] = status
    if set(statuses) != {x["case_number"] for x in cases}:
        raise ValueError("case recordability is incomplete")

    if (source_rows is None) != (source_archive_sha256 is None):
        raise ValueError("source rows and archive fingerprint must be provided together")
    alternate = []
    available = None
    if source_rows is not None:
        if _digest(source_archive_sha256, "candidate archive") != preview["source_archive_sha256"]:
            raise ValueError("candidate archive differs from original author preview archive")
        if max_alternatives:
            checked_contexts = {c["following_context"] for c in cases}
            filtered = [r for r in source_rows if r["context"] not in checked_contexts]
            if filtered:
                candidates = build_preview(
                    filtered, preview["original_user_labels"],
                    max_cases=max_alternatives,
                    archive_sha256=source_archive_sha256,
                )
                available = candidates["source_linked_candidate_cases"]
                alternate = [{
                    **case,
                    "author_recordability": "unreviewed",
                    "review_only_not_recording_instruction": True,
                } for case in candidates["selected_review_cases"]]
            else:
                available = 0
    statuses_by_context = [{
        "case_number": case["case_number"],
        "bank": case["bank"],
        "following_context": case["following_context"],
        "source_wav": case["source_wav"],
        "author_recordability": statuses[case["case_number"]],
        "recording_ready": False,
    } for case in cases]
    return {
        "schema": SCHEMA,
        "input_preview_sha256": preview_file_sha256,
        "input_source_archive_sha256": preview["source_archive_sha256"],
        "original_user_labels": preview["original_user_labels"][:],
        "feedback_scope": "only_the_three_or_other_selected_v2.6_cases",
        "author_confirmation": {
            "original_alias_mapping": alias,
            "desire_to_keep_existing_realization_labels_distinct": distinct,
            "scientifically_verified_phonetic_separation": False,
        },
        "reviewed_contexts": statuses_by_context,
        "recording_readiness": (
            "blocked_by_unconfirmed_context_recordability"
            if any(s != "yes" for s in statuses.values())
            else "still_requires_independent_source_and_explicit_recording_plan"
        ),
        "alternative_archived_contexts_for_future_inspection_only": alternate,
        "unreviewed_source_linked_alternative_count": available,
        "next_development_action": (
            "preserve_existing_author_labels_and_search_existing_sources_without_new_recordings"
            if distinct == "yes" and alias == "yes"
            else "resolve_author_label_mapping_before_any_inventory_decision"
        ),
        "automatic_recording_requests": [],
        "automatic_unit_merges": [],
        "automatic_unit_splits": [],
        "original_voicebank_modified": False,
        "warnings": [
            "Unknown recording feasibility is not a rejection or consent to record.",
            "Author preference to keep labels separate is not independent acoustic/phoneme proof.",
            "Alternative contexts are historic OTO-linked aliases, not guaranteed pronounceable scripts.",
            "No new recording is required or authorized by this feedback.",
        ],
    }


def run(
    preview_path: Path,
    feedback_path: Path,
    source_archive: Path | None = None, *,
    max_alternatives: int = 3,
) -> dict[str, Any]:
    raw_preview = Path(preview_path).read_bytes()
    preview = json.loads(raw_preview)
    feedback = json.loads(Path(feedback_path).read_bytes())
    kw = {}
    if source_archive is not None:
        rows, digest = _read_archive(source_archive)
        kw = {"source_rows": rows, "source_archive_sha256": digest}
    return evaluate_author_feedback(
        preview, feedback,
        preview_file_sha256=sha256(raw_preview).hexdigest(),
        max_alternatives=max_alternatives, **kw,
    )


def attach_author_feedback(
    recording_report: dict[str, Any],
    preview_path: Path,
    feedback_path: Path,
    source_archive: Path | None = None,
    *, max_alternatives: int = 3,
) -> dict[str, Any]:
    if not isinstance(recording_report, dict) or not isinstance(recording_report.get("recording_plan"), dict):
        raise ValueError("valid preexisting recording plan is required")
    if "native_author_feedback_review" in recording_report:
        raise ValueError("report already contains author feedback; refusing overwrite")
    review = run(preview_path, feedback_path, source_archive, max_alternatives=max_alternatives)
    return {**recording_report, "native_author_feedback_review": review}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Apply author v2.6 feedback without editing the recording plan")
    p.add_argument("--preview", required=True, type=Path)
    p.add_argument("--feedback", required=True, type=Path)
    p.add_argument("--source-archive", type=Path)
    p.add_argument("--max-alternatives", type=int, default=3)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        p.error("output exists; refusing overwrite")
    result = run(args.preview, args.feedback, args.source_archive,
                 max_alternatives=args.max_alternatives)
    if not output.parent.is_dir():
        p.error("output parent must already exist")
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({
        "recording_readiness": result["recording_readiness"],
        "alternative_review_examples": len(result["alternative_archived_contexts_for_future_inspection_only"]),
        "new_recording_requests": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
