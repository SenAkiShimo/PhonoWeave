"""Review decisions -> explicitly approved recording candidates (read-only).

The source action plan remains immutable. Human decisions are tied to an
exact SHA-256 of the plan and to individual pitch cells. Nothing is silently
approved, no new alias is generated, and this does not edit oto.ini/WAVs.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

PLAN_SCHEMA = "phonoweave.recording_scheme_action_plan.v0.1"
REVIEW_SCHEMA = "phonoweave.recording_scheme_decisions.v0.1"
OUTPUT_SCHEMA = "phonoweave.reviewed_recording_scheme.v0.1"

RESOLUTIONS = {
    "review_cross_pitch_coverage": {"pending", "present_verified", "missing_verified"},
    "review_context_mapping": {"pending", "mapping_verified", "insufficient_evidence"},
    "review_role_scoped_mapping": {"pending", "mapping_verified", "insufficient_evidence"},
    "review_internal_candidate_mapping": {
        "pending", "candidate_accepted", "candidate_rejected"
    },
    "review_provisional_unit_decision": {
        "pending", "split_confirmed", "merge_confirmed", "inconclusive"
    },
}


def _hash_plan(plan: dict) -> str:
    payload = json.dumps(plan, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _index_plan(plan: dict):
    if plan.get("schema") != PLAN_SCHEMA or not plan.get("speaker_id"):
        raise ValueError("unsupported or incomplete action plan")
    tasks = plan.get("tasks")
    diagnostics = plan.get("optional_diagnostics")
    if not isinstance(tasks, list) or not isinstance(diagnostics, list):
        raise ValueError("invalid action plan tasks/diagnostics")
    indices, lines = {}, {}
    for task in tasks:
        tid = task["task_id"]
        pitches = task["pitch_groups"]
        if (tid in indices or task.get("review_action") not in RESOLUTIONS
                or not pitches or len(pitches) != len(set(pitches))):
            raise ValueError("invalid or duplicate plan task")
        for pitch in pitches:
            key = (tid, pitch)
            indices[key] = task
    for row in diagnostics:
        line = row["recording_line"]
        if not line or line in lines:
            raise ValueError("empty or duplicate diagnostic line")
        lines[line] = row
    if (len(tasks) != plan.get("consolidated_tasks")
            or len(diagnostics) != plan.get("optional_diagnostic_lines")
            or len(indices) != plan.get("source_review_rows")):
        raise ValueError("action plan count mismatch")
    return indices, lines


def make_decisions_template(plan: dict) -> dict:
    indices, lines = _index_plan(plan)
    return {
        "schema": REVIEW_SCHEMA,
        "speaker_id": plan["speaker_id"],
        "source_plan_sha256": _hash_plan(plan),
        "reviewer_id": "",
        "task_reviews": [
            {"task_id": tid, "pitch_group": pitch, "resolution": "pending",
             "note": "", "evidence_reference": ""}
            for tid, pitch in indices
        ],
        "diagnostic_reviews": [
            {"recording_line": line, "decision": "defer",
             "related_task_id": "", "note": ""}
            for line in lines
        ],
    }


def compile_decisions(plan: dict, reviews: dict) -> dict:
    """Fail on forged/stale approvals; all 123 pitch cells remain explicit."""
    indices, lines = _index_plan(plan)
    if (reviews.get("schema") != REVIEW_SCHEMA or
            reviews.get("speaker_id") != plan["speaker_id"] or
            reviews.get("source_plan_sha256") != _hash_plan(plan)):
        raise ValueError("decision template does not match this exact source plan")
    task_reviews = reviews.get("task_reviews")
    diagnostic_reviews = reviews.get("diagnostic_reviews")
    if not isinstance(task_reviews, list) or not isinstance(diagnostic_reviews, list):
        raise ValueError("review lists required")
    current, resolved, pending = {}, [], []
    for row in task_reviews:
        key = (row.get("task_id"), row.get("pitch_group"))
        if key not in indices or key in current:
            raise ValueError(f"unknown or duplicate pitch review: {key}")
        task = indices[key]
        resolution = row.get("resolution")
        if resolution not in RESOLUTIONS[task["review_action"]]:
            raise ValueError(f"invalid review resolution for {key}")
        note, evidence = row.get("note"), row.get("evidence_reference")
        if not isinstance(note, str) or not isinstance(evidence, str):
            raise ValueError("review note and evidence reference must be strings")
        if resolution != "pending" and (not note.strip() or not evidence.strip()):
            raise ValueError("resolved pitch review requires note and evidence reference")
        if resolution == "candidate_accepted" and not any(
                s.get("pitch_group") == key[1] for s in task["source_examples"]):
            raise ValueError("accepted internal candidate has no source example")
        item = {
            "task_id": key[0], "pitch_group": key[1],
            "unit_id": task["unit_id"], "base_unit": task["base_unit"],
            "review_action": task["review_action"], "resolution": resolution,
            "reviewer_note": note, "evidence_reference": evidence,
            "status": "pending" if resolution == "pending" else "human_review_recorded",
        }
        current[key] = item
        (pending if resolution == "pending" else resolved).append(item)
    if set(current) != set(indices):
        raise ValueError("missing pitch-specific task review")

    seen, approved, deferred, rejected = set(), [], [], []
    valid_tasks = {task["task_id"]: task for task in plan["tasks"]}
    for row in diagnostic_reviews:
        line = row.get("recording_line")
        if line not in lines or line in seen:
            raise ValueError("unknown or duplicate diagnostic recording line")
        seen.add(line)
        decision = row.get("decision")
        note = row.get("note")
        task_id = row.get("related_task_id")
        if decision not in {"defer", "approve", "reject"}:
            raise ValueError("invalid diagnostic decision")
        if not isinstance(note, str) or not isinstance(task_id, str):
            raise ValueError("diagnostic note and related task ID must be strings")
        if decision != "defer" and not note.strip():
            raise ValueError("diagnostic approval/rejection requires a rationale")
        if decision == "approve":
            task = valid_tasks.get(task_id)
            if task is None or task["base_unit"] != lines[line]["base_unit"]:
                raise ValueError("approved diagnostic must link to a task for the same base")
            if not any(
                item["task_id"] == task_id for item in resolved
            ):
                raise ValueError("approved diagnostic requires reviewed task evidence")
        item = {
            "recording_line": line, "syllable": lines[line]["syllable"],
            "base_unit": lines[line]["base_unit"],
            "decision": decision, "related_task_id": task_id, "reviewer_note": note,
            "scope": "diagnostic_only_no_production_alias",
        }
        (approved if decision == "approve" else
         rejected if decision == "reject" else deferred).append(item)
    if seen != set(lines):
        raise ValueError("missing diagnostic review")
    reviewer = reviews.get("reviewer_id")
    if not isinstance(reviewer, str) or (
            (resolved or approved or rejected) and not reviewer.strip()):
        raise ValueError("recorded decisions require reviewer_id")
    # An approved diagnostic is a human recording proposal, never a
    # production alias assignment or an inference of missing WAVs.
    return {
        "schema": OUTPUT_SCHEMA,
        "speaker_id": plan["speaker_id"],
        "source_plan_sha256": _hash_plan(plan),
        "reviewer_id": reviewer,
        "status": "partial_human_review_no_production_export",
        "reviewed_pitch_cells": len(resolved),
        "pending_pitch_cells": len(pending),
        "approved_diagnostic_lines": len(approved),
        "rejected_diagnostic_lines": len(rejected),
        "deferred_diagnostic_lines": len(deferred),
        "recorded_resolutions": dict(sorted(Counter(
            item["resolution"] for item in resolved).items())),
        "resolved_pitch_reviews": resolved,
        "pending_pitch_reviews": pending,
        "approved_diagnostics": approved,
        "rejected_diagnostics": rejected,
        "deferred_diagnostics": deferred,
        "production_alias_export": "disabled",
        "oto_timing_changes": "none",
    }


def save_review_template(plan: dict, path: Path) -> None:
    template = make_decisions_template(plan)
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(template, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_reviewed_scheme(result: dict, output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phonoweave-reviewed-",
                                     dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "reviewed_scheme.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        approved_lines = [x["recording_line"] for x in result["approved_diagnostics"]]
        (stage / "approved_diagnostic_reclist.txt").write_text(
            "".join(line + "\n" for line in approved_lines), encoding="utf-8",
        )
        (stage / "README.txt").write_text(
            "HUMAN-APPROVED DIAGNOSTIC PROPOSALS ONLY; NOT A PRODUCTION RECLIST.\n"
            "No original WAV or OTO files have been modified.\n"
            "No alias mapping or OTO timings have been generated.\n"
            "The reclist contains only manually approved pre-existing candidates.\n"
            "An empty reclist is valid when no approvals have been given.\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError("output directory appeared during compilation")
        os.rename(stage, output)
    return output
