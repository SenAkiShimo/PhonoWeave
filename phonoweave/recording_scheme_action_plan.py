"""Consolidated operational recording-scheme review plan.

Transforms pitch-level evidence review rows into actionable cross-pitch
checkpoints with source examples. Never orders a recording or writes oto.ini.
"""
from __future__ import annotations

from collections import Counter, defaultdict

SCHEMA = "phonoweave.recording_scheme_action_plan.v0.1"

CHECKS = {
    "review_cross_pitch_coverage": (
        "Compare the named CV alias in the affected pitch against the "
        "corresponding OTO entries in other pitches. Inspect the source "
        "recording before deciding whether any sample is absent."
    ),
    "review_role_scoped_mapping": (
        "Inspect the original reclist, relevant WAV and OTO segmentation for "
        "the internal context. No sequence-verified candidate currently "
        "supports assignment; do not mark a missing recording."
    ),
    "review_internal_candidate_mapping": (
        "Listen to and inspect the linked original WAV/OTO boundary. Verify "
        "that the proposed internal consonant and following-vowel context "
        "match the source before accepting the candidate link."
    ),
    "review_context_mapping": (
        "Inspect existing CV/VC aliases for this base consonant and establish "
        "whether a distinct inventory context can be assigned. Absence of "
        "a direct link does not establish absence of recorded audio."
    ),
    "review_provisional_unit_decision": (
        "Check the split/merge decision against existing coverage and any "
        "independent synthesis evidence before adopting it as a recording "
        "design rule."
    ),
}


def build_action_plan(
    speaker_id: str,
    coverage_rows: list[dict],
    review_queue: list[dict],
    internal_links: list[dict],
    supplemental_targets: list[dict],
) -> dict:
    """Collapse repeated pitch tasks without losing pitch-level evidence."""
    cells = {}
    for row in coverage_rows:
        key = (row["unit_id"], row["pitch_group"])
        if key in cells:
            raise ValueError("duplicate unit/pitch coverage cell")
        cells[key] = row
    groups = defaultdict(list)
    for row in review_queue:
        key = (row["unit_id"], row["pitch_group"])
        cell = cells.get(key)
        if cell is None or cell["recommended_review_action"] != row["review_action"]:
            raise ValueError("review queue does not agree with coverage matrix")
        if row["review_action"] not in CHECKS:
            raise ValueError("unknown review action")
        groups[(row["review_action"], row["unit_id"])].append(row)
    linked = defaultdict(list)
    for link in internal_links:
        key = (link["unit_id"], link["pitch_group"])
        if key not in cells:
            raise ValueError("internal link refers to absent coverage cell")
        linked[key].append(link)

    tasks = []
    for (action, uid), rows in groups.items():
        rows.sort(key=lambda r: r["pitch_group"])
        pitches = [r["pitch_group"] for r in rows]
        all_syllables = sorted({
            word for row in rows for word in
            row.get("suggested_syllables", "").split("|") if word
        })
        examples = []
        for row in rows:
            key = (uid, row["pitch_group"])
            for link in sorted(linked[key], key=lambda x: (
                    x["source_oto"], x["line_number"],
                    x["source_wav"]))[:1]:
                examples.append({
                    "pitch_group": row["pitch_group"],
                    "source_oto": link["source_oto"],
                    "line_number": link["line_number"],
                    "source_wav": link["source_wav"],
                    "original_alias": link["original_alias"],
                    "syllable": link["syllable"],
                    "review_status": link["review_status"],
                })
        tasks.append({
            "task_id": "",  # Filled after deterministic sort.
            "priority": int(rows[0]["priority"]),
            "review_action": action,
            "unit_id": uid,
            "base_unit": rows[0]["base_unit"],
            "pitch_groups": pitches,
            "suggested_syllables": all_syllables,
            "affected_pitch_cells": len(rows),
            "source_examples": examples,
            "verification": CHECKS[action],
            "status": "manual_review_pending",
            "recording_required": "undetermined",
        })
    tasks.sort(key=lambda x: (
        x["priority"], x["review_action"], x["base_unit"], x["unit_id"]
    ))
    for index, task in enumerate(tasks, 1):
        task["task_id"] = f"RS-{index:03d}"

    supplements = [
        {
            "recording_line": row["recording_line"],
            "syllable": row["syllable"],
            "base_unit": row["base_unit"],
            "context_family": row["context_family"],
            "role_scope": row.get("role_scope") or "",
            "existing_observations": row["existing_observations"],
            "status": "optional_diagnostic_candidate_not_recording_order",
        }
        for row in supplemental_targets
    ]
    return {
        "schema": SCHEMA,
        "speaker_id": speaker_id,
        "action_plan_status": "human_review_required",
        "source_pitch_cells": len(coverage_rows),
        "source_review_rows": len(review_queue),
        "consolidated_tasks": len(tasks),
        "action_counts": dict(sorted(Counter(
            t["review_action"] for t in tasks
        ).items())),
        "tasks": tasks,
        "optional_diagnostics": supplements,
        "optional_diagnostic_lines": len(supplements),
        "recordings_automatically_authorized": 0,
        "alias_or_oto_modifications": 0,
        "limitations": [
            "Context links are candidate evidence from WAV names and OTO order.",
            "Pitch-level coverage is not independent take or acoustic quality evidence.",
            "Optional diagnostic lines are not an approved production reclist.",
            "Any confirmed new recording requires a separate reviewed decision.",
        ],
    }


def _safe(text: object) -> str:
    """Keep external alias strings printable without Markdown/code injection."""
    return str(text).replace("\n", " ").replace("\r", " ").replace(
        "`", "'").replace("|", "/")


def action_plan_markdown(plan: dict) -> str:
    """Human-readable decision order with exact units and source references."""
    lines = [
        "# PhonoWeave recording-scheme review plan",
        "",
        f"Speaker: `{_safe(plan['speaker_id'])}`",
        "",
        "This is a prioritized source-audit plan, **not** an automatic "
        "request to record, install aliases, or change OTO timing.",
        "",
        f"{plan['source_review_rows']} pitch-specific queue rows condensed "
        f"to **{plan['consolidated_tasks']} actionable checks**.",
        "",
    ]
    for priority, heading in [
        (1, "Check apparent pitch-coverage differences"),
        (2, "Resolve context and sequence evidence"),
        (3, "Review provisional split/merge decisions"),
        (4, "Other checks"),
    ]:
        subset = [t for t in plan["tasks"] if t["priority"] == priority]
        if not subset:
            continue
        lines.extend([f"## {heading}", ""])
        for task in subset:
            words = ", ".join(task["suggested_syllables"]) or "not specified"
            pitches = ", ".join(task["pitch_groups"])
            lines.extend([
                f"### {task['task_id']} — `{_safe(task['unit_id'])}`",
                "",
                f"- Check: `{task['review_action']}`",
                f"- Pitch group(s): {pitches}",
                f"- Related syllables: {words}",
                f"- Method: {task['verification']}",
            ])
            for sample in task["source_examples"]:
                lines.append(
                    "- Source example: "
                    f"`{_safe(sample['source_oto'])}:{sample['line_number']}`, "
                    f"`{_safe(sample['original_alias'])}`, "
                    f"WAV `{_safe(sample['source_wav'])}` "
                    f"({sample['pitch_group']}; candidate evidence only)"
                )
            lines.append("")
    lines.extend([
        "## Optional diagnostic recording candidates",
        "",
        f"{plan['optional_diagnostic_lines']} pre-existing diagnostic "
        "targets. These are proposals, not approved recording orders; "
        "first resolve relevant coverage and naming checks.",
        "",
    ])
    for target in plan["optional_diagnostics"]:
        lines.append(
            f"- `{_safe(target['recording_line'])}` "
            f"({target['context_family']}, "
            f"{target['status']})"
        )
    lines.extend([
        "",
        "## Safeguards",
        "",
        "- Original voicebank WAV and OTO files are never altered.",
        "- No new alias is declared import-ready.",
        "- No absent recording is inferred from a missing alias or candidate link.",
        "- Human review must decide whether any additional recording is justified.",
        "",
    ])
    return "\n".join(lines)
