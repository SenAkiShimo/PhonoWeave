"""Read-only presentation of recording-plan review tasks.

This file intentionally does not generate new phonemes or edit WAV/OTO files.
The recording reclist remains separate from repair/source-verification tasks.
"""
from __future__ import annotations

from typing import Any


def grouped_review_tasks(report: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Return strict OTO-config and source-verification queues from a v2.3 report.

    Older/Mandarin reports have no native review-task schema: return empty
    queues rather than interpreting an absence as verified coverage.
    """
    plan = report.get("recording_plan", {})
    tasks = plan.get("review_tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("recording-plan review_tasks must be a list")
    grouped: dict[str, list[dict[str, str]]] = {
        "oto_configuration": [], "source_verification": []
    }
    seen = set()
    for item in tasks:
        if not isinstance(item, dict):
            raise ValueError("invalid review task")
        category = item.get("category")
        if category not in grouped:
            raise ValueError("unknown review task category")
        values = {k: item.get(k) for k in ("line", "bank", "status", "reason")}
        if not all(isinstance(v, str) and v for v in values.values()):
            raise ValueError("review task requires nonempty original label/bank/status/reason")
        key = (category, values["line"], values["bank"], values["status"])
        if key in seen:
            continue
        seen.add(key)
        grouped[category].append({"category": category, **values})
    return grouped


def format_review_tasks(report: dict[str, Any]) -> str:
    """Human-readable review list, always distinct from a new-recording TXT."""
    grouped = grouped_review_tasks(report)
    lines = [
        "PhonoWeave | Review tasks (NOT additional recording instructions)",
        "Original author labels preserved. Verify files/aliases before rerecording.",
        "",
        "OTO configuration review",
    ]
    for task in grouped["oto_configuration"]:
        lines.append(f"- {task['line']} | {task['bank']} | {task['status']} | {task['reason']}")
    if not grouped["oto_configuration"]:
        lines.append("(no OTO configuration tasks reported)")
    lines += ["", "Source / audio verification"]
    for task in grouped["source_verification"]:
        lines.append(f"- {task['line']} | {task['bank']} | {task['status']} | {task['reason']}")
    if not grouped["source_verification"]:
        lines.append("(no source-verification tasks reported)")
    lines += [
        "",
        "An empty review list is not proof of complete phonetic coverage.",
    ]
    return "\n".join(lines) + "\n"
