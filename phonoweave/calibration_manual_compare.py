"""Compare calibration predictions with existing manual JSON labels.

The browser annotator writes JSON, whereas the older evaluation read TSV.
This module joins predictions to the JSON labels without altering labels,
recordings, or the underlying acoustic detectors.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .calibration_manual_labels import load_manual_labels
from .calibration_paired_anchor import analyze_paired_anchors
from .calibration_sibilant_onset import SIBILANTS, analyze_sibilant_session

OUTPUT_NAME = "calibration_manual_comparison_v0.8.json"


def _valid_labels(payload: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, int]]:
    labels = payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("manual JSON is missing labels")
    valid: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for raw in labels.values():
        if not isinstance(raw, dict):
            continue
        occurrences = raw.get("occurrences", {})
        if not isinstance(occurrences, dict):
            continue
        for number, observation in occurrences.items():
            if not isinstance(observation, dict):
                continue
            try:
                prompt_index = int(raw["prompt_index"])
                occurrence = int(number)
                value = float(observation["anchor_ms_after_cue"])
            except (KeyError, TypeError, ValueError):
                counts["invalid"] += 1
                continue
            if occurrence not in (1, 2) or not np.isfinite(value):
                counts["invalid"] += 1
                continue
            status = str(observation.get("status", "uncertain"))
            out_of_slot = bool(observation.get("outside_expected_slot", False))
            counts["total"] += 1
            if status == "uncertain":
                counts["uncertain"] += 1
            elif status != "ok":
                counts["other"] += 1
            elif out_of_slot:
                counts["out_of_slot_review"] += 1
            else:
                counts["eligible"] += 1
            valid.append({
                "prompt_index": prompt_index,
                "occurrence": occurrence,
                "base_unit": str(raw.get("base_unit", "")),
                "class_name": str(raw.get("class_name", "other")),
                "context": str(raw.get("context_family", "")),
                "manual_ms": value,
                "status": status,
                "outside_expected_slot": out_of_slot,
            })
    return valid, dict(counts)


def _metrics(rows: list[dict[str, object]], *, total_eligible: int) -> dict[str, object]:
    matched = [r for r in rows if r["signed_error_ms"] is not None]
    errors = np.asarray([float(r["signed_error_ms"]) for r in matched], dtype=np.float64)
    missing = total_eligible - len(matched)
    return {
        "eligible": total_eligible,
        "matched": len(matched),
        "missing": missing,
        "coverage": round(len(matched) / total_eligible, 4) if total_eligible else None,
        "mae_ms": round(float(np.mean(np.abs(errors))), 3) if len(errors) else None,
        "p90_absolute_error_ms": round(float(np.percentile(np.abs(errors), 90)), 3) if len(errors) else None,
        "median_signed_error_ms": round(float(np.median(errors)), 3) if len(errors) else None,
    }


def compare_predictions(
    labels_payload: dict[str, object],
    paired_payload: dict[str, object],
    sibilant_payload: dict[str, object],
) -> dict[str, object]:
    """Pure join/metrics function; injectable synthetic inputs enable unit tests."""
    labels, counts = _valid_labels(labels_payload)
    eligible = [r for r in labels if r["status"] == "ok" and not r["outside_expected_slot"]]

    paired: dict[tuple[int, int], dict[str, object]] = {}
    for item in paired_payload.get("anchors", []):
        if not isinstance(item, dict):
            continue
        for occurrence in (1, 2):
            value = item.get(f"occurrence_{occurrence}_anchor_ms_after_cue")
            if value is not None:
                paired[(int(item["prompt_index"]), occurrence)] = {
                    "ms": float(value), "status": str(item.get("qc", "unknown")),
                    "reason": str(item.get("alignment_qc", "")),
                }

    sibilant: dict[tuple[int, int], dict[str, object]] = {}
    for item in sibilant_payload.get("rows", []):
        if not isinstance(item, dict):
            continue
        sibilant[(int(item["prompt_index"]), int(item["occurrence"]))] = {
            "ms": item.get("onset_ms_after_cue"),
            "status": str(item.get("status", "unresolved")),
            "reason": str(item.get("reason", "unknown")),
        }

    models: dict[str, dict[str, object]] = {}
    for name, estimates, supported, accepted in (
        ("paired_v0.3.3_all", paired, None, None),
        ("paired_v0.3.3_usable", paired, None, {"usable"}),
        ("sibilant_v0.6_all", sibilant, SIBILANTS, {"candidate", "review"}),
        ("sibilant_v0.6_candidate", sibilant, SIBILANTS, {"candidate"}),
    ):
        target = [r for r in eligible if supported is None or r["base_unit"] in supported]
        rendered: list[dict[str, object]] = []
        by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
        by_base: dict[str, list[dict[str, object]]] = defaultdict(list)
        total_by_class: Counter[str] = Counter()
        total_by_base: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        for gold in target:
            key = (int(gold["prompt_index"]), int(gold["occurrence"]))
            estimate = estimates.get(key)
            state = str(estimate["status"]) if estimate else "missing"
            status_counts[state] += 1
            permitted = bool(
                estimate is not None
                and estimate["ms"] is not None
                and (accepted is None or state in accepted)
            )
            value = float(estimate["ms"]) if permitted else None
            error = None if value is None else round(value - float(gold["manual_ms"]), 3)
            row = {
                "prompt_index": gold["prompt_index"],
                "occurrence": gold["occurrence"],
                "base_unit": gold["base_unit"],
                "class_name": gold["class_name"],
                "context": gold["context"],
                "manual_ms": gold["manual_ms"],
                "estimated_ms": value,
                "estimate_status": state,
                "estimate_reason": None if estimate is None else estimate["reason"],
                "signed_error_ms": error,
            }
            rendered.append(row)
            cls, base = str(gold["class_name"]), str(gold["base_unit"])
            total_by_class[cls] += 1
            total_by_base[base] += 1
            by_class[cls].append(row)
            by_base[base].append(row)

        ranked = sorted(
            (row for row in rendered if row["signed_error_ms"] is not None),
            key=lambda row: abs(float(row["signed_error_ms"])),
            reverse=True,
        )
        models[name] = {
            "summary": _metrics(rendered, total_eligible=len(target)),
            "status_counts": dict(status_counts),
            "by_class": {
                cls: _metrics(rows, total_eligible=total_by_class[cls])
                for cls, rows in sorted(by_class.items())
            },
            "by_base": {
                base: _metrics(rows, total_eligible=total_by_base[base])
                for base, rows in sorted(by_base.items())
            },
            "worst_cases": ranked[:10],
            "rows": rendered,
        }

    return {
        "analysis": "calibration_manual_comparison",
        "version": "0.8-development",
        "manual_annotations": counts,
        "limitations": (
            "Same-session developmental data, not held-out evidence. "
            "MAE is conditional on a prediction being returned; inspect coverage "
            "and unresolved counts together. Uncertain and out-of-slot labels "
            "are excluded from primary metrics but preserved in the label file. "
            "This compares acoustic anchor definitions, not phonemic recognition accuracy."
        ),
        "models": models,
        "sibilant_all_statuses": dict(Counter(
            str(r.get("status", "unknown"))
            for r in sibilant_payload.get("rows", []) if isinstance(r, dict)
        )),
        "sibilant_unresolved_reasons": dict(Counter(
            str(r.get("reason", "unknown"))
            for r in sibilant_payload.get("rows", [])
            if isinstance(r, dict) and r.get("status") == "unresolved"
        )),
        "failures": {
            "paired": paired_payload.get("failed", []),
            "sibilant": sibilant_payload.get("failures", []),
        },
    }


def evaluate_session(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    labels = load_manual_labels(session_dir)
    if not labels.get("labels"):
        raise ValueError("no manual labels found in this session")
    paired = analyze_paired_anchors(session_dir)
    sibilant = analyze_sibilant_session(session_dir)
    result = compare_predictions(labels, paired, sibilant)
    result["session_id"] = session_dir.name
    output_dir = session_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / OUTPUT_NAME).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
