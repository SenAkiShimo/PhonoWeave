from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from .calibration_early_repeat_alignment import analyze_early_repeat_alignment
from .calibration_paired_anchor import analyze_paired_anchors


MANUAL_LABELS_NAME = "calibration_manual_anchor_labels_v0.2.tsv"
OUTPUT_NAME = "calibration_anchor_evaluation_v0.4.json"


def _read_manual_labels(session_dir: Path) -> dict[tuple[int, int], dict[str, object]]:
    path = session_dir / "analysis" / MANUAL_LABELS_NAME
    if not path.is_file():
        raise ValueError(f"manual labels not found: {path}")

    rows: dict[tuple[int, int], dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader((line for line in handle if not line.startswith("#")), delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            try:
                prompt_index = int(row[0])
                occurrence = int(row[1])
                anchor = float(row[2])
            except ValueError:
                continue
            status = str(row[3]).strip() or "ok"
            rows[(prompt_index, occurrence)] = {
                "prompt_index": prompt_index,
                "occurrence": occurrence,
                "anchor_ms_after_cue": anchor,
                "status": status,
            }
    if not rows:
        raise ValueError("manual label file contains no labels")
    return rows


def _metrics(errors: list[float]) -> dict[str, object]:
    if not errors:
        return {
            "n": 0,
            "mean_signed_error_ms": None,
            "median_signed_error_ms": None,
            "mae_ms": None,
            "median_absolute_error_ms": None,
            "p90_absolute_error_ms": None,
            "early_count": 0,
            "late_count": 0,
        }
    absolute = np.abs(np.asarray(errors, dtype=np.float64))
    values = np.asarray(errors, dtype=np.float64)
    return {
        "n": len(errors),
        "mean_signed_error_ms": round(float(np.mean(values)), 3),
        "median_signed_error_ms": round(float(np.median(values)), 3),
        "mae_ms": round(float(np.mean(absolute)), 3),
        "median_absolute_error_ms": round(float(np.median(absolute)), 3),
        "p90_absolute_error_ms": round(float(np.percentile(absolute, 90.0)), 3),
        "early_count": int(np.sum(values < 0.0)),
        "late_count": int(np.sum(values > 0.0)),
    }


def _group_metrics(rows: list[dict[str, object]], field: str) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(float(row["error_ms"]))
    return {name: _metrics(errors) for name, errors in sorted(grouped.items())}


def _diagnosis(error_ms: float, auto_ms: float) -> str:
    if auto_ms <= 125.0:
        return "early_or_cue_proximal"
    if auto_ms >= 390.0:
        return "late_or_vowel_proximal"
    if error_ms <= -80.0:
        return "substantially_early"
    if error_ms >= 80.0:
        return "substantially_late"
    if abs(error_ms) >= 40.0:
        return "moderate_timing_error"
    return "near_manual"


def evaluate_manual_anchors(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    manual = _read_manual_labels(session_dir)

    paired = analyze_paired_anchors(session_dir)
    early = analyze_early_repeat_alignment(session_dir)

    paired_by_prompt = {
        int(row["prompt_index"]): row
        for row in paired.get("anchors", [])
        if isinstance(row, dict)
    }
    early_by_prompt = {
        int(row["prompt_index"]): row
        for row in early.get("alignments", [])
        if isinstance(row, dict)
    }

    absolute_rows: list[dict[str, object]] = []
    for (prompt_index, occurrence), manual_row in sorted(manual.items()):
        auto = paired_by_prompt.get(prompt_index)
        if auto is None:
            continue
        field = f"occurrence_{occurrence}_anchor_ms_after_cue"
        if field not in auto:
            continue
        manual_ms = float(manual_row["anchor_ms_after_cue"])
        auto_ms = float(auto[field])
        error = auto_ms - manual_ms
        absolute_rows.append(
            {
                "prompt_index": prompt_index,
                "occurrence": occurrence,
                "base_unit": str(auto.get("base_unit", "")),
                "class_name": str(auto.get("class_name", "")),
                "context_family": str(auto.get("context_family", "")),
                "syllable": str(auto.get("syllable", "")),
                "manual_status": str(manual_row.get("status", "ok")),
                "manual_ms_after_cue": round(manual_ms, 3),
                "auto_ms_after_cue": round(auto_ms, 3),
                "error_ms": round(error, 3),
                "absolute_error_ms": round(abs(error), 3),
                "auto_qc": str(auto.get("qc", "")),
                "diagnosis": _diagnosis(error, auto_ms),
            }
        )

    primary_absolute = [row for row in absolute_rows if row["manual_status"] == "ok"]
    uncertain_absolute = [row for row in absolute_rows if row["manual_status"] != "ok"]

    pair_rows: list[dict[str, object]] = []
    prompt_ids = sorted({prompt for prompt, _ in manual})
    for prompt_index in prompt_ids:
        first = manual.get((prompt_index, 1))
        second = manual.get((prompt_index, 2))
        auto = early_by_prompt.get(prompt_index)
        if first is None or second is None or auto is None:
            continue
        manual_lag = float(second["anchor_ms_after_cue"]) - float(first["anchor_ms_after_cue"])
        auto_lag = float(auto["consensus_lag_ms"])
        error = auto_lag - manual_lag
        pair_rows.append(
            {
                "prompt_index": prompt_index,
                "base_unit": str(auto.get("base_unit", "")),
                "class_name": str(auto.get("class_name", "")),
                "context_family": str(auto.get("context_family", "")),
                "syllable": str(auto.get("syllable", "")),
                "manual_status": "ok"
                if first.get("status") == "ok" and second.get("status") == "ok"
                else "uncertain",
                "manual_repeat_lag_ms": round(manual_lag, 3),
                "auto_consensus_lag_ms": round(auto_lag, 3),
                "error_ms": round(error, 3),
                "absolute_error_ms": round(abs(error), 3),
                "auto_qc": str(auto.get("qc", "")),
            }
        )

    primary_pairs = [row for row in pair_rows if row["manual_status"] == "ok"]

    absolute_errors = [float(row["error_ms"]) for row in primary_absolute]
    relative_errors = [float(row["error_ms"]) for row in primary_pairs]
    diagnosis_counts: dict[str, int] = defaultdict(int)
    for row in primary_absolute:
        diagnosis_counts[str(row["diagnosis"])] += 1

    worst = sorted(primary_absolute, key=lambda row: float(row["absolute_error_ms"]), reverse=True)[:10]

    payload: dict[str, object] = {
        "analysis": "calibration_anchor_evaluation",
        "version": "0.4",
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "development_note": (
            "This session was used during detector development. These metrics diagnose the current "
            "method on development data and are not independent confirmatory validation."
        ),
        "manual_labels": {
            "total": len(manual),
            "ok": sum(1 for row in manual.values() if row.get("status") == "ok"),
            "uncertain": sum(1 for row in manual.values() if row.get("status") != "ok"),
        },
        "absolute_anchor": {
            "definition": "paired-anchor occurrence estimate minus manual occurrence anchor",
            "primary_metrics": _metrics(absolute_errors),
            "by_class": _group_metrics(primary_absolute, "class_name"),
            "by_context": _group_metrics(primary_absolute, "context_family"),
            "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
            "worst_cases": worst,
            "rows": absolute_rows,
            "uncertain_rows": uncertain_absolute,
        },
        "relative_repeat_alignment": {
            "definition": (
                "v0.3.2 consensus lag minus manual (occurrence 2 anchor - occurrence 1 anchor)"
            ),
            "primary_metrics": _metrics(relative_errors),
            "by_class": _group_metrics(primary_pairs, "class_name"),
            "by_context": _group_metrics(primary_pairs, "context_family"),
            "rows": pair_rows,
        },
    }

    output_dir = session_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / OUTPUT_NAME
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
