"""Read v0.3 dual anchors without conflating consonant onset and CV boundary."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .audio import read_wav
from .calibration_cv_transition import cv_tracks, detect_cv_boundary
from .calibration_dual_labels import load_dual_labels
from .calibration_io import load_calibration_session
from .calibration_sibilant_onset import SIBILANTS
from .calibration_transition_onset import acoustic_tracks, detect_transition, nominal_target_cue_ms

OUTPUT_NAME = "calibration_dual_anchor_evaluation_v1.4.json"


def _stats(rows: list[dict[str, object]], target: str, *, source: str | None = None) -> dict[str, object]:
    selected = [
        row for row in rows
        if row["anchor_type"] == target
        and row["manual_status"] == "ok"
        and not row["outside_expected_slot"]
        and (source is None or row["manual_source"] == source)
    ]
    errors = np.asarray(
        [float(row["error_ms"]) for row in selected if row["error_ms"] is not None],
        dtype=np.float64,
    )
    return {
        "eligible": len(selected),
        "matched": int(len(errors)),
        "coverage": round(len(errors) / len(selected), 4) if selected else None,
        "mae_ms": round(float(np.mean(np.abs(errors))), 3) if len(errors) else None,
        "p90_absolute_error_ms": round(float(np.percentile(np.abs(errors), 90)), 3) if len(errors) else None,
        "median_signed_error_ms": round(float(np.median(errors)), 3) if len(errors) else None,
    }


def compare_dual_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    result = {
        "consonant_onset": _stats(rows, "consonant_onset"),
        "cv_boundary_all_provisional": _stats(rows, "cv_boundary"),
        "cv_boundary_manually_placed": _stats(rows, "cv_boundary", source="manual_v0.3"),
        "cv_boundary_legacy_import": _stats(rows, "cv_boundary", source="legacy_reviewed_copy"),
    }
    return result


def evaluate_dual_session(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    payload = load_dual_labels(session_dir)
    labels = payload["labels"]
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    durations: list[dict[str, object]] = []
    for index, rec in sorted(session["recordings"].items()):
        prompt = session["protocol"]["prompts"][index]
        base = str(prompt.get("base_unit", ""))
        if base not in SIBILANTS:
            continue
        tokens = prompt.get("tokens") or str(prompt.get("spoken_pattern", "")).split()
        positions = [i for i, word in enumerate(tokens) if word == prompt["syllable"]]
        if len(positions) != 2:
            failures.append({"prompt_index": index, "reason": "invalid_target_occurrence_count"})
            continue
        try:
            samples, rate = read_wav(session_dir / "recordings" / rec["wav"])
            transition = acoustic_tracks(samples, rate)
            cv_features = cv_tracks(samples, rate)
            for occurrence, token_index in enumerate(positions, 1):
                onset = detect_transition(transition, nominal_target_cue_ms(token_index))
                cv = detect_cv_boundary(cv_features, onset.onset_ms)
                annotation = labels.get(str(index), {}).get("occurrences", {}).get(str(occurrence), {})
                estimates = {
                    "consonant_onset": onset.onset_ms,
                    "cv_boundary": cv.absolute_ms,
                }
                for anchor_type, estimated in estimates.items():
                    mark = annotation.get(anchor_type)
                    if not isinstance(mark, dict):
                        continue
                    manual = float(mark["absolute_anchor_ms"])
                    is_valid = mark.get("status") == "ok" and not mark.get("outside_expected_slot", False)
                    delta = round(float(estimated - manual), 3) if estimated is not None and is_valid else None
                    rows.append({
                        "prompt_index": index, "base_unit": base,
                        "context": prompt["context_family"], "occurrence": occurrence,
                        "anchor_type": anchor_type,
                        "manual_ms": manual,
                        "manual_status": mark.get("status"),
                        "manual_source": mark.get("source", "unknown"),
                        "outside_expected_slot": bool(mark.get("outside_expected_slot", False)),
                        "predicted_ms": estimated,
                        "prediction_status": onset.status if anchor_type == "consonant_onset" else cv.status,
                        "prediction_reason": onset.reason if anchor_type == "consonant_onset" else cv.reason,
                        "error_ms": delta,
                    })
                if (isinstance(annotation.get("consonant_onset"), dict)
                        and isinstance(annotation.get("cv_boundary"), dict)):
                    start, end = annotation["consonant_onset"], annotation["cv_boundary"]
                    d = float(end["absolute_anchor_ms"]) - float(start["absolute_anchor_ms"])
                    durations.append({
                        "prompt_index": index, "base_unit": base, "occurrence": occurrence,
                        "duration_ms": round(d, 3),
                        "cv_status": end.get("status"),
                        "cv_source": end.get("source", "unknown"),
                        "quality_flag": "review_order" if d < 0 else ("review_cv_uncertain" if end.get("status") != "ok" else None),
                    })
        except (OSError, ValueError, KeyError, IndexError) as exc:
            failures.append({"prompt_index": index, "reason": str(exc)})
    output = {
        "analysis": "dual_anchor_evaluation",
        "version": "1.4-development",
        "session_id": session_dir.name,
        "summary": {
            "comparison": compare_dual_rows(rows),
            "prediction_statuses": dict(Counter(
                f"{row['anchor_type']}:{row['prediction_status']}" for row in rows
            )),
            "failures": len(failures),
        },
        "rows": rows,
        "manual_durations": durations,
        "failures": failures,
        "cautions": [
            "All measurements are on the same speaker's development session, not held-out validation.",
            "Consonant onset and CV boundary are separate labels and never pooled into one MAE.",
            "Legacy-imported CV marks are kept in a separate provenance stratum, NOT independently verified gold.",
            "CV detector is a fricative-specific prototype; paired stop, affricate, or nasal evaluation must use a class-specific model.",
            "Predictions may be missing or wrong; coverage and errors must be reported together.",
        ],
    }
    destination = session_dir / "analysis" / OUTPUT_NAME
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
