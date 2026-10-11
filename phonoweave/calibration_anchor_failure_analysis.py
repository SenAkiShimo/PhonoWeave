from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


EVALUATION_NAME = "calibration_anchor_evaluation_v0.4.json"
OUTPUT_NAME = "calibration_anchor_failure_analysis_v0.5.json"


def _load_evaluation(session_dir: Path) -> dict[str, object]:
    path = session_dir / "analysis" / EVALUATION_NAME
    if not path.is_file():
        raise ValueError(f"anchor evaluation not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("anchor evaluation is invalid")
    return payload


def _metric(block: dict[str, object], name: str) -> float | None:
    value = block.get(name)
    return None if value is None else float(value)


def _class_rank(by_class: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_name, raw in by_class.items():
        if not isinstance(raw, dict):
            continue
        rows.append(
            {
                "class_name": class_name,
                "n": int(raw.get("n", 0)),
                "mae_ms": raw.get("mae_ms"),
                "median_absolute_error_ms": raw.get("median_absolute_error_ms"),
                "p90_absolute_error_ms": raw.get("p90_absolute_error_ms"),
                "median_signed_error_ms": raw.get("median_signed_error_ms"),
            }
        )
    return sorted(
        rows,
        key=lambda row: -1.0 if row["mae_ms"] is None else float(row["mae_ms"]),
        reverse=True,
    )


def _recommend_class(class_name: str, diagnoses: Counter[str], median_signed: float | None) -> dict[str, object]:
    dominant = diagnoses.most_common(1)[0][0] if diagnoses else "insufficient_data"
    if class_name == "stop":
        detector = "release transient / burst detector"
    elif class_name == "affricate":
        detector = "closure-to-sustained-frication transition detector"
    elif class_name == "fricative":
        detector = "stable sustained-frication onset detector"
    elif class_name == "sonorant":
        detector = "stable voicing / sonorant-energy onset detector"
    else:
        detector = "class-specific acoustic transition detector"

    actions = [
        f"replace pooled generic event selection with a {detector}",
        "retain adjacent-beep timing only as a search prior, not as a hard exclusion zone",
        "validate candidate consistency across the two repetitions before accepting the anchor",
    ]
    if dominant == "early_or_cue_proximal":
        actions.append("penalize cue-proximal candidates using explicit beep-residual features")
    elif dominant == "late_or_vowel_proximal":
        actions.append("penalize vowel-transition candidates using periodicity/formant-rise evidence")
    elif dominant == "substantially_early":
        actions.append("require stronger sustained evidence before accepting an onset")
    elif dominant == "substantially_late":
        actions.append("prefer the earliest stable class-appropriate event over later local maxima")

    if median_signed is not None and abs(median_signed) >= 25.0:
        direction = "late" if median_signed > 0 else "early"
        actions.append(f"treat the current class detector as systematically {direction}, not merely noisy")

    return {
        "class_name": class_name,
        "dominant_failure": dominant,
        "diagnosis_counts": dict(diagnoses),
        "recommended_detector": detector,
        "actions": actions,
    }


def analyze_anchor_failures(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    evaluation = _load_evaluation(session_dir)
    absolute = evaluation.get("absolute_anchor")
    relative = evaluation.get("relative_repeat_alignment")
    if not isinstance(absolute, dict) or not isinstance(relative, dict):
        raise ValueError("evaluation is missing anchor metrics")

    absolute_metrics = absolute.get("primary_metrics")
    relative_metrics = relative.get("primary_metrics")
    if not isinstance(absolute_metrics, dict) or not isinstance(relative_metrics, dict):
        raise ValueError("evaluation primary metrics are missing")

    abs_mae = _metric(absolute_metrics, "mae_ms")
    rel_mae = _metric(relative_metrics, "mae_ms")
    abs_p90 = _metric(absolute_metrics, "p90_absolute_error_ms")
    rel_p90 = _metric(relative_metrics, "p90_absolute_error_ms")

    relative_better = (
        abs_mae is not None
        and rel_mae is not None
        and rel_mae + 10.0 < abs_mae
    )
    relative_materially_better = (
        abs_mae is not None
        and rel_mae is not None
        and rel_mae <= 0.65 * abs_mae
    )

    rows = absolute.get("rows")
    if not isinstance(rows, list):
        rows = []
    primary_rows = [
        row for row in rows
        if isinstance(row, dict) and row.get("manual_status") == "ok"
    ]

    diagnoses_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for row in primary_rows:
        diagnoses_by_class[str(row.get("class_name", "other"))][str(row.get("diagnosis", "unknown"))] += 1

    by_class = absolute.get("by_class")
    if not isinstance(by_class, dict):
        by_class = {}
    class_ranking = _class_rank(by_class)

    recommendations = []
    for ranked in class_ranking:
        class_name = str(ranked["class_name"])
        median_signed = ranked.get("median_signed_error_ms")
        recommendations.append(
            _recommend_class(
                class_name,
                diagnoses_by_class[class_name],
                None if median_signed is None else float(median_signed),
            )
        )

    if relative_materially_better:
        alignment_decision = "retain"
        alignment_reason = (
            "Relative repeat alignment is materially more accurate than absolute anchor selection. "
            "Preserve v0.3.2-style repeat alignment as a reusable component."
        )
    elif relative_better:
        alignment_decision = "retain_with_review"
        alignment_reason = (
            "Relative repeat alignment is better than absolute anchor selection, but the margin is modest. "
            "Keep it provisionally and re-check after the absolute detector is redesigned."
        )
    else:
        alignment_decision = "unresolved"
        alignment_reason = (
            "Current development metrics do not establish a clear advantage for relative repeat alignment. "
            "Do not promote it to a frozen component yet."
        )

    absolute_decision = "replace"
    absolute_reason = (
        "The v0.3.3 pooled absolute-event selector should not be tuned further as one generic detector. "
        "Manual labels are now available, so redesign around class-specific acoustic events instead."
    )

    payload: dict[str, object] = {
        "analysis": "calibration_anchor_failure_analysis",
        "version": "0.5",
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "development_note": (
            "This report diagnoses development data used during method iteration. Decisions here are "
            "architecture decisions, not independent validation claims."
        ),
        "headline": {
            "absolute_anchor": {
                "decision": absolute_decision,
                "reason": absolute_reason,
                "mae_ms": abs_mae,
                "p90_absolute_error_ms": abs_p90,
            },
            "relative_repeat_alignment": {
                "decision": alignment_decision,
                "reason": alignment_reason,
                "mae_ms": rel_mae,
                "p90_absolute_error_ms": rel_p90,
            },
        },
        "architecture": {
            "keep": [
                "physical beep sequence detection and actual cue timing",
                "manual-label evaluation infrastructure",
            ] + (["v0.3.2-style relative repeat alignment"] if alignment_decision.startswith("retain") else []),
            "remove_or_replace": [
                "v0.3.3 generic pooled absolute-event selector",
                "hard assumptions that speech onset must occur after the cue",
                "single detector behavior shared across all onset manners",
            ],
            "next_pipeline": [
                "detect physical beep sequence",
                "build a broad cue-relative search region that allows anticipatory speech",
                "run a class-specific absolute event detector on one or both repetitions",
                "use repeat alignment as cross-repetition support when justified by metrics",
                "reject or flag candidates that disagree across repetitions",
                "evaluate against frozen manual labels without changing the labels",
            ],
        },
        "class_ranking_by_absolute_mae": class_ranking,
        "class_recommendations": recommendations,
        "diagnosis_counts": absolute.get("diagnosis_counts", {}),
        "worst_cases": absolute.get("worst_cases", []),
    }

    output = session_dir / "analysis" / OUTPUT_NAME
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
