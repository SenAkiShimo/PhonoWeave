from __future__ import annotations

import json
from pathlib import Path

from phonoweave.calibration_anchor_failure_analysis import analyze_anchor_failures


def _write_eval(session: Path, absolute_mae: float, relative_mae: float) -> None:
    analysis = session / "analysis"
    analysis.mkdir(parents=True)
    payload = {
        "absolute_anchor": {
            "primary_metrics": {
                "mae_ms": absolute_mae,
                "p90_absolute_error_ms": absolute_mae * 2,
            },
            "by_class": {
                "fricative": {
                    "n": 2,
                    "mae_ms": 80.0,
                    "median_absolute_error_ms": 80.0,
                    "p90_absolute_error_ms": 100.0,
                    "median_signed_error_ms": 70.0,
                },
                "stop": {
                    "n": 2,
                    "mae_ms": 20.0,
                    "median_absolute_error_ms": 20.0,
                    "p90_absolute_error_ms": 30.0,
                    "median_signed_error_ms": -5.0,
                },
            },
            "diagnosis_counts": {
                "late_or_vowel_proximal": 2,
                "near_manual": 2,
            },
            "rows": [
                {
                    "manual_status": "ok",
                    "class_name": "fricative",
                    "diagnosis": "late_or_vowel_proximal",
                },
                {
                    "manual_status": "ok",
                    "class_name": "fricative",
                    "diagnosis": "late_or_vowel_proximal",
                },
                {
                    "manual_status": "ok",
                    "class_name": "stop",
                    "diagnosis": "near_manual",
                },
            ],
            "worst_cases": [{"syllable": "xi", "absolute_error_ms": 100.0}],
        },
        "relative_repeat_alignment": {
            "primary_metrics": {
                "mae_ms": relative_mae,
                "p90_absolute_error_ms": relative_mae * 2,
            }
        },
    }
    (analysis / "calibration_anchor_evaluation_v0.4.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_retain_relative_alignment_when_materially_better(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _write_eval(session, absolute_mae=60.0, relative_mae=20.0)
    payload = analyze_anchor_failures(session)
    assert payload["headline"]["absolute_anchor"]["decision"] == "replace"
    assert payload["headline"]["relative_repeat_alignment"]["decision"] == "retain"
    assert payload["class_ranking_by_absolute_mae"][0]["class_name"] == "fricative"
    fricative = next(
        row for row in payload["class_recommendations"] if row["class_name"] == "fricative"
    )
    assert fricative["dominant_failure"] == "late_or_vowel_proximal"
    assert "stable sustained-frication onset detector" in fricative["recommended_detector"]
    assert (session / "analysis" / "calibration_anchor_failure_analysis_v0.5.json").is_file()


def test_relative_alignment_remains_unresolved_without_advantage(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _write_eval(session, absolute_mae=35.0, relative_mae=32.0)
    payload = analyze_anchor_failures(session)
    assert payload["headline"]["relative_repeat_alignment"]["decision"] == "unresolved"
    assert "v0.3.2-style relative repeat alignment" not in payload["architecture"]["keep"]
