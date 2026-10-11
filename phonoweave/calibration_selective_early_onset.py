"""Selective early-frication onset replay on an immutable v2.8 report.

The selection decision uses only unlabeled prediction timestamps and the
v2.8 sound-derived contiguous backtrace. Manual anchors are used strictly
after selection, for evaluation. This is a development-set replay; the 20 ms
rule is not a validated production operating point.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .calibration_early_frication_multiseed import REPORT_NAME as SOURCE_REPORT
from .calibration_transition_onset import FRAME_MS

OUTPUT_NAME = "calibration_selective_early_onset_v2.9.json"
MIN_PRECURSOR_MS = FRAME_MS
MAX_BACKTRACE_MS = 100.0
MIN_CV_GAP_MS = 45.0
MAX_CV_GAP_MS = 320.0


def select_early_onset(
    *, paired_status: str,
    baseline_onset_ms: float | None,
    shadow_onset_ms: float | None,
    cv_boundary_ms: float | None,
) -> dict:
    """Make a label-blind selection; never create new CV-paired candidates.

    A backtrace must contain at least one 20 ms analysis-frame equivalent
    of contiguous pre-core energy and preserve plausible onset-to-CV timing.
    No manual reference, noise condition or phoneme identity is an input.
    """
    if paired_status != "paired_candidate" or baseline_onset_ms is None:
        return {"onset_ms": None, "applied": False, "reason": "not_paired"}
    try:
        original = float(baseline_onset_ms)
        if not math.isfinite(original) or original < 0:
            raise ValueError("invalid baseline onset")
        if shadow_onset_ms is None or cv_boundary_ms is None:
            return {"onset_ms": original, "applied": False, "reason": "missing_evidence"}
        shadow = float(shadow_onset_ms)
        cv = float(cv_boundary_ms)
        if not all(math.isfinite(v) for v in (shadow, cv)) or shadow < 0:
            return {"onset_ms": original, "applied": False, "reason": "invalid_timestamp"}
    except (TypeError, ValueError):
        return {"onset_ms": None, "applied": False, "reason": "invalid_baseline"}
    shift = original - shadow
    if shift < -1e-6 or shift > MAX_BACKTRACE_MS + 1e-6:
        return {"onset_ms": original, "applied": False, "reason": "invalid_backtrace"}
    if shift < MIN_PRECURSOR_MS - 1e-6:
        return {"onset_ms": original, "applied": False, "reason": "short_precursor"}
    cv_gap = cv - shadow
    if not MIN_CV_GAP_MS < cv_gap <= MAX_CV_GAP_MS:
        return {"onset_ms": original, "applied": False, "reason": "cv_timing"}
    return {"onset_ms": shadow, "applied": True, "reason": "supported_precursor"}


def score_rows(rows: list[dict], error_key: str) -> dict:
    eligible = [r for r in rows if r.get("manual_onset_ms") is not None]
    errors = np.asarray(
        [r[error_key] for r in eligible if r.get(error_key) is not None],
        dtype=float,
    )
    absolute = np.abs(errors)
    return {
        "eligible": len(eligible),
        "matched": len(absolute),
        "coverage": round(len(absolute) / len(eligible), 4) if eligible else None,
        "conditional_mae_ms": round(float(np.mean(absolute)), 3) if len(absolute) else None,
        "conditional_p90_ms": round(float(np.percentile(absolute, 90)), 3) if len(absolute) else None,
        "median_signed_ms": round(float(np.median(errors)), 3) if len(errors) else None,
        "errors_over_40ms": int(np.sum(absolute > 40.0)),
    }


def pairwise_score(rows: list[dict]) -> dict:
    common = [
        r for r in rows
        if r.get("baseline_error_ms") is not None
        and r.get("selected_error_ms") is not None
    ]
    gain = np.asarray(
        [abs(r["baseline_error_ms"]) - abs(r["selected_error_ms"]) for r in common],
        dtype=float,
    )
    return {
        "common": len(common),
        "improved": int(np.sum(gain > 1e-3)),
        "worsened": int(np.sum(gain < -1e-3)),
        "unchanged": int(np.sum(np.abs(gain) <= 1e-3)),
        "mean_absolute_error_improvement_ms": (
            round(float(np.mean(gain)), 3) if len(gain) else None
        ),
    }


def evaluate_selective_replay(
    session_dir: Path, *, report_path: Path | None = None
) -> dict:
    session_dir = session_dir.expanduser().resolve()
    source = (report_path or session_dir / "analysis" / SOURCE_REPORT).expanduser().resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    if payload.get("version") != "2.8-development":
        raise ValueError("expected v2.8 early-frication report")
    if not isinstance(payload.get("positives"), list) or not isinstance(payload.get("trials"), list):
        raise ValueError("v2.8 report missing trial data")
    rows = []
    for original in payload["positives"]:
        if not isinstance(original, dict):
            raise ValueError("malformed positive row")
        decision = select_early_onset(
            paired_status=original["paired_status"],
            baseline_onset_ms=original["baseline_onset_ms"],
            shadow_onset_ms=original["shadow_onset_ms"],
            cv_boundary_ms=original["cv_boundary_ms"],
        )
        if decision["reason"] in {"invalid_backtrace", "invalid_baseline", "invalid_timestamp"}:
            raise ValueError("invalid source onset timestamp")
        gold = original["manual_onset_ms"]
        selected = decision["onset_ms"]
        row = {
            "trial": original["trial"],
            "noise_type": original["noise_type"],
            "snr_db": original["snr_db"],
            "seed": original["seed"],
            "prompt_index": original["prompt_index"],
            "base_unit": original["base_unit"],
            "context": original["context"],
            "occurrence": original["occurrence"],
            "paired_status": original["paired_status"],
            "manual_onset_ms": gold,
            "baseline_onset_ms": original["baseline_onset_ms"],
            "shadow_onset_ms": original["shadow_onset_ms"],
            "selected_onset_ms": selected,
            "decision_applied": decision["applied"],
            "decision_reason": decision["reason"],
            "backtrace_ms": original["backtrace_ms"],
            "baseline_error_ms": original["baseline_error_ms"],
            "shadow_error_ms": original["shadow_error_ms"],
            "selected_error_ms": (
                round(float(selected) - float(gold), 3)
                if selected is not None and gold is not None else None
            ),
        }
        rows.append(row)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["trial"]].append(row)
    trials = []
    for src_trial in payload["trials"]:
        name = src_trial["trial"]
        trial_rows = grouped[name]
        if len(trial_rows) != src_trial["positive_total"]:
            raise ValueError("v2.8 target counts do not agree with trial rows")
        trial = {
            "trial": name, "noise_type": src_trial["noise_type"],
            "snr_db": src_trial["snr_db"], "seed": src_trial["seed"],
            "total_targets": len(trial_rows),
            "baseline": score_rows(trial_rows, "baseline_error_ms"),
            "always_backtrace": score_rows(trial_rows, "shadow_error_ms"),
            "selective": score_rows(trial_rows, "selected_error_ms"),
            "selection_count": sum(r["decision_applied"] for r in trial_rows),
            "decision_reasons": {
                key: sum(r["decision_reason"] == key for r in trial_rows)
                for key in sorted({r["decision_reason"] for r in trial_rows})
            },
            "paired_comparison": pairwise_score(trial_rows),
            "inherited_negative_paired_recordings": src_trial["negative_paired_recordings"],
        }
        trials.append(trial)
    aggregates = {}
    for name in sorted({
        (t["noise_type"], t["snr_db"]) for t in trials if t["noise_type"] != "none"
    }):
        kind, snr = name
        cases = [t for t in trials if (t["noise_type"], t["snr_db"]) == name]
        item = {"trials": len(cases), "target_occurrences_per_trial": cases[0]["total_targets"]}
        for method in ("baseline", "always_backtrace", "selective"):
            values = [t[method]["conditional_mae_ms"] for t in cases
                      if t[method]["conditional_mae_ms"] is not None]
            item[method + "_median_mae_ms"] = round(float(np.median(values)), 3) if values else None
            item[method + "_errors_over_40ms_total"] = sum(
                t[method]["errors_over_40ms"] for t in cases
            )
        item["selective_better_trials_vs_baseline"] = sum(
            t["selective"]["conditional_mae_ms"] is not None
            and t["baseline"]["conditional_mae_ms"] is not None
            and t["selective"]["conditional_mae_ms"] < t["baseline"]["conditional_mae_ms"]
            for t in cases
        )
        item["selective_worsened_occurrences"] = sum(
            t["paired_comparison"]["worsened"] for t in cases
        )
        item["selective_applied_total"] = sum(t["selection_count"] for t in cases)
        aggregates[f"{kind}_{snr:g}db"] = item
    report = {
        "analysis": "selective_early_frication_replay",
        "version": "2.9-development",
        "session_id": session_dir.name,
        "source_report": source.name,
        "source_sha256": sha256(raw).hexdigest(),
        "rule": {
            "minimum_contiguous_precursor_ms": MIN_PRECURSOR_MS,
            "maximum_backtrace_ms": MAX_BACKTRACE_MS,
            "minimum_cv_gap_ms_exclusive": MIN_CV_GAP_MS,
            "maximum_cv_gap_ms_inclusive": MAX_CV_GAP_MS,
            "selection_uses_manual_labels": False,
            "selection_uses_snr_labels": False,
        },
        "clean": trials[0] if trials and trials[0]["trial"] == "original" else None,
        "aggregate": aggregates,
        "trials": trials,
        "positives": rows,
        "negative_eval": {
            "inherited_recording_condition_rows": len(payload.get("negatives", [])),
            "note": "CV acceptance is not changed by timestamp replay. This report does not establish whether earlier selected timestamps overlap non-speech or adjacent phonemes.",
        },
        "limitations": [
            "The 20 ms precursor rule was constructed during development after observing the earlier dataset; this replay is not held-out validation.",
            "The replay uses stored v2.8 baseline and shoulder estimates; no audio feature recomputation or novel acoustic classification is performed.",
            "All 24 noise perturbations derive from the same four WAV recordings; seeds are correlated observations, not independent utterances.",
            "Manual references are used only to score decisions, never as selection inputs.",
            "Inherited negative CV-pair outcomes cannot quantify backtrace boundary contamination.",
            "No default detector, recording, or annotation is changed."
        ],
    }
    destination = session_dir / "analysis" / OUTPUT_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
