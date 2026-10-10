"""Gate-attribution diagnostics for noisy fricative onset detection.

Ablated gates are counterfactual measurements, not onset predictions.
Production thresholds and original recordings remain unchanged.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np

from .audio import read_wav
from .calibration_cv_consistency import classify_syllable
from .calibration_cv_transition import cv_tracks
from .calibration_dual_labels import load_dual_labels
from .calibration_io import load_calibration_session
from .calibration_robustness import SCENARIOS, perturb
from .calibration_sibilant_onset import SIBILANTS
from .calibration_transition_onset import (
    LEFT_MS, MIN_ABSOLUTE_HIGH_DBFS, MIN_HIGH_FRACTION_DB,
    MIN_HIGH_RISE_DB, MIN_SUSTAIN_MS, RIGHT_MS, acoustic_tracks,
    nominal_target_cue_ms,
)

REPORT_FILENAME = "calibration_noise_gate_attribution_v2.3.json"


def longest_run(mask: np.ndarray) -> int:
    """Longest contiguous true run, in frames."""
    count = longest = 0
    for value in np.asarray(mask, dtype=bool):
        count = count + 1 if value else 0
        longest = max(longest, count)
    return longest


def gate_attribution(
    tracks: dict[str, np.ndarray], cue_ms: float,
    *, manual_onset_ms: float | None = None,
) -> dict[str, object]:
    """Describe each gate without using reference labels for detection."""
    keys = ("centers_ms", "high_dbfs", "high_fraction_db")
    t, high, fraction = [np.asarray(tracks[k], dtype=np.float64) for k in keys]
    if (len(t) < 50 or any(
        v.ndim != 1 or len(v) != len(t) or not np.all(np.isfinite(v))
        for v in (t, high, fraction)
    )):
        raise ValueError("invalid acoustic tracks")
    indices = np.flatnonzero((t >= cue_ms - LEFT_MS) & (t <= cue_ms + RIGHT_MS))
    if len(indices) < 50:
        raise ValueError("insufficient frames inside nominal cue window")
    times, values, fractions = t[indices], high[indices], fraction[indices]
    spacing = float(np.median(np.diff(times)))
    if spacing <= 0:
        raise ValueError("invalid frame time axis")
    baseline = float(np.percentile(values[:min(45, len(values))], 20))
    required = max(3, int(np.ceil(MIN_SUSTAIN_MS / spacing)))
    gates = {
        "high_rise": values >= baseline + MIN_HIGH_RISE_DB,
        "high_fraction": fractions >= MIN_HIGH_FRACTION_DB,
        "absolute_floor": values >= MIN_ABSOLUTE_HIGH_DBFS,
    }
    joint = longest_run(np.logical_and.reduce(list(gates.values())))
    ablations = {}
    for name in gates:
        remaining = [mask for key, mask in gates.items() if key != name]
        run = longest_run(np.logical_and.reduce(remaining))
        ablations[name] = {
            "max_consecutive_frames_without_gate": int(run),
            "sustained_without_gate": bool(run >= required),
        }
    blockers = (
        [name for name, data in ablations.items() if data["sustained_without_gate"]]
        if joint < required else []
    )
    manual_context = None
    if manual_onset_ms is not None:
        nearby = np.abs(times - float(manual_onset_ms)) <= 60.0
        manual_context = {
            "frames": int(np.count_nonzero(nearby)),
            "manual_inside_nominal_window": bool(times[0] <= manual_onset_ms <= times[-1]),
            "high_rise_margin_median_db": (
                round(float(np.median(
                    values[nearby] - baseline - MIN_HIGH_RISE_DB
                )), 3) if np.any(nearby) else None
            ),
            "gate_counts_near_manual": {
                key: int(np.count_nonzero(mask & nearby))
                for key, mask in gates.items()
            },
        }
    return {
        "high_baseline_dbfs": round(baseline, 3),
        "high_rise_threshold_dbfs": round(baseline + MIN_HIGH_RISE_DB, 3),
        "required_consecutive_frames": required,
        "frame_hop_ms": round(spacing, 3),
        "gate_frame_counts": {
            key: int(np.count_nonzero(mask)) for key, mask in gates.items()
        },
        "max_joint_consecutive_frames": int(joint),
        "strict_sustained_gate_pass": bool(joint >= required),
        "ablation_diagnostic_only": ablations,
        "single_gate_ablation_restores_persistence": blockers,
        "manual_context_diagnostic_only": manual_context,
    }


def audit_noise_gates(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    labels = load_dual_labels(session_dir)["labels"]
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for index, record in sorted(session["recordings"].items()):
        prompt = session["protocol"]["prompts"][index]
        if prompt.get("base_unit") not in SIBILANTS:
            continue
        tokens = prompt.get("tokens") or str(prompt.get("spoken_pattern", "")).split()
        occurrences = [
            i for i, token in enumerate(tokens) if token == prompt.get("syllable")
        ]
        if len(occurrences) != 2:
            failures.append({
                "prompt_index": index, "reason": "expected_two_target_occurrences"
            })
            continue
        try:
            audio, rate = read_wav(session_dir / "recordings" / record["wav"])
            for scenario_idx, scenario in enumerate(SCENARIOS):
                signal = perturb(
                    np.asarray(audio, dtype=np.float64), scenario,
                    np.random.default_rng(8500 + index * 11 + scenario_idx),
                )
                onset_tracks = acoustic_tracks(signal, rate)
                vowel_tracks = cv_tracks(signal, rate)
                for occurrence, token_idx in enumerate(occurrences, 1):
                    mark = (
                        labels.get(str(index), {}).get("occurrences", {})
                        .get(str(occurrence), {}).get("consonant_onset")
                    )
                    eligible = (
                        isinstance(mark, dict) and mark.get("status") == "ok"
                        and not mark.get("outside_expected_slot", False)
                        and mark.get("source") == "manual_v0.3"
                    )
                    gold = float(mark["absolute_anchor_ms"]) if eligible else None
                    cue = nominal_target_cue_ms(token_idx)
                    audit = gate_attribution(
                        onset_tracks, cue, manual_onset_ms=gold
                    )
                    prediction = classify_syllable(
                        onset_tracks, vowel_tracks, cue
                    )
                    rows.append({
                        "prompt_index": index,
                        "base_unit": prompt["base_unit"],
                        "context": prompt.get("context_family"),
                        "occurrence": occurrence,
                        "scenario": scenario,
                        "prediction_status": prediction.status,
                        "prediction_reason": prediction.reason,
                        "candidate_onset_ms": prediction.consonant_onset_ms,
                        "candidate_cv_ms": prediction.cv_boundary_ms,
                        "manual_onset_ms": gold,
                        "audit": audit,
                    })
        except (ValueError, OSError, KeyError, TypeError) as exc:
            failures.append({"prompt_index": index, "reason": str(exc)})

    summary = {}
    for scenario in SCENARIOS:
        subset = [r for r in rows if r["scenario"] == scenario]
        eligible = [r for r in subset if r["manual_onset_ms"] is not None]
        missing = [
            r for r in subset if r["prediction_status"] != "paired_candidate"
        ]
        summary[scenario] = {
            "target_occurrences": len(subset),
            "paired_candidates": len(subset) - len(missing),
            "eligible_manual": len(eligible),
            "eligible_paired": sum(
                r["prediction_status"] == "paired_candidate" for r in eligible
            ),
            "strict_gate_failures": sum(
                not r["audit"]["strict_sustained_gate_pass"] for r in subset
            ),
            "cv_or_sequence_review_after_core": sum(
                r["audit"]["strict_sustained_gate_pass"]
                and r["prediction_status"] != "paired_candidate"
                for r in subset
            ),
            "single_gate_ablation_recovery_diagnostic_only": dict(Counter(
                name for r in subset
                for name in r["audit"]["single_gate_ablation_restores_persistence"]
            )),
            "prediction_failure_reasons": dict(Counter(
                r["prediction_reason"] for r in missing
            )),
        }
    report = {
        "analysis": "noise_gate_attribution",
        "version": "2.3-development",
        "session_id": session_dir.name,
        "summary": summary,
        "rows": rows,
        "failures": failures,
        "cautions": [
            "Gate ablations are Boolean diagnostics, not validated onset predictions.",
            "Ablated gates never change the current detector's output.",
            "Manual-anchor neighborhoods are descriptive and do not select predictions.",
            "Noise perturbations reuse development audio and do not establish generalization.",
            "Missing paired candidates may reflect failed onset, CV evidence, or cue alignment.",
        ],
    }
    destination = session_dir / "analysis" / REPORT_FILENAME
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
