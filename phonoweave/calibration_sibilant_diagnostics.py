"""Non-destructive acoustic gate diagnostics for calibration sibilants.

This is a diagnostic tool, not a new onset detector. Do not tune gates based
solely on counts without inspecting audio and manually labeled examples.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio import AudioReadError, read_wav
from .calibration_beep_alignment import detect_beep_sequence
from .calibration_early_repeat_alignment import _suppress_beeps
from .calibration_io import load_calibration_session
from .calibration_sibilant_onset import (
    FRAME_MS,
    HOP_MS,
    SEARCH_END_MS,
    SEARCH_START_MS,
    SIBILANTS,
    select_sibilant_onset,
)
from .calibration_target_alignment import TargetAlignmentError, _frame_features

OUTPUT_NAME = "calibration_sibilant_gate_diagnostics_v0.6.1.json"
GATE_NAMES = ("high_rise", "rms_rise", "flatness", "zcr", "aperiodic")


def _longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for item in mask:
        current = current + 1 if bool(item) else 0
        longest = max(longest, current)
    return longest


def summarize_gates(features: dict[str, np.ndarray], cue_ms: float) -> dict[str, object]:
    """Report each original v0.6 gate independently and jointly.

    All v0.6 thresholds are intentionally unchanged in this diagnostic.
    In particular, high/rms floors use the 20th percentile of the same
    search window, exactly as the existing onset detector does.
    """
    keys = ("centers", "high_db", "rms_db", "flatness", "zcr", "periodicity")
    if any(key not in features for key in keys):
        raise ValueError("missing feature track")
    x = {name: np.asarray(features[name], dtype=np.float64) for name in keys}
    n = len(x["centers"])
    if n < 3 or any(len(value) != n for value in x.values()):
        raise ValueError("inconsistent or insufficient feature tracks")
    if any(not np.all(np.isfinite(v)) for v in x.values()):
        return {"frame_count": n, "error": "non_finite_features"}

    high_floor = float(np.percentile(x["high_db"], 20))
    rms_floor = float(np.percentile(x["rms_db"], 20))
    high_rise = x["high_db"] - high_floor
    rms_rise = x["rms_db"] - rms_floor
    gates = {
        "high_rise": high_rise >= 9.0,
        "rms_rise": rms_rise >= 7.0,
        "flatness": x["flatness"] >= 0.055,
        "zcr": x["zcr"] >= 0.085,
        "aperiodic": x["periodicity"] < 0.58,
    }
    mask = np.logical_and.reduce(tuple(gates.values()))
    gate_score = np.sum(np.stack(tuple(gates.values()), axis=0), axis=0)
    closest = int(np.argmax(gate_score))
    # Nearby four-of-five frames are particularly informative for deciding
    # which condition blocks the conjunction; not accepted as onsets.
    best_missing = [name for name in GATE_NAMES if not gates[name][closest]]
    return {
        "frame_count": n,
        "window_first_ms_after_cue": round(float(x["centers"][0] - FRAME_MS / 2 - cue_ms), 2),
        "window_last_ms_after_cue": round(float(x["centers"][-1] + FRAME_MS / 2 - cue_ms), 2),
        "high_floor_db": round(high_floor, 3),
        "rms_floor_db": round(rms_floor, 3),
        "gate_pass_counts": {name: int(np.count_nonzero(gates[name])) for name in GATE_NAMES},
        "all_five_pass_frames": int(np.count_nonzero(mask)),
        "four_or_more_pass_frames": int(np.count_nonzero(gate_score >= 4)),
        "longest_all_five_run_frames": _longest_run(mask),
        "longest_four_or_more_run_frames": _longest_run(gate_score >= 4),
        "best_gate_count": int(gate_score[closest]),
        "best_frame_ms_after_cue": round(float(x["centers"][closest] - FRAME_MS / 2 - cue_ms), 2),
        "missing_gates_at_best_frame": best_missing,
        "best_frame_values": {
            "high_rise_db": round(float(high_rise[closest]), 3),
            "rms_rise_db": round(float(rms_rise[closest]), 3),
            "flatness": round(float(x["flatness"][closest]), 5),
            "zcr": round(float(x["zcr"][closest]), 5),
            "periodicity": round(float(x["periodicity"][closest]), 5),
        },
        "feature_p90": {
            "high_rise_db": round(float(np.percentile(high_rise, 90)), 3),
            "rms_rise_db": round(float(np.percentile(rms_rise, 90)), 3),
            "flatness": round(float(np.percentile(x["flatness"], 90)), 5),
            "zcr": round(float(np.percentile(x["zcr"], 90)), 5),
            "periodicity": round(float(np.percentile(x["periodicity"], 90)), 5),
        },
    }


def diagnose_session(session_dir: Path) -> dict[str, object]:
    """Diagnose the existing session without modifying any recordings."""
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    protocol = session["protocol"]
    recordings = session["recordings"]
    prompts = protocol["prompts"]
    if not isinstance(prompts, list) or not isinstance(recordings, dict):
        raise ValueError("malformed calibration session")
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for index, recording in sorted(recordings.items()):
        if not (0 <= index < len(prompts)):
            continue
        prompt = prompts[index]
        if not isinstance(prompt, dict) or not isinstance(recording, dict):
            continue
        base = str(prompt.get("base_unit", ""))
        if base not in SIBILANTS:
            continue
        tokens = prompt.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            tokens = str(prompt.get("spoken_pattern", "")).split()
        syllable = str(prompt.get("syllable", ""))
        target_indices = [i for i, token in enumerate(tokens) if str(token) == syllable]
        if len(target_indices) != 2:
            failures.append({"prompt_index": index, "reason": "expected_two_targets"})
            continue

        wav_path = session_dir / "recordings" / str(recording.get("wav", ""))
        try:
            samples, sample_rate = read_wav(wav_path)
            sequence = detect_beep_sequence(samples, sample_rate, len(tokens))
            cleaned = _suppress_beeps(samples, sample_rate, sequence.events)
        except (AudioReadError, ValueError, IndexError, OSError) as exc:
            failures.append({"prompt_index": index, "reason": str(exc)})
            continue

        for occurrence, token_index in enumerate(target_indices, start=1):
            cue = float(sequence.events[token_index + 1].time_ms)
            try:
                features = _frame_features(
                    cleaned, sample_rate, cue + SEARCH_START_MS, cue + SEARCH_END_MS,
                    frame_ms=FRAME_MS, hop_ms=HOP_MS,
                )
                diag = summarize_gates(features, cue)
                choice = select_sibilant_onset(features, cue)
            except (TargetAlignmentError, ValueError) as exc:
                failures.append({
                    "prompt_index": index,
                    "occurrence": occurrence,
                    "reason": str(exc),
                })
                continue
            rows.append({
                "prompt_index": index,
                "occurrence": occurrence,
                "base_unit": base,
                "context": str(prompt.get("context_family", "")),
                "syllable": syllable,
                "cue_ms": round(cue, 3),
                "detector_status": choice.status,
                "detector_reason": choice.reason,
                "detector_onset_ms_after_cue": choice.onset_ms_after_cue,
                "beep_tone_score": round(float(sequence.events[token_index + 1].tone_score), 5),
                "beep_frequency_margin": round(float(sequence.events[token_index + 1].frequency_margin), 5),
                "gate_diagnostics": diag,
            })

    report: dict[str, object] = {
        "analysis": "calibration_sibilant_gate_diagnostics",
        "version": "0.6.1",
        "session_id": session["session_id"],
        "note": "Diagnostics only. No phoneme inference or validated accuracy claim.",
        "thresholds": {
            "high_rise_min_db": 9.0,
            "rms_rise_min_db": 7.0,
            "flatness_min": 0.055,
            "zcr_min": 0.085,
            "periodicity_max_exclusive": 0.58,
            "minimum_consecutive_frames": 3,
        },
        "summary": {
            "target_occurrences": len(rows),
            "failures": len(failures),
            "candidate": sum(r["detector_status"] == "candidate" for r in rows),
            "review": sum(r["detector_status"] == "review" for r in rows),
            "unresolved": sum(r["detector_status"] == "unresolved" for r in rows),
        },
        "rows": rows,
        "failures": failures,
    }
    output = session_dir / "analysis" / OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
