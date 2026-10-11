"""Compare unchanged v0.6 gates with overlap-safe periodicity on real audio."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from .audio import read_wav
from .calibration_io import load_calibration_session
from .calibration_beep_alignment import detect_beep_sequence
from .calibration_early_repeat_alignment import _suppress_beeps
from .calibration_target_alignment import _frame_features
from .calibration_periodicity import periodicity_track
from .calibration_sibilant_onset import (
    SIBILANTS, SEARCH_START_MS, SEARCH_END_MS, FRAME_MS, HOP_MS,
    select_sibilant_onset,
)
from .calibration_manual_labels import load_manual_labels

OUTPUT_NAME = "calibration_periodicity_repair_v1.0.json"


def assess_frame_gates(f: dict[str, np.ndarray], cue_ms: float) -> dict[str, object]:
    """Use original thresholds but repair periodicity; report diagnostic ablations."""
    p = {name: np.asarray(values) for name, values in f.items()}
    high = p["high_db"] - np.percentile(p["high_db"], 20)
    rms = p["rms_db"] - np.percentile(p["rms_db"], 20)
    gates = {
        "high": high >= 9, "rms": rms >= 7,
        "flat": p["flatness"] >= .055,
        "zcr": p["zcr"] >= .085,
        "aperiodic": p["periodicity"] < .58,
    }
    strict = np.logical_and.reduce(list(gates.values()))
    no_rms = np.logical_and.reduce([gate for name, gate in gates.items() if name != "rms"])
    guess = select_sibilant_onset(p, cue_ms)
    def run_length(mask: np.ndarray) -> int:
        best = current = 0
        for value in mask:
            current = current + 1 if bool(value) else 0
            best = max(current, best)
        return best
    return {
        "status": guess.status,
        "reason": guess.reason,
        "onset_ms_after_cue": guess.onset_ms_after_cue,
        "frames_by_gate": {name: int(np.count_nonzero(mask)) for name, mask in gates.items()},
        "strict_joint_frames": int(np.count_nonzero(strict)),
        "strict_longest_run": run_length(strict),
        "no_rms_joint_frames_diagnostic_only": int(np.count_nonzero(no_rms)),
        "no_rms_longest_run_diagnostic_only": run_length(no_rms),
    }


def evaluate_periodicity_repair(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    labels = load_manual_labels(session_dir).get("labels", {})
    rows = []
    for index, recording in sorted(session["recordings"].items()):
        prompt = session["protocol"]["prompts"][index]
        base = str(prompt.get("base_unit", ""))
        if base not in SIBILANTS:
            continue
        tokens = prompt.get("tokens") or str(prompt.get("spoken_pattern", "")).split()
        indices = [i for i, token in enumerate(tokens) if str(token) == str(prompt["syllable"])]
        if len(indices) != 2:
            continue
        samples, rate = read_wav(session_dir / "recordings" / str(recording["wav"]))
        beeps = detect_beep_sequence(samples, rate, len(tokens))
        cleaned = _suppress_beeps(samples, rate, beeps.events)
        for occurrence, token_index in enumerate(indices, 1):
            cue = float(beeps.events[token_index + 1].time_ms)
            old = _frame_features(cleaned, rate, cue + SEARCH_START_MS,
                                  cue + SEARCH_END_MS, frame_ms=FRAME_MS, hop_ms=HOP_MS)
            repaired = dict(old)
            repaired["periodicity"] = periodicity_track(cleaned, rate, old["centers"])
            estimate = assess_frame_gates(repaired, cue)
            raw = select_sibilant_onset(old, cue)
            lab = labels.get(str(index), {}).get("occurrences", {}).get(str(occurrence))
            gold = (
                float(lab["anchor_ms_after_cue"])
                if lab and lab.get("status") == "ok"
                and not lab.get("outside_expected_slot", False) else None
            )
            prediction = estimate["onset_ms_after_cue"]
            rows.append({
                "prompt_index": index, "occurrence": occurrence,
                "base_unit": base, "context": prompt.get("context_family"),
                "manual_ms": gold, "legacy_status": raw.status,
                "legacy_onset_ms": raw.onset_ms_after_cue,
                "repaired": estimate,
                "signed_error_ms": (
                    round(float(prediction) - gold, 3)
                    if gold is not None and prediction is not None else None
                ),
                "periodicity_p50_old": round(float(np.median(old["periodicity"])), 4),
                "periodicity_p50_repaired": round(float(np.median(repaired["periodicity"])), 4),
            })
    eligible = [r for r in rows if r["manual_ms"] is not None]
    errors = [abs(float(r["signed_error_ms"])) for r in eligible if r["signed_error_ms"] is not None]
    result = {
        "analysis": "periodicity_repair_ablation",
        "version": "1.0-development",
        "session_id": session_dir.name,
        "summary": {
            "rows": len(rows), "eligible_manual": len(eligible),
            "legacy_detected": sum(r["legacy_onset_ms"] is not None for r in rows),
            "repaired_detected": sum(r["repaired"]["onset_ms_after_cue"] is not None for r in rows),
            "eligible_matched": len(errors),
            "mae_ms_conditional": round(float(np.mean(errors)), 3) if errors else None,
            "p90_absolute_error_ms_conditional": round(float(np.percentile(errors,90)),3) if errors else None,
        },
        "limitations": "Development session, no held-out validation. No-RMS ablation is diagnostics only, not accepted onsets.",
        "rows": rows,
    }
    out = session_dir / "analysis" / OUTPUT_NAME
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
