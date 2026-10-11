"""Manual-onset-centered acoustic audit for unresolved Mandarin fricatives.

This diagnostic preserves old scores and recordings. The four-gate/no-RMS
candidate is deliberately NOT treated as a usable automatic prediction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio import read_wav
from .calibration_beep_alignment import detect_beep_sequence
from .calibration_early_repeat_alignment import _suppress_beeps
from .calibration_io import load_calibration_session
from .calibration_manual_labels import load_manual_labels
from .calibration_periodicity import periodicity_track
from .calibration_sibilant_onset import (
    FRAME_MS, HOP_MS, SEARCH_START_MS, SEARCH_END_MS, SIBILANTS,
)
from .calibration_target_alignment import _frame_features

OUTPUT_NAME = "calibration_manual_window_audit_v1.1.json"


def first_sustained(mask: np.ndarray, frames: int = 3) -> int | None:
    count = 0
    for index, passed in enumerate(mask):
        count = count + 1 if bool(passed) else 0
        if count >= frames:
            return index - frames + 1
    return None


def gate_audit(
    features: dict[str, np.ndarray],
    *,
    cue_ms: float,
    gold_ms: float | None,
) -> dict[str, object]:
    """Compare gates inside a local, human-marked onset neighborhood.

    Noise and silence indicators are evidence only. No decision threshold
    is fitted using manual labels. The existing five-gate v0.6 rule is
    left unchanged.
    """
    keys = ("centers", "high_db", "rms_db", "flatness", "zcr", "periodicity")
    tracks = {k: np.asarray(features[k], dtype=np.float64) for k in keys}
    n = len(tracks["centers"])
    if n < 3 or any(len(t) != n or not np.all(np.isfinite(t)) for t in tracks.values()):
        raise ValueError("invalid acoustic tracks")
    high_rise = tracks["high_db"] - np.percentile(tracks["high_db"], 20)
    rms_rise = tracks["rms_db"] - np.percentile(tracks["rms_db"], 20)
    gates = {
        "high_rise": high_rise >= 9.0,
        "rms_rise": rms_rise >= 7.0,
        "flatness": tracks["flatness"] >= 0.055,
        "zcr": tracks["zcr"] >= 0.085,
        "aperiodic": tracks["periodicity"] < 0.58,
    }
    full = np.logical_and.reduce(list(gates.values()))
    no_rms = np.logical_and.reduce([v for k, v in gates.items() if k != "rms_rise"])
    idx = first_sustained(no_rms, 3)
    candidate = None if idx is None else float(tracks["centers"][idx] - FRAME_MS / 2 - cue_ms)
    data: dict[str, object] = {
        "search_first_ms_after_cue": round(float(tracks["centers"][0] - FRAME_MS / 2 - cue_ms), 2),
        "search_last_ms_after_cue": round(float(tracks["centers"][-1] + FRAME_MS / 2 - cue_ms), 2),
        "full_joint_frames": int(np.count_nonzero(full)),
        "no_rms_joint_frames": int(np.count_nonzero(no_rms)),
        "no_rms_candidate_ms_diagnostic_only": None if candidate is None else round(candidate, 2),
        "rms_floor_db": round(float(np.percentile(tracks["rms_db"], 20)), 3),
        "high_floor_db": round(float(np.percentile(tracks["high_db"], 20)), 3),
    }
    if gold_ms is None:
        data["gold"] = {"available": False}
        return data

    absolute = cue_ms + gold_ms
    centers = tracks["centers"]
    in_window = bool(centers[0] - FRAME_MS / 2 <= absolute <= centers[-1] + FRAME_MS / 2)
    around = np.abs(centers - absolute) <= 60.0
    if not np.any(around):
        around = np.zeros(n, dtype=bool)

    closest = int(np.argmin(np.abs(centers - absolute)))
    nearby = np.flatnonzero(around)
    local_stats: dict[str, object] = {
        "available": True,
        "ms_after_cue": round(float(gold_ms), 2),
        "inside_automatic_search": in_window,
        "nearest_frame_center_ms_after_cue": round(float(centers[closest] - cue_ms), 2),
        "nearest_frame_offset_ms": round(float(centers[closest] - absolute), 2),
        "local_frame_count": int(len(nearby)),
        "local_gate_counts": {k: int(np.count_nonzero(g[around])) for k, g in gates.items()},
        "gates_at_nearest_frame": {k: bool(v[closest]) for k, v in gates.items()},
        "periodicity_near_gold_median": (
            round(float(np.median(tracks["periodicity"][around])), 4) if len(nearby) else None
        ),
        "rms_nearest_db": round(float(tracks["rms_db"][closest]), 3),
        "rms_rise_nearest_db": round(float(rms_rise[closest]), 3),
        "high_rise_nearest_db": round(float(high_rise[closest]), 3),
        "no_rms_candidate_error_ms_diagnostic_only": (
            None if candidate is None else round(float(candidate - gold_ms), 2)
        ),
    }
    data["gold"] = local_stats
    return data


def spectral_tone_audit(samples: np.ndarray, sample_rate: int, center_ms: float) -> dict[str, object]:
    """Measure short-time RMS and energy concentration near beep frequencies.

    The 700/1000 Hz fractions are checks for *possible* cue bleed. They
    are not reliable independent speech/no-speech classifiers.
    """
    size = max(256, round(sample_rate * 0.050))
    center = round(center_ms * sample_rate / 1000.0)
    start = max(0, center - size // 2)
    end = min(len(samples), start + size)
    x = np.asarray(samples[start:end], dtype=np.float64)
    if len(x) < 128:
        return {"available": False}
    x = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x * x)))
    window = np.hanning(len(x))
    spectrum = np.abs(np.fft.rfft(x * window)) ** 2
    frequencies = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    total_mask = (frequencies >= 180) & (frequencies < min(10000.0, sample_rate * .48))
    total = float(np.sum(spectrum[total_mask])) + 1e-20
    concentrations = {}
    for hz in (700, 1000):
        mask = (frequencies >= hz - 35) & (frequencies <= hz + 35)
        concentrations[str(hz)] = round(float(np.sum(spectrum[mask]) / total), 4)
    return {
        "available": True,
        "rms_dbfs": round(20 * float(np.log10(rms + 1e-12)), 2),
        "tone_energy_fraction": concentrations,
    }


def audit_session(session_dir: Path) -> dict[str, object]:
    """Write a report alongside previous versions without touching labels."""
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    labels = load_manual_labels(session_dir)["labels"]
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for index, rec in sorted(session["recordings"].items()):
        prompt = session["protocol"]["prompts"][index]
        base = str(prompt.get("base_unit", ""))
        if base not in SIBILANTS:
            continue
        tokens = prompt.get("tokens") or str(prompt.get("spoken_pattern", "")).split()
        matches = [i for i, token in enumerate(tokens) if token == prompt["syllable"]]
        if len(matches) != 2:
            errors.append({"prompt_index": index, "reason": "target_token_count_not_two"})
            continue
        try:
            samples, rate = read_wav(session_dir / "recordings" / str(rec["wav"]))
            seq = detect_beep_sequence(samples, rate, len(tokens))
            clean = _suppress_beeps(samples, rate, seq.events)
            for occurrence, token_index in enumerate(matches, 1):
                cue = float(seq.events[token_index + 1].time_ms)
                f = _frame_features(clean, rate, cue + SEARCH_START_MS,
                                    cue + SEARCH_END_MS, frame_ms=FRAME_MS, hop_ms=HOP_MS)
                f["periodicity"] = periodicity_track(clean, rate, f["centers"])
                label = labels.get(str(index), {}).get("occurrences", {}).get(str(occurrence))
                eligible = bool(
                    isinstance(label, dict) and label.get("status") == "ok"
                    and not label.get("outside_expected_slot", False)
                )
                gold = float(label["anchor_ms_after_cue"]) if eligible else None
                diag = gate_audit(f, cue_ms=cue, gold_ms=gold)
                extra = (
                    spectral_tone_audit(clean, rate, cue + gold)
                    if gold is not None else {"available": False}
                )
                rows.append({
                    "prompt_index": index, "base_unit": base,
                    "context": prompt["context_family"], "occurrence": occurrence,
                    "cue_ms": round(cue, 2), "sample_rate": rate,
                    "label_status": label.get("status") if isinstance(label, dict) else "unlabeled",
                    "human_anchor_eligible": eligible,
                    "audit": diag, "at_manual_anchor": extra,
                })
        except (OSError, ValueError, IndexError, KeyError) as exc:
            errors.append({"prompt_index": index, "reason": str(exc)})
    primary = [r for r in rows if r["human_anchor_eligible"]]
    with_candidates = [
        r for r in primary if r["audit"]["no_rms_candidate_ms_diagnostic_only"] is not None
    ]
    payload: dict[str, object] = {
        "analysis": "manual_window_signal_audit",
        "version": "1.1-development",
        "session_id": session_dir.name,
        "summary": {
            "occurrences": len(rows),
            "eligible_manual": len(primary),
            "manual_outside_search": sum(
                not r["audit"]["gold"]["inside_automatic_search"] for r in primary
            ),
            "no_rms_candidates_diagnostic_only": len(with_candidates),
            "failures": len(errors),
        },
        "rows": rows,
        "failures": errors,
        "warning": (
            "Development-set acoustic inspection only. No-RMS candidates are NOT "
            "automatic predictions. Do not weaken thresholds based on this same session."
        ),
    }
    dest = session_dir / "analysis" / OUTPUT_NAME
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
