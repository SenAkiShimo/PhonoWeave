"""Non-destructive v1.2 session evaluation of frication transitions.

Important: v0.2 manual labels in this session may mark the subsequent
vowel onset. Do not compute or advertise an onset MAE against suspect labels.
"""
from __future__ import annotations

import json
from pathlib import Path

from .audio import read_wav
from .calibration_io import load_calibration_session
from .calibration_manual_labels import load_manual_labels
from .calibration_sibilant_onset import SIBILANTS
from .calibration_transition_onset import (
    acoustic_tracks, detect_transition, nominal_target_cue_ms
)

OUTPUT_NAME = "calibration_transition_onset_v1.2.json"
MANUAL_REVIEW_DELTA_MS = 80.0


def analyze_transition_session(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    labels = load_manual_labels(session_dir)["labels"]
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for index, record in sorted(session["recordings"].items()):
        prompt = session["protocol"]["prompts"][index]
        base = str(prompt.get("base_unit", ""))
        if base not in SIBILANTS:
            continue
        tokens = prompt.get("tokens") or str(prompt.get("spoken_pattern", "")).split()
        matched = [i for i, token in enumerate(tokens) if token == prompt.get("syllable")]
        if len(matched) != 2:
            failures.append({"prompt_index": index, "reason": "expected_two_targets"})
            continue
        try:
            wav = session_dir / "recordings" / str(record["wav"])
            samples, rate = read_wav(wav)
            features = acoustic_tracks(samples, rate)
            for occurrence, token_index in enumerate(matched, 1):
                nominal_cue = nominal_target_cue_ms(token_index)
                guess = detect_transition(features, nominal_cue)
                label = labels.get(str(index), {}).get("occurrences", {}).get(str(occurrence))
                manual_valid = bool(
                    isinstance(label, dict) and label.get("status") == "ok"
                    and not label.get("outside_expected_slot", False)
                )
                manual_abs = (
                    float(label.get("absolute_anchor_ms",
                        label["cue_ms"] + label["anchor_ms_after_cue"]))
                    if manual_valid else None
                )
                difference = (
                    None if manual_abs is None or guess.onset_ms is None
                    else round(manual_abs - guess.onset_ms, 3)
                )
                conflict = difference is not None and abs(difference) > MANUAL_REVIEW_DELTA_MS
                rows.append({
                    "prompt_index": index,
                    "base_unit": base,
                    "context": prompt.get("context_family"),
                    "occurrence": occurrence,
                    "nominal_cue_ms": round(nominal_cue, 3),
                    "nominal_schedule_unverified": True,
                    "detected_onset_absolute_ms": guess.onset_ms,
                    "detected_onset_after_nominal_cue_ms": (
                        None if guess.onset_ms is None else round(guess.onset_ms - nominal_cue, 3)
                    ),
                    "status": guess.status,
                    "reason": guess.reason,
                    "high_band_baseline_db": guess.high_band_baseline_db,
                    "frication_duration_ms": guess.frication_duration_ms,
                    "manual_anchor_absolute_ms": manual_abs,
                    "manual_minus_candidate_ms": difference,
                    "manual_requires_review": conflict,
                })
        except (OSError, ValueError, KeyError, IndexError) as exc:
            failures.append({"prompt_index": index, "reason": str(exc)})

    payload: dict[str, object] = {
        "analysis": "calibration_transition_onset",
        "version": "1.2-development",
        "session_id": session_dir.name,
        "summary": {
            "occurrences": len(rows),
            "candidates": sum(r["status"] == "candidate" for r in rows),
            "review": sum(r["status"] == "review" for r in rows),
            "unresolved": sum(r["status"] == "unresolved" for r in rows),
            "human_anchor_conflicts": sum(r["manual_requires_review"] for r in rows),
            "failures": len(failures),
        },
        "notes": [
            "Nominal token times are derived from browser protocol, not reliable mic-detected beep landmarks.",
            "Human JSON annotation may target vowel onset instead of frication onset: conflicts require visual/manual review.",
            "DO NOT interpret the number of proposed onsets as accuracy or train on these same recordings.",
            "Feature extraction is amplitude-dependent; robust external-speaker validation is required."
        ],
        "rows": rows,
        "failures": failures
    }
    output = session_dir / "analysis" / OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
