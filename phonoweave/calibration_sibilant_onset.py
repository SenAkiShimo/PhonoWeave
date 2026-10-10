"""Experimental class-specific sibilant onset detection (calibration v0.6).

This module deliberately supports Mandarin sh/s/x only. It does not change
the legacy generic detector or assert phonemic-boundary accuracy. Session
reports may be compared with frozen manual labels, but never change them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, read_wav
from .calibration_anchor_evaluation import _read_manual_labels
from .calibration_beep_alignment import detect_beep_sequence
from .calibration_early_repeat_alignment import (
    EarlyRepeatAlignmentError,
    _prompt_alignment as repeat_alignment,
    _suppress_beeps,
)
from .calibration_io import load_calibration_session
from .calibration_target_alignment import _frame_features, TargetAlignmentError


SIBILANTS = frozenset({"sh", "s", "x"})
SEARCH_START_MS = -140.0
SEARCH_END_MS = 520.0
FRAME_MS = 14.0
HOP_MS = 4.0
OUTPUT_NAME = "calibration_sibilant_onset_v0.6.json"


@dataclass(frozen=True)
class SibilantCandidate:
    onset_ms_after_cue: float | None
    status: str
    reason: str
    high_band_rise_db: float | None = None


def _first_run(mask: np.ndarray, minimum_frames: int) -> int | None:
    count = 0
    for index, item in enumerate(mask):
        count = count + 1 if bool(item) else 0
        if count >= minimum_frames:
            return index - minimum_frames + 1
    return None


def select_sibilant_onset(
    features: dict[str, np.ndarray],
    cue_ms: float,
    *,
    frame_ms: float = FRAME_MS,
) -> SibilantCandidate:
    """Find the earliest sustained aperiodic sibilant region.

    Absolute thresholds are provisional and *not* learned from or validated
    against the original development recordings. Flatness/periodicity gates
    aim to reject tonal beeps and later vowel nuclei. If the evidence is
    weak, return an unresolved result rather than a spurious max peak.
    """
    required = ("centers", "high_db", "rms_db", "flatness", "zcr", "periodicity")
    if any(key not in features for key in required):
        raise ValueError("missing required acoustic feature track")
    tracks = {key: np.asarray(features[key], dtype=np.float64) for key in required}
    lengths = {len(track) for track in tracks.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 3:
        raise ValueError("feature tracks have unequal or insufficient lengths")
    if not all(np.all(np.isfinite(v)) for v in tracks.values()):
        return SibilantCandidate(None, "unresolved", "non_finite_features")

    floor_high = float(np.percentile(tracks["high_db"], 20.0))
    floor_rms = float(np.percentile(tracks["rms_db"], 20.0))
    rise = tracks["high_db"] - floor_high

    # No hard no-speech zone around the cue: singers may anticipate it.
    # Removing the beep sinusoid is the responsibility of the caller.
    candidate = (
        (rise >= 9.0)
        & (tracks["rms_db"] - floor_rms >= 7.0)
        & (tracks["flatness"] >= 0.055)
        & (tracks["zcr"] >= 0.085)
        & (tracks["periodicity"] < 0.58)
    )
    index = _first_run(candidate, minimum_frames=3)
    if index is None:
        return SibilantCandidate(None, "unresolved", "no_sustained_aperiodic_frication")
    onset = float(tracks["centers"][index] - frame_ms / 2.0 - cue_ms)
    lower = float(tracks["centers"][0] - frame_ms / 2.0 - cue_ms)
    upper = float(tracks["centers"][-1] + frame_ms / 2.0 - cue_ms)
    if onset <= lower + 12.0 or onset >= upper - 25.0:
        return SibilantCandidate(onset, "review", "candidate_near_search_edge", float(rise[index]))
    return SibilantCandidate(onset, "candidate", "sustained_aperiodic_frication", float(rise[index]))


def _detect_one(
    cleaned: np.ndarray,
    sample_rate: int,
    cue_ms: float,
) -> SibilantCandidate:
    try:
        features = _frame_features(
            cleaned,
            sample_rate,
            cue_ms + SEARCH_START_MS,
            cue_ms + SEARCH_END_MS,
            frame_ms=FRAME_MS,
            hop_ms=HOP_MS,
        )
    except (TargetAlignmentError, ValueError) as exc:
        return SibilantCandidate(None, "unresolved", str(exc))
    return select_sibilant_onset(features, cue_ms)


def analyze_sibilant_session(session_dir: Path) -> dict[str, object]:
    """Produce a new diagnostic report; preserve recordings and old reports."""
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    protocol = session["protocol"]
    recordings = session["recordings"]
    prompts = protocol["prompts"]
    if not isinstance(prompts, list) or not isinstance(recordings, dict):
        raise ValueError("malformed calibration session")

    manual_path = session_dir / "analysis" / "calibration_manual_anchor_labels_v0.2.tsv"
    labels = _read_manual_labels(session_dir) if manual_path.is_file() else {}
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for index, recording in sorted(recordings.items()):
        if index < 0 or index >= len(prompts):
            continue
        prompt = prompts[index]
        if not isinstance(prompt, dict) or not isinstance(recording, dict):
            continue
        base = str(prompt.get("base_unit", ""))
        if base not in SIBILANTS:
            continue
        syllable = str(prompt.get("syllable", ""))
        tokens = prompt.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            tokens = str(prompt.get("spoken_pattern", "")).split()
        indices = [i for i, token in enumerate(tokens) if token == syllable]
        if len(indices) != 2:
            failures.append({"prompt_index": index, "reason": "expected_two_target_occurrences"})
            continue
        wav_path = session_dir / "recordings" / str(recording.get("wav", ""))
        try:
            samples, rate = read_wav(wav_path)
            sequence = detect_beep_sequence(samples, rate, len(tokens))
            cleaned = _suppress_beeps(samples, rate, sequence.events)
            cues = [sequence.events[i + 1].time_ms for i in indices]
        except (AudioReadError, ValueError, IndexError, OSError) as exc:
            failures.append({"prompt_index": index, "reason": str(exc)})
            continue

        pair: list[dict[str, object]] = []
        for occurrence, cue in enumerate(cues, start=1):
            guess = _detect_one(cleaned, rate, cue)
            label = labels.get((index, occurrence))
            manual = (
                float(label["anchor_ms_after_cue"])
                if label is not None and label.get("status") == "ok"
                else None
            )
            error = (
                float(guess.onset_ms_after_cue - manual)
                if manual is not None and guess.onset_ms_after_cue is not None
                else None
            )
            item: dict[str, object] = {
                "prompt_index": index,
                "occurrence": occurrence,
                "base_unit": base,
                "syllable": syllable,
                "cue_ms": round(float(cue), 3),
                "onset_ms_after_cue": (
                    None if guess.onset_ms_after_cue is None
                    else round(guess.onset_ms_after_cue, 3)
                ),
                "status": guess.status,
                "reason": guess.reason,
                "high_band_rise_db": guess.high_band_rise_db,
                "manual_anchor_ms_after_cue": manual,
                "signed_error_ms": None if error is None else round(error, 3),
            }
            pair.append(item)
            rows.append(item)

        # Repeat alignment is supporting evidence only, not a substitute
        # for detecting actual absolute acoustic events.
        try:
            alignment = repeat_alignment(session_dir, index, prompt, recording)
            lag = alignment.consensus_lag_ms
            alignment_qc = alignment.qc
        except (EarlyRepeatAlignmentError, AudioReadError, ValueError) as exc:
            lag, alignment_qc = None, "unavailable"
            failures.append({"prompt_index": index, "reason": f"repeat_alignment: {exc}"})
        if all(row["onset_ms_after_cue"] is not None for row in pair):
            observed = float(pair[1]["onset_ms_after_cue"]) - float(pair[0]["onset_ms_after_cue"])
            for row in pair:
                row["repeat_lag_ms"] = lag
                row["repeat_alignment_qc"] = alignment_qc
                row["repeat_onset_lag_disagreement_ms"] = (
                    None if lag is None else round(abs(observed - lag), 3)
                )
                if (
                    lag is not None
                    and alignment_qc == "usable"
                    and abs(observed - lag) > 65.0
                    and row["status"] == "candidate"
                ):
                    row["status"] = "review"
                    row["reason"] = "repeat_lag_disagreement"

    errors = [
        abs(float(row["signed_error_ms"]))
        for row in rows
        if row["signed_error_ms"] is not None
    ]
    payload: dict[str, object] = {
        "analysis": "calibration_sibilant_onset",
        "version": "0.6-experimental",
        "session_id": session["session_id"],
        "supported_bases": sorted(SIBILANTS),
        "development_note": (
            "Diagnostic thresholds are provisional. Report evaluates the existing "
            "development session, not held-out generalization. Candidate timestamps "
            "are acoustic estimates, not verified phoneme boundaries."
        ),
        "summary": {
            "analyzed_occurrences": len(rows),
            "candidate": sum(row["status"] == "candidate" for row in rows),
            "review": sum(row["status"] == "review" for row in rows),
            "unresolved": sum(row["status"] == "unresolved" for row in rows),
            "manual_pairs_evaluated": len(errors),
            "mae_ms": None if not errors else round(float(np.mean(errors)), 3),
            "p90_absolute_error_ms": (
                None if not errors else round(float(np.percentile(errors, 90.0)), 3)
            ),
        },
        "rows": rows,
        "failures": failures,
    }
    output = session_dir / "analysis" / OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
