"""Exploratory two-stage frication detector; strict sustained core plus weak shoulder.

This module does NOT replace v1.4. Backtracing from a confirmed frication
core is explicitly experimental and requires held-out speaker validation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .calibration_transition_onset import (
    FRAME_MS, LEFT_MS, RIGHT_MS, MIN_HIGH_RISE_DB,
    MIN_ABSOLUTE_HIGH_DBFS, MIN_HIGH_FRACTION_DB, MIN_SUSTAIN_MS,
    detect_transition,
)

@dataclass(frozen=True)
class ShoulderEstimate:
    core_onset_ms: float | None
    shoulder_onset_ms: float | None
    status: str
    reason: str
    backtrace_ms: float | None


def detect_frication_shoulder(
    tracks: dict[str, np.ndarray], nominal_cue_ms: float,
    *, backtrace_max_ms: float = 80.0,
    weak_high_rise_db: float = 10.0,
) -> ShoulderEstimate:
    """Trace a weaker high-band lead-in only after strict frication is proven.

    Uses the same local baseline as v1.4. Every traced frame must exceed
    both an absolute high-band floor and a relative local energy threshold.
    The high-band fraction threshold applies only to the sustained core.
    """
    original = detect_transition(tracks, nominal_cue_ms)
    if original.onset_ms is None:
        return ShoulderEstimate(None, None, original.status, original.reason, None)
    centers = np.asarray(tracks["centers_ms"], dtype=float)
    high = np.asarray(tracks["high_dbfs"], dtype=float)
    ratio = np.asarray(tracks["high_fraction_db"], dtype=float)
    if any(a.ndim != 1 or len(a) != len(centers) for a in (high, ratio)):
        return ShoulderEstimate(original.onset_ms, None, "review", "invalid_feature_tracks", None)
    indices = np.flatnonzero((centers >= nominal_cue_ms - LEFT_MS) & (centers <= nominal_cue_ms + RIGHT_MS))
    if len(indices) < 50:
        return ShoulderEstimate(original.onset_ms, None, "review", "insufficient_search_window", None)
    baseline = float(np.percentile(high[indices[:45]], 20))
    step = float(np.median(np.diff(centers)))
    if step <= 0:
        return ShoulderEstimate(original.onset_ms, None, "review", "invalid_hop", None)
    core_index = int(np.argmin(np.abs(centers - (original.onset_ms + FRAME_MS / 2))))
    i = core_index
    minimum = max(MIN_ABSOLUTE_HIGH_DBFS, baseline + weak_high_rise_db)
    while i > indices[0] and (centers[core_index] - centers[i - 1] <= backtrace_max_ms):
        if high[i - 1] < minimum:
            break
        i -= 1
    shoulder = round(float(centers[i] - FRAME_MS / 2), 3)
    shift = round(float(original.onset_ms - shoulder), 3)
    return ShoulderEstimate(original.onset_ms, shoulder, "candidate", "confirmed_core_with_weak_shoulder", shift)
