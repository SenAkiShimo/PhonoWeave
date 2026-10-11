"""Experimental noise-floor-compensated frication fallback.

The strict v2.1 detector remains authoritative. A second candidate is
evaluated only when strict frication has no sustained core. An alternative
prediction requires an independently detected CV transition. This is a
development-only candidate, not a calibrated classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .calibration_cv_consistency import classify_syllable, SyllableCandidate
from .calibration_cv_transition import detect_cv_boundary
from .calibration_noise_floor_contrast import power_excess_db
from .calibration_transition_onset import (
    FRAME_MS, LEFT_MS, RIGHT_MS, MIN_SUSTAIN_MS,
    MIN_HIGH_FRACTION_DB, MIN_ABSOLUTE_HIGH_DBFS, _runs,
)

EXCESS_QUANTILE = 90.0
EXCESS_PEAK_MARGIN_DB = 14.0
MIN_NOISE_RISE_DB = 3.0
MAX_ONSET_TO_CV_MS = 320.0


@dataclass(frozen=True)
class FallbackResult:
    consonant_onset_ms: float | None
    cv_boundary_ms: float | None
    status: str
    reason: str
    method: str
    threshold_dbfs: float | None = None


def detect_excess_fallback(onset_tracks: dict, vowel_tracks: dict, cue_ms: float) -> FallbackResult:
    """Return strict output or CV-confirmed alternative when strict core fails.

    The excess-power threshold is adaptive to the 90th percentile within the
    nominal window. This is *not* a learned threshold and may include vowel
    energy. Gate settings must not be promoted using development-set results.
    """
    strict = classify_syllable(onset_tracks, vowel_tracks, cue_ms)
    if strict.status == "paired_candidate":
        return FallbackResult(
            strict.consonant_onset_ms, strict.cv_boundary_ms,
            strict.status, strict.reason, "strict_v2.1"
        )
    if strict.reason != "no_sustained_high_band_transition":
        return FallbackResult(
            strict.consonant_onset_ms, strict.cv_boundary_ms,
            strict.status, strict.reason, "strict_v2.1"
        )
    try:
        t = np.asarray(onset_tracks["centers_ms"], dtype=float)
        high = np.asarray(onset_tracks["high_dbfs"], dtype=float)
        fraction = np.asarray(onset_tracks["high_fraction_db"], dtype=float)
    except (TypeError, KeyError, ValueError):
        return FallbackResult(None, None, "unresolved", "invalid_feature_tracks", "excess_fallback")
    if (t.ndim != 1 or len(t) < 50 or
            any(a.ndim != 1 or len(a) != len(t) or not np.all(np.isfinite(a))
                for a in (high, fraction)) or not np.all(np.isfinite(t))):
        return FallbackResult(None, None, "unresolved", "invalid_feature_tracks", "excess_fallback")
    idx = np.flatnonzero((t >= cue_ms - LEFT_MS) & (t <= cue_ms + RIGHT_MS))
    if len(idx) < 50:
        return FallbackResult(None, None, "unresolved", "insufficient_search_window", "excess_fallback")
    values, ratios, times = high[idx], fraction[idx], t[idx]
    baseline = float(np.percentile(values[:45], 20))
    excess = power_excess_db(values, baseline)
    peak = float(np.percentile(excess, EXCESS_QUANTILE))
    threshold = peak - EXCESS_PEAK_MARGIN_DB
    mask = (
        (values >= baseline + MIN_NOISE_RISE_DB)
        & (values >= MIN_ABSOLUTE_HIGH_DBFS)
        & (ratios >= MIN_HIGH_FRACTION_DB)
        & (excess >= threshold)
    )
    spacing = float(np.median(np.diff(times)))
    if not np.isfinite(spacing) or spacing <= 0:
        return FallbackResult(None, None, "unresolved", "invalid_time_axis", "excess_fallback")
    needed = max(3, int(np.ceil(MIN_SUSTAIN_MS / spacing)))
    candidates = []
    for first, end in _runs(mask):
        if end - first < needed or first < 4:
            continue
        onset = round(float(times[first] - FRAME_MS / 2), 3)
        cv = detect_cv_boundary(vowel_tracks, onset)
        if cv.absolute_ms is None:
            continue
        duration = cv.absolute_ms - onset
        if 45.0 < duration <= MAX_ONSET_TO_CV_MS:
            candidates.append((abs(onset - cue_ms), onset, cv.absolute_ms))
    if not candidates:
        return FallbackResult(None, None, "review",
                              "no_cv_confirmed_excess_candidate", "excess_fallback",
                              round(threshold, 3))
    _, onset, cv_ms = min(candidates)
    return FallbackResult(
        onset, cv_ms, "paired_candidate",
        "noise_excess_frication_followed_by_vowel",
        "experimental_excess_fallback", round(threshold, 3)
    )
