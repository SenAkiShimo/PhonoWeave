"""Non-deployable early-frication backtrace diagnostics.

Produces alternative timestamps only for already paired candidates. The
existing strict/fallback predictions are retained independently.
"""
from __future__ import annotations
import numpy as np
from .calibration_transition_onset import FRAME_MS, LEFT_MS, RIGHT_MS, MIN_ABSOLUTE_HIGH_DBFS
from .calibration_noise_floor_contrast import power_excess_db

def trace_early_frication(tracks: dict, onset_ms: float | None, cue_ms: float,
                         *, max_back_ms: float = 100.0,
                         rise_db: float = 3.0) -> dict:
    if onset_ms is None:
        return {"status": "unavailable", "onset_ms": None, "backtrace_ms": None}
    t=np.asarray(tracks["centers_ms"],dtype=float)
    high=np.asarray(tracks["high_dbfs"],dtype=float)
    fraction=np.asarray(tracks["high_fraction_db"],dtype=float)
    if (t.ndim!=1 or len(t)<50 or high.shape!=t.shape or fraction.shape!=t.shape
            or not all(np.all(np.isfinite(v)) for v in (t,high,fraction))):
        return {"status":"invalid_tracks","onset_ms":None,"backtrace_ms":None}
    window=np.flatnonzero((t>=cue_ms-LEFT_MS)&(t<=cue_ms+RIGHT_MS))
    if len(window)<50:
        return {"status":"insufficient_window","onset_ms":None,"backtrace_ms":None}
    baseline=float(np.percentile(high[window[:45]],20))
    excess=power_excess_db(high,baseline)
    core=int(np.argmin(np.abs(t-(onset_ms+FRAME_MS/2))))
    minimum=max(float(MIN_ABSOLUTE_HIGH_DBFS),baseline+rise_db)
    i=core
    while i>int(window[0]) and t[core]-t[i-1]<=max_back_ms:
        if high[i-1]<minimum or excess[i-1]<minimum:
            break
        i-=1
    # The tracked frame center and the input onset need not share the same
    # time-grid origin. Bound the proposed timestamp in physical milliseconds,
    # not by the difference between two rounded frame centers.
    earliest = float(onset_ms) - float(max_back_ms)
    proposed_frame_start = float(t[i] - FRAME_MS / 2)
    proposal = round(min(float(onset_ms), max(earliest, proposed_frame_start)), 3)
    shift = round(float(onset_ms) - proposal, 3)
    return {"status":"shadow_candidate","onset_ms":proposal,
            "backtrace_ms":shift,
            "baseline_dbfs":round(baseline,3)}
