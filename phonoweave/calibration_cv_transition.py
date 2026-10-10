"""Experimental fricative-to-vowel (CV) transition detector.

Finds the rise of low-frequency vowel energy after a previously detected
sustained fricative. A distinct acoustic event from consonant onset.
Diagnostic candidate only; no held-out validation has been performed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

FRAME_MS = 20.0
HOP_MS = 4.0
SMOOTH_FRAMES = 5
LOW_BAND_HZ = (150.0, 1600.0)
HIGH_BAND_HZ = (2500.0, 8000.0)


@dataclass(frozen=True)
class CvCandidate:
    absolute_ms: float | None
    status: str
    reason: str
    low_baseline_db: float | None = None


def cv_tracks(samples: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
    """Frame-centered low-band and high-band log energies (relative dB)."""
    y = np.asarray(samples, dtype=np.float64)
    if y.ndim != 1 or sample_rate < 10000:
        raise ValueError("expected mono audio with sample_rate >= 10000")
    frame = max(256, int(round(FRAME_MS * sample_rate / 1000.0)))
    hop = max(1, int(round(HOP_MS * sample_rate / 1000.0)))
    if len(y) < frame:
        raise ValueError("recording too short")
    positions = np.arange(0, len(y) - frame + 1, hop, dtype=np.int64)
    freq = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    low_mask = (freq >= LOW_BAND_HZ[0]) & (freq <= LOW_BAND_HZ[1])
    high_mask = (freq >= HIGH_BAND_HZ[0]) & (
        freq <= min(HIGH_BAND_HZ[1], sample_rate * 0.45)
    )
    if not np.any(low_mask) or not np.any(high_mask):
        raise ValueError("insufficient frequency resolution")
    window = np.hanning(frame)
    low = np.empty(len(positions), dtype=np.float64)
    high = np.empty(len(positions), dtype=np.float64)
    for k, pos in enumerate(positions):
        x = y[pos:pos + frame]
        x = x - np.mean(x)
        power = np.abs(np.fft.rfft(x * window)) ** 2
        low[k] = 10 * np.log10(np.sum(power[low_mask]) + 1e-15)
        high[k] = 10 * np.log10(np.sum(power[high_mask]) + 1e-15)
    kernel = np.ones(SMOOTH_FRAMES, dtype=np.float64) / SMOOTH_FRAMES
    low = np.convolve(low, kernel, mode="same")
    high = np.convolve(high, kernel, mode="same")
    return {
        "centers_ms": (positions + frame / 2.0) * 1000.0 / sample_rate,
        "low_db": low,
        "high_db": high,
        "low_minus_high_db": low - high,
    }


def detect_cv_boundary(
    tracks: dict[str, np.ndarray],
    consonant_onset_absolute_ms: float | None,
    *,
    min_sustain_ms: float = 32.0,
) -> CvCandidate:
    """Find a sustained vowel-dominant transition *after* frication begins.

    The early-consonant band provides an utterance-local baseline; requiring
    both an increase in low-band energy and a shift in low/high balance
    reduces false CV detections on quiet frication or transient artifacts.
    """
    if consonant_onset_absolute_ms is None or not np.isfinite(consonant_onset_absolute_ms):
        return CvCandidate(None, "unresolved", "missing_consonant_onset")
    keys = ("centers_ms", "low_db", "high_db", "low_minus_high_db")
    try:
        a = {name: np.asarray(tracks[name], dtype=np.float64) for name in keys}
    except KeyError:
        return CvCandidate(None, "unresolved", "missing_acoustic_tracks")
    length = len(a["centers_ms"])
    if (length < 20 or any(len(x) != length or not np.all(np.isfinite(x))
                           for x in a.values())):
        return CvCandidate(None, "unresolved", "invalid_acoustic_tracks")
    times, low, ratio = a["centers_ms"], a["low_db"], a["low_minus_high_db"]
    onset = float(consonant_onset_absolute_ms)

    # Evaluate baseline within the consonant's early frication phase.
    early = (times >= onset + 35.0) & (times <= onset + 110.0)
    if np.count_nonzero(early) < 5:
        return CvCandidate(None, "unresolved", "insufficient_consonant_baseline")
    baseline = float(np.percentile(low[early], 25))
    # Require some preceding high-band dominance; otherwise a following
    # vowel or a tonal noise could be misidentified as a CV transition.
    if np.count_nonzero(ratio[early] < 8.0) < 3:
        return CvCandidate(None, "unresolved", "no_frication_reference")

    window = (times >= onset + 45.0) & (times <= onset + 520.0)
    active = window & (low > baseline + 12.0) & (ratio > 12.0)
    interval = float(np.median(np.diff(times)))
    if interval <= 0:
        return CvCandidate(None, "unresolved", "invalid_time_axis")
    needed = max(3, int(np.ceil(min_sustain_ms / interval)))
    run = 0
    for i, value in enumerate(active):
        run = run + 1 if value else 0
        if run >= needed:
            first = i - needed + 1
            absolute = float(times[first] - FRAME_MS / 2.0)
            if absolute <= onset + 45.0:
                return CvCandidate(None, "review", "cv_at_search_edge", round(baseline, 3))
            return CvCandidate(round(absolute, 3), "candidate", "sustained_vowel_energy_transition", round(baseline, 3))
    return CvCandidate(None, "unresolved", "no_sustained_vowel_transition", round(baseline, 3))
