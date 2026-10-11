"""Experimental transition-based high-band frication onset detector.

This uses the captured recording's nominal beat schedule, NOT the legacy
microphone-inferred beep sequence. The estimate identifies an acoustic
broadband-frication onset, not phonemic identity or a confirmed gold label.

This initial single-speaker method is unvalidated on held-out sessions.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

FRAME_MS = 20.0
HOP_MS = 4.0
LEFT_MS = 350.0
RIGHT_MS = 500.0
MIN_SUSTAIN_MS = 40.0
MAX_HIGH_HZ = 10000.0
HIGH_START_HZ = 2500.0
MIN_HIGH_RISE_DB = 12.0
MIN_HIGH_FRACTION_DB = -12.0
MIN_ABSOLUTE_HIGH_DBFS = -90.0


@dataclass(frozen=True)
class TransitionCandidate:
    onset_ms: float | None  # absolute time within the WAV
    status: str
    reason: str
    high_band_baseline_db: float | None = None
    frication_duration_ms: float | None = None


def acoustic_tracks(
    samples: np.ndarray, sample_rate: int,
    *, frame_ms: float = FRAME_MS, hop_ms: float = HOP_MS,
) -> dict[str, np.ndarray]:
    """Extract RMS and normalized high-band energy using numpy alone."""
    y = np.asarray(samples, dtype=np.float64)
    if y.ndim != 1 or sample_rate <= 0:
        raise ValueError("expected mono samples with positive sample rate")
    n = max(256, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if len(y) < n:
        raise ValueError("recording too short")
    positions = np.arange(0, len(y) - n + 1, hop, dtype=np.int64)
    window = np.hanning(n)
    win_power = float(np.sum(window**2))
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    high_band = (frequencies >= HIGH_START_HZ) & (
        frequencies <= min(MAX_HIGH_HZ, 0.45 * sample_rate)
    )
    if not np.any(high_band):
        raise ValueError("sample rate too low for high-band frication analysis")

    high_db = np.empty(len(positions), dtype=np.float64)
    rms_db = np.empty(len(positions), dtype=np.float64)
    fraction_db = np.empty(len(positions), dtype=np.float64)
    for k, index in enumerate(positions):
        chunk = y[index:index+n]
        chunk = chunk - float(np.mean(chunk))
        spectrum = np.fft.rfft(chunk * window)
        power = np.abs(spectrum)**2
        band_rms = float(np.sqrt(2 * np.sum(power[high_band]) / (n * win_power) + 1e-18))
        overall = float(np.sqrt(np.mean(chunk**2) + 1e-18))
        high_db[k] = 20 * np.log10(band_rms)
        rms_db[k] = 20 * np.log10(overall)
        fraction_db[k] = high_db[k] - rms_db[k]
    return {
        "centers_ms": (positions + n / 2.0) * 1000.0 / sample_rate,
        "high_dbfs": high_db,
        "rms_dbfs": rms_db,
        "high_fraction_db": fraction_db,
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    transitions = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return list(zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)))


def detect_transition(
    tracks: dict[str, np.ndarray],
    nominal_cue_ms: float,
    *, min_sustain_ms: float = MIN_SUSTAIN_MS,
) -> TransitionCandidate:
    """Return the first *sustained transition* near a nominal browser beat.

    The 40ms persistence check rejects short clicks and search-edge spikes.
    Borderline cases abstain rather than returning a guessed timestamp.
    """
    centers = np.asarray(tracks["centers_ms"], dtype=np.float64)
    high = np.asarray(tracks["high_dbfs"], dtype=np.float64)
    ratio = np.asarray(tracks["high_fraction_db"], dtype=np.float64)
    if (len(centers) < 12 or len(high) != len(centers)
            or len(ratio) != len(centers)
            or not all(np.all(np.isfinite(x)) for x in (centers, high, ratio))):
        return TransitionCandidate(None, "unresolved", "invalid_feature_tracks")
    in_window = (centers >= nominal_cue_ms - LEFT_MS) & (
        centers <= nominal_cue_ms + RIGHT_MS
    )
    indices = np.flatnonzero(in_window)
    if len(indices) < 50:
        return TransitionCandidate(None, "unresolved", "insufficient_search_window")
    times = centers[indices]
    values = high[indices]
    fractions = ratio[indices]
    initial = values[:min(45, len(values))]
    baseline = float(np.percentile(initial, 20))
    passing = (
        (values >= baseline + MIN_HIGH_RISE_DB)
        & (fractions >= MIN_HIGH_FRACTION_DB)
        & (values >= MIN_ABSOLUTE_HIGH_DBFS)
    )
    spacing = float(np.median(np.diff(times)))
    needed = max(3, int(np.ceil(min_sustain_ms / spacing)))
    viable = [(s, e) for s, e in _runs(passing) if e - s >= needed]
    if not viable:
        return TransitionCandidate(None, "unresolved", "no_sustained_high_band_transition", baseline)
    first, end = viable[0]
    # When already active at the left edge, no acoustic transition is
    # observable in the window. Do not invent an onset.
    if first < 4:
        return TransitionCandidate(None, "review", "frication_at_search_edge", baseline)

    onset = float(times[first] - FRAME_MS/2)
    duration = float((end - first) * spacing)
    return TransitionCandidate(
        round(onset, 3), "candidate", "sustained_high_band_transition",
        round(baseline, 3), round(duration, 3)
    )


def nominal_target_cue_ms(token_index: int, *, bpm: float = 72.0) -> float:
    """Replicate browser recorder timing: 250 ms count-in + N beats.

    No precise synchronization is recorded by the current browser version.
    This clock is only a search prior, not a ground-truth cue location.
    """
    if token_index < 0 or bpm <= 0:
        raise ValueError("invalid token index or BPM")
    return 250.0 + (token_index + 1) * 60000.0 / bpm
