"""Read-only microphone take screening; never a scientific outcome metric.

PCM16 mono features are diagnostics for operator review, not automatic
identification of a /sh/ boundary or proof of a valid singing sample.
"""
from __future__ import annotations

import math
import numpy as np

RATE = 48000


def _f0_track(audio: np.ndarray, rate: int, target_hz: float | None):
    """Autocorrelation F0 on energetic periodic frames, excluding /sh/ noise."""
    length = round(0.060 * rate)
    hop = round(0.025 * rate)
    low = max(70.0, target_hz / 1.8) if target_hz else 70.0
    high = min(800.0, target_hz * 1.8) if target_hz else 650.0
    lag_lo = max(2, math.ceil(rate / high))
    lag_hi = min(length - 2, math.floor(rate / low))
    if len(audio) < length or lag_hi <= lag_lo:
        return []
    hann = np.hanning(length)
    fft_size = 1 << (2 * length - 1).bit_length()
    values = []
    for start in range(0, len(audio) - length + 1, hop):
        frame = audio[start:start + length]
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms < 10 ** (-42 / 20):
            continue
        centered = (frame - float(np.mean(frame))) * hann
        spectrum = np.fft.rfft(centered, fft_size)
        corr = np.fft.irfft(spectrum * np.conj(spectrum), fft_size)
        if corr[0] < 1e-10:
            continue
        segment = corr[lag_lo:lag_hi + 1] / corr[0]
        j = int(np.argmax(segment))
        strength = float(segment[j])
        if strength < 0.65:
            continue
        lag = lag_lo + j
        if lag_lo < lag < lag_hi:
            a, b, c = (float(corr[lag - 1]), float(corr[lag]),
                       float(corr[lag + 1]))
            denom = a - 2.0 * b + c
            if abs(denom) > 1e-12:
                lag += float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
        values.append(rate / lag)
    return values


def analyze_pcm(pcm: bytes, *, rate: int = RATE,
                target_pitch_hz: float | None = None) -> dict:
    """Conservative operator-screening diagnostics for mono signed PCM16."""
    if len(pcm) % 2 or not pcm or rate <= 0:
        raise ValueError("nonempty, 16-bit mono PCM data required")
    if target_pitch_hz is not None and not 60 <= target_pitch_hz <= 1000:
        raise ValueError("target_pitch_hz must be 60..1000")
    samples = np.frombuffer(pcm, dtype="<i2")
    audio = samples.astype(np.float64) / 32768.0
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio * audio)))
    peak_dbfs = round(20 * math.log10(max(peak, 1e-10)), 2)
    rms_dbfs = round(20 * math.log10(max(rms, 1e-10)), 2)
    clipped = int(np.count_nonzero((samples >= 32760) | (samples <= -32760)))
    estimated = _f0_track(audio, rate, target_pitch_hz)
    median = float(np.median(estimated)) if estimated else None
    deviation = (round(1200 * math.log2(median / target_pitch_hz), 1)
                 if median is not None and target_pitch_hz is not None else None)
    spread = (round(float(np.percentile(1200 * np.log2(
                 np.asarray(estimated) / median), 90) -
                 np.percentile(1200 * np.log2(
                 np.asarray(estimated) / median), 10)), 1)
              if len(estimated) >= 4 else None)
    flags = []
    if clipped:
        flags.append("digital_clipping")
    elif peak_dbfs > -3:
        flags.append("low_peak_headroom")
    if peak_dbfs < -30:
        flags.append("low_input_level")
    if len(estimated) < 6:
        flags.append("no_reliable_voiced_segment")
    elif deviation is not None and abs(deviation) > 100:
        flags.append("pitch_outside_one_semitone")
    if spread is not None and spread > 100:
        flags.append("pitch_variability")
    return {
        "screen_version": "phonoweave.capture_screen.v1",
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "clipped_sample_count": clipped,
        "voiced_frames": len(estimated),
        "estimated_f0_median_hz": round(median, 2) if median is not None else None,
        "target_deviation_cents": deviation,
        "f0_p10_p90_spread_cents": spread,
        "flags": flags,
        "screen_status": "manual_review" if flags else "screen_pass",
        "not_validated": "F0 is approximate, vowel segmentation and phonetic accuracy "
                         "require independent human review",
    }


def format_screen(result: dict) -> str:
    f0 = result["estimated_f0_median_hz"]
    pitch = f"{f0:.1f} Hz" if f0 is not None else "unavailable"
    flags = ", ".join(result["flags"]) if result["flags"] else "none"
    return (f"peak {result['peak_dbfs']:.1f} dBFS; "
            f"estimated voiced F0 {pitch}; review flags: {flags}")
