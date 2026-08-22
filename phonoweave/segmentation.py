from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import AudioReadError, AudioSegment, read_wav
from .oto import OtoEntry


@dataclass(frozen=True)
class AffricateSegmentation:
    frication: AudioSegment
    release_ms: float
    vowel_onset_ms: float
    release_strength: float
    vowel_periodicity: float


def _zscore(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    scale = float(np.std(values))
    if scale < 1e-9:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / scale


def _frame_positions(length: int, frame: int, hop: int) -> list[int]:
    if length < frame:
        return []
    return list(range(0, length - frame + 1, hop))


def _spectral_stats(frame: np.ndarray, sample_rate: int) -> tuple[float, float]:
    frame = frame.astype(np.float64, copy=False)
    frame = frame - np.mean(frame)
    power = np.abs(np.fft.rfft(frame * np.hanning(len(frame)))) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sample_rate)

    valid = freqs >= 500.0
    band = power[valid]
    if not len(band):
        return 0.0, 0.0

    flatness = float(np.exp(np.mean(np.log(band))) / np.mean(band))
    high = float(np.sum(power[freqs >= 2000.0]) / np.sum(power))
    return flatness, high


def _periodicity(frame: np.ndarray, sample_rate: int) -> float:
    frame = frame.astype(np.float64, copy=False)
    frame = frame - np.mean(frame)
    energy = float(np.dot(frame, frame))
    if energy < 1e-12:
        return 0.0

    min_lag = max(1, int(sample_rate / 500.0))
    max_lag = min(len(frame) - 2, int(sample_rate / 70.0))
    if max_lag <= min_lag:
        return 0.0

    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        left = frame[:-lag]
        right = frame[lag:]
        denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
        if denom > 1e-12:
            best = max(best, float(np.dot(left, right) / denom))
    return max(0.0, min(best, 1.0))


def _detect_release(
    samples: np.ndarray,
    sample_rate: int,
    start_ms: float,
    end_ms: float,
) -> tuple[float, float]:
    start = max(0, int(round(start_ms * sample_rate / 1000.0)))
    end = min(len(samples), int(round(end_ms * sample_rate / 1000.0)))
    region = samples[start:end]

    frame = max(64, int(round(sample_rate * 0.004)))
    hop = max(1, int(round(sample_rate * 0.001)))
    positions = _frame_positions(len(region), frame, hop)
    if len(positions) < 6:
        raise AudioReadError("affricate window is too short for release detection")

    rms_db: list[float] = []
    high_ratio: list[float] = []
    centers_ms: list[float] = []
    for position in positions:
        chunk = region[position : position + frame]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) + 1e-12)
        _, high = _spectral_stats(chunk, sample_rate)
        rms_db.append(20.0 * np.log10(rms))
        high_ratio.append(high)
        centers_ms.append(start_ms + 1000.0 * (position + frame / 2.0) / sample_rate)

    rms_array = np.asarray(rms_db, dtype=np.float64)
    high_array = np.asarray(high_ratio, dtype=np.float64)
    drms = np.diff(rms_array, prepend=rms_array[0])
    dhigh = np.diff(high_array, prepend=high_array[0])
    score = _zscore(drms) + 0.45 * _zscore(dhigh) + 0.20 * _zscore(high_array)

    centers = np.asarray(centers_ms, dtype=np.float64)
    candidate = (centers >= start_ms + 3.0) & (centers <= end_ms - 12.0)
    if not np.any(candidate):
        raise AudioReadError("no usable release search interval")

    indices = np.flatnonzero(candidate)
    best_index = int(indices[np.argmax(score[indices])])
    release_ms = float(centers[best_index])
    strength = float(score[best_index] - np.median(score[indices]))
    return release_ms, strength


def _detect_vowel_onset(
    samples: np.ndarray,
    sample_rate: int,
    prior_ms: float,
    release_ms: float,
) -> tuple[float, float]:
    search_start_ms = max(release_ms + 12.0, prior_ms - 22.0)
    search_end_ms = prior_ms + 28.0
    start = max(0, int(round(search_start_ms * sample_rate / 1000.0)))
    end = min(len(samples), int(round(search_end_ms * sample_rate / 1000.0)))
    region = samples[start:end]

    frame = max(128, int(round(sample_rate * 0.018)))
    hop = max(1, int(round(sample_rate * 0.002)))
    positions = _frame_positions(len(region), frame, hop)
    if not positions:
        return prior_ms, 0.0

    rows: list[tuple[float, float, float, float]] = []
    rms_values: list[float] = []
    for position in positions:
        chunk = region[position : position + frame]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) + 1e-12)
        flatness, _ = _spectral_stats(chunk, sample_rate)
        periodicity = _periodicity(chunk, sample_rate)
        center_ms = search_start_ms + 1000.0 * (position + frame / 2.0) / sample_rate
        rows.append((center_ms, periodicity, flatness, rms))
        rms_values.append(rms)

    rms_floor = max(rms_values) * 0.08
    for center_ms, periodicity, flatness, rms in rows:
        if center_ms < release_ms + 15.0:
            continue
        if periodicity >= 0.48 and flatness <= 0.45 and rms >= rms_floor:
            onset = center_ms - 0.25 * (1000.0 * frame / sample_rate)
            return float(onset), periodicity

    best = max(rows, key=lambda row: row[1] - 0.25 * row[2])
    if abs(best[0] - prior_ms) <= 24.0 and best[1] >= 0.30:
        onset = best[0] - 0.25 * (1000.0 * frame / sample_rate)
        return float(onset), float(best[1])
    return prior_ms, float(best[1])


def detect_affricate_frication(entry: OtoEntry) -> AffricateSegmentation:
    if entry.preutterance <= 0:
        raise AudioReadError("preutterance is not positive")

    samples, sample_rate = read_wav(entry.wav_path)
    coarse_start_ms = max(0.0, entry.offset)
    coarse_end_ms = entry.offset + entry.preutterance
    if coarse_end_ms - coarse_start_ms < 22.0:
        raise AudioReadError("affricate window is too short")

    release_ms, release_strength = _detect_release(
        samples,
        sample_rate,
        coarse_start_ms,
        coarse_end_ms,
    )
    vowel_onset_ms, vowel_periodicity = _detect_vowel_onset(
        samples,
        sample_rate,
        coarse_end_ms,
        release_ms,
    )

    frication_start_ms = release_ms + 2.0
    frication_end_ms = vowel_onset_ms
    if frication_end_ms - frication_start_ms < 14.0:
        raise AudioReadError("detected frication is too short")

    start = max(0, int(round(frication_start_ms * sample_rate / 1000.0)))
    end = min(len(samples), int(round(frication_end_ms * sample_rate / 1000.0)))
    if end - start < 32:
        raise AudioReadError("detected frication is outside the WAV")

    segment = AudioSegment(
        samples=samples[start:end],
        sample_rate=sample_rate,
        start_ms=frication_start_ms,
        end_ms=frication_end_ms,
    )
    return AffricateSegmentation(
        frication=segment,
        release_ms=release_ms,
        vowel_onset_ms=vowel_onset_ms,
        release_strength=release_strength,
        vowel_periodicity=vowel_periodicity,
    )
