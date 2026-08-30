from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, read_wav
from .calibration_io import load_calibration_session


TEMPO_BPM = 72.0
EXPECTED_BEAT_MS = 60000.0 / TEMPO_BPM
COUNT_IN_DELAY_MS = 250.0
BEEP_DURATION_MS = 60.0
ACCENT_FREQUENCY_HZ = 1000.0
PLAIN_FREQUENCY_HZ = 700.0

_HOP_MS = 4.0
_ENVELOPE_MS = 22.0
_MIN_INTERVAL_MS = 620.0
_MAX_INTERVAL_MS = 1080.0
_INTERVAL_SIGMA_MS = 105.0
_START_SIGMA_MS = 180.0
_ONSET_LOOKBACK_MS = 80.0


@dataclass(frozen=True)
class BeepEvent:
    index: int
    role: str
    expected_frequency_hz: float
    time_ms: float
    peak_time_ms: float
    tone_score: float
    competing_score: float

    @property
    def frequency_margin(self) -> float:
        return self.tone_score - self.competing_score


@dataclass(frozen=True)
class BeepSequence:
    events: tuple[BeepEvent, ...]
    score: float
    median_interval_ms: float | None
    max_interval_error_ms: float | None


@dataclass(frozen=True)
class PromptAlignment:
    prompt_index: int
    base_unit: str
    context_family: str
    syllable: str
    wav: str
    expected_beeps: int
    detected_beeps: int
    sequence_score: float
    median_interval_ms: float | None
    max_interval_error_ms: float | None
    events: tuple[BeepEvent, ...]
    target_cues: tuple[dict[str, object], ...]


class BeepAlignmentError(RuntimeError):
    pass


def expected_beep_frequencies(token_count: int) -> tuple[float, ...]:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    return (
        ACCENT_FREQUENCY_HZ,
        ACCENT_FREQUENCY_HZ,
        *([PLAIN_FREQUENCY_HZ] * max(0, token_count - 1)),
        ACCENT_FREQUENCY_HZ,
    )


def expected_beep_roles(token_count: int) -> tuple[str, ...]:
    return (
        "count_in",
        *[f"token_{index}" for index in range(token_count)],
        "trailing",
    )


def _moving_average_complex(values: np.ndarray, width: int) -> np.ndarray:
    width = max(3, int(width))
    kernel = np.ones(width, dtype=np.float64) / width
    real = np.convolve(values.real, kernel, mode="same")
    imag = np.convolve(values.imag, kernel, mode="same")
    return real + 1j * imag


def _moving_rms(values: np.ndarray, width: int) -> np.ndarray:
    width = max(3, int(width))
    kernel = np.ones(width, dtype=np.float64) / width
    return np.sqrt(np.convolve(values * values, kernel, mode="same") + 1e-12)


def _tone_score(
    samples: np.ndarray,
    sample_rate: int,
    frequency_hz: float,
) -> np.ndarray:
    x = samples.astype(np.float64, copy=False)
    x = x - float(np.mean(x))
    n = np.arange(len(x), dtype=np.float64)
    oscillator = np.exp(-2j * np.pi * frequency_hz * n / sample_rate)
    width = max(3, int(round(_ENVELOPE_MS * sample_rate / 1000.0)))
    demodulated = _moving_average_complex(x * oscillator, width)
    amplitude = 2.0 * np.abs(demodulated)
    rms = _moving_rms(x, width)
    return np.clip(amplitude / (math.sqrt(2.0) * rms + 1e-12), 0.0, 2.5)


def _score_grid(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    hop = max(1, int(round(_HOP_MS * sample_rate / 1000.0)))
    indices = np.arange(0, len(samples), hop, dtype=np.int64)
    times_ms = indices.astype(np.float64) * 1000.0 / sample_rate
    scores = {
        PLAIN_FREQUENCY_HZ: _tone_score(samples, sample_rate, PLAIN_FREQUENCY_HZ)[indices],
        ACCENT_FREQUENCY_HZ: _tone_score(samples, sample_rate, ACCENT_FREQUENCY_HZ)[indices],
    }
    return times_ms, scores


def _frequency_evidence(
    target: np.ndarray,
    competing: np.ndarray,
) -> np.ndarray:
    margin = target - 0.55 * competing
    purity = np.maximum(target, 0.0)
    return 2.2 * margin + 0.7 * purity


def _sequence_path(
    times_ms: np.ndarray,
    scores: dict[float, np.ndarray],
    frequencies: tuple[float, ...],
) -> tuple[list[int], float]:
    if len(times_ms) < 2:
        raise BeepAlignmentError("recording is too short")

    evidence: list[np.ndarray] = []
    for frequency in frequencies:
        other = (
            ACCENT_FREQUENCY_HZ
            if frequency == PLAIN_FREQUENCY_HZ
            else PLAIN_FREQUENCY_HZ
        )
        evidence.append(_frequency_evidence(scores[frequency], scores[other]))

    n = len(times_ms)
    m = len(frequencies)
    neg_inf = -1e18
    dp = np.full((m, n), neg_inf, dtype=np.float64)
    back = np.full((m, n), -1, dtype=np.int32)

    start_mask = (times_ms >= 40.0) & (times_ms <= 760.0)
    start_penalty = ((times_ms - COUNT_IN_DELAY_MS) / _START_SIGMA_MS) ** 2
    dp[0, start_mask] = evidence[0][start_mask] - start_penalty[start_mask]

    for row in range(1, m):
        for current in range(n):
            t = times_ms[current]
            lower = t - _MAX_INTERVAL_MS
            upper = t - _MIN_INTERVAL_MS
            left = int(np.searchsorted(times_ms, lower, side="left"))
            right = int(np.searchsorted(times_ms, upper, side="right"))
            if right <= left:
                continue
            previous = dp[row - 1, left:right]
            valid = np.flatnonzero(previous > neg_inf / 2)
            if not len(valid):
                continue
            previous_indices = left + valid
            intervals = t - times_ms[previous_indices]
            timing_penalty = ((intervals - EXPECTED_BEAT_MS) / _INTERVAL_SIGMA_MS) ** 2
            values = previous[valid] - timing_penalty
            best_local = int(np.argmax(values))
            best_previous = int(previous_indices[best_local])
            dp[row, current] = values[best_local] + evidence[row][current]
            back[row, current] = best_previous

    last = int(np.argmax(dp[-1]))
    if dp[-1, last] <= neg_inf / 2:
        raise BeepAlignmentError("could not fit expected beep sequence")

    path = [last]
    for row in range(m - 1, 0, -1):
        previous = int(back[row, path[-1]])
        if previous < 0:
            raise BeepAlignmentError("incomplete beep sequence path")
        path.append(previous)
    path.reverse()
    return path, float(dp[-1, last] / m)


def _refine_onset(
    times_ms: np.ndarray,
    target_score: np.ndarray,
    peak_index: int,
) -> float:
    peak_value = float(target_score[peak_index])
    peak_time = float(times_ms[peak_index])
    lower_time = peak_time - _ONSET_LOOKBACK_MS
    lower = int(np.searchsorted(times_ms, lower_time, side="left"))
    local = target_score[lower : peak_index + 1]
    if not len(local):
        return peak_time
    baseline = float(np.percentile(local, 15.0))
    threshold = baseline + 0.32 * max(0.0, peak_value - baseline)
    candidates = np.flatnonzero(local >= threshold)
    if not len(candidates):
        return peak_time
    crossing = lower + int(candidates[0])
    return float(times_ms[crossing])


def detect_beep_sequence(
    samples: np.ndarray,
    sample_rate: int,
    token_count: int,
) -> BeepSequence:
    frequencies = expected_beep_frequencies(token_count)
    roles = expected_beep_roles(token_count)
    times_ms, scores = _score_grid(samples, sample_rate)
    path, sequence_score = _sequence_path(times_ms, scores, frequencies)

    events: list[BeepEvent] = []
    for index, (grid_index, frequency, role) in enumerate(zip(path, frequencies, roles, strict=True)):
        other = (
            ACCENT_FREQUENCY_HZ
            if frequency == PLAIN_FREQUENCY_HZ
            else PLAIN_FREQUENCY_HZ
        )
        onset_ms = _refine_onset(times_ms, scores[frequency], grid_index)
        events.append(
            BeepEvent(
                index=index,
                role=role,
                expected_frequency_hz=frequency,
                time_ms=onset_ms,
                peak_time_ms=float(times_ms[grid_index]),
                tone_score=float(scores[frequency][grid_index]),
                competing_score=float(scores[other][grid_index]),
            )
        )

    intervals = np.diff([event.time_ms for event in events])
    if len(intervals):
        median_interval = float(np.median(intervals))
        max_interval_error = float(np.max(np.abs(intervals - median_interval)))
    else:
        median_interval = None
        max_interval_error = None
    return BeepSequence(
        events=tuple(events),
        score=sequence_score,
        median_interval_ms=median_interval,
        max_interval_error_ms=max_interval_error,
    )


def _prompt_alignment(
    session_dir: Path,
    index: int,
    prompt: dict[str, object],
    recording: dict[str, object],
) -> PromptAlignment:
    tokens = prompt.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        tokens = str(prompt.get("spoken_pattern", "")).split()
    if not tokens:
        raise BeepAlignmentError("prompt has no tokens")

    wav_name = str(recording["wav"])
    wav_path = session_dir / "recordings" / wav_name
    try:
        samples, sample_rate = read_wav(wav_path)
    except AudioReadError as exc:
        raise BeepAlignmentError(str(exc)) from exc

    sequence = detect_beep_sequence(samples, sample_rate, len(tokens))
    syllable = str(prompt.get("syllable", ""))
    target_cues: list[dict[str, object]] = []
    occurrence = 0
    for token_index, token in enumerate(tokens):
        if str(token) != syllable:
            continue
        occurrence += 1
        event = sequence.events[token_index + 1]
        target_cues.append(
            {
                "occurrence": occurrence,
                "token_index": token_index,
                "token": str(token),
                "cue_ms": round(event.time_ms, 3),
                "cue_peak_ms": round(event.peak_time_ms, 3),
                "tone_score": round(event.tone_score, 6),
                "frequency_margin": round(event.frequency_margin, 6),
            }
        )

    return PromptAlignment(
        prompt_index=index,
        base_unit=str(prompt.get("base_unit", "")),
        context_family=str(prompt.get("context_family", "")),
        syllable=syllable,
        wav=wav_name,
        expected_beeps=len(tokens) + 2,
        detected_beeps=len(sequence.events),
        sequence_score=sequence.score,
        median_interval_ms=sequence.median_interval_ms,
        max_interval_error_ms=sequence.max_interval_error_ms,
        events=sequence.events,
        target_cues=tuple(target_cues),
    )


def analyze_beep_alignment(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    protocol = session["protocol"]
    recordings = session["recordings"]
    if not isinstance(protocol, dict) or not isinstance(recordings, dict):
        raise ValueError("invalid calibration session")
    prompts = protocol.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("calibration protocol is missing prompts")

    alignments: list[PromptAlignment] = []
    failed: list[dict[str, object]] = []
    for index in sorted(recordings):
        if index < 0 or index >= len(prompts):
            continue
        prompt = prompts[index]
        recording = recordings[index]
        if not isinstance(prompt, dict) or not isinstance(recording, dict):
            continue
        try:
            alignments.append(_prompt_alignment(session_dir, index, prompt, recording))
        except BeepAlignmentError as exc:
            failed.append({"prompt_index": index, "reason": str(exc)})

    intervals = [
        alignment.median_interval_ms
        for alignment in alignments
        if alignment.median_interval_ms is not None
    ]
    interval_errors = [
        alignment.max_interval_error_ms
        for alignment in alignments
        if alignment.max_interval_error_ms is not None
    ]
    margins = [
        event.frequency_margin
        for alignment in alignments
        for event in alignment.events
    ]
    scores = [alignment.sequence_score for alignment in alignments]

    payload: dict[str, object] = {
        "analysis": "calibration_beep_alignment",
        "version": "0.3",
        "session_id": session["session_id"],
        "session_dir": str(session_dir),
        "recorder_model": {
            "tempo_bpm": TEMPO_BPM,
            "expected_beat_ms": EXPECTED_BEAT_MS,
            "count_in_delay_ms": COUNT_IN_DELAY_MS,
            "beep_duration_ms": BEEP_DURATION_MS,
            "accent_frequency_hz": ACCENT_FREQUENCY_HZ,
            "plain_frequency_hz": PLAIN_FREQUENCY_HZ,
        },
        "summary": {
            "recordings": len(recordings),
            "aligned": len(alignments),
            "failed": len(failed),
            "median_actual_interval_ms": None if not intervals else round(float(np.median(intervals)), 3),
            "p90_max_interval_error_ms": None if not interval_errors else round(float(np.percentile(interval_errors, 90.0)), 3),
            "minimum_frequency_margin": None if not margins else round(float(np.min(margins)), 6),
            "median_sequence_score": None if not scores else round(float(np.median(scores)), 6),
        },
        "alignments": [
            {
                "prompt_index": item.prompt_index,
                "base_unit": item.base_unit,
                "context_family": item.context_family,
                "syllable": item.syllable,
                "wav": item.wav,
                "expected_beeps": item.expected_beeps,
                "detected_beeps": item.detected_beeps,
                "sequence_score": round(item.sequence_score, 6),
                "median_interval_ms": None if item.median_interval_ms is None else round(item.median_interval_ms, 3),
                "max_interval_error_ms": None if item.max_interval_error_ms is None else round(item.max_interval_error_ms, 3),
                "events": [
                    {
                        "index": event.index,
                        "role": event.role,
                        "expected_frequency_hz": event.expected_frequency_hz,
                        "time_ms": round(event.time_ms, 3),
                        "peak_time_ms": round(event.peak_time_ms, 3),
                        "tone_score": round(event.tone_score, 6),
                        "competing_score": round(event.competing_score, 6),
                        "frequency_margin": round(event.frequency_margin, 6),
                    }
                    for event in item.events
                ],
                "target_cues": list(item.target_cues),
            }
            for item in alignments
        ],
        "failed": failed,
    }

    output_dir = session_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "calibration_beep_alignment_v0.3.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
