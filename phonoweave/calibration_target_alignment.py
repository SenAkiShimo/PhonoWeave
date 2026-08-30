from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, read_wav
from .calibration_beep_alignment import (
    ACCENT_FREQUENCY_HZ,
    PLAIN_FREQUENCY_HZ,
    BeepEvent,
    detect_beep_sequence,
)
from .calibration_io import load_calibration_session
from .stop import _detect_release


_ONSET_CLASS = {
    **{item: "stop" for item in ("b", "p", "d", "t", "g", "k")},
    **{item: "affricate" for item in ("zh", "ch", "z", "c", "j", "q")},
    **{item: "fricative" for item in ("f", "h", "x", "sh", "s")},
    **{item: "nasal" for item in ("m", "n")},
    "l": "lateral",
    "r": "rhotic",
}

_SEARCH_START_MS = 78.0
_SEARCH_END_MS = 560.0
_BEEP_SUBTRACT_START_MS = -12.0
_BEEP_SUBTRACT_END_MS = 105.0
_REPEAT_GOOD_MS = 95.0
_REPEAT_REVIEW_MS = 150.0


@dataclass(frozen=True)
class TargetAnchor:
    occurrence: int
    token_index: int
    cue_ms: float
    anchor_ms: float
    cue_to_anchor_ms: float
    anchor_type: str
    strength: float
    boundary_distance_ms: float


@dataclass(frozen=True)
class TargetPromptAlignment:
    prompt_index: int
    base_unit: str
    class_name: str
    context_family: str
    syllable: str
    wav: str
    anchors: tuple[TargetAnchor, ...]
    repeat_spread_ms: float | None
    qc: str


class TargetAlignmentError(RuntimeError):
    pass


def _suppress_beeps(
    samples: np.ndarray,
    sample_rate: int,
    events: tuple[BeepEvent, ...],
) -> np.ndarray:
    cleaned = samples.astype(np.float64, copy=True)
    for event in events:
        frequency = float(event.expected_frequency_hz)
        start_ms = event.time_ms + _BEEP_SUBTRACT_START_MS
        end_ms = event.time_ms + _BEEP_SUBTRACT_END_MS
        start = max(0, int(round(start_ms * sample_rate / 1000.0)))
        end = min(len(cleaned), int(round(end_ms * sample_rate / 1000.0)))
        if end - start < 32:
            continue
        local = cleaned[start:end].copy()
        t = np.arange(start, end, dtype=np.float64) / sample_rate
        sin = np.sin(2.0 * np.pi * frequency * t)
        cos = np.cos(2.0 * np.pi * frequency * t)
        design = np.column_stack([sin, cos])
        try:
            coeff, *_ = np.linalg.lstsq(design, local, rcond=None)
        except np.linalg.LinAlgError:
            continue
        tone = design @ coeff
        taper = np.ones(len(local), dtype=np.float64)
        edge = max(1, min(len(local) // 3, int(round(0.012 * sample_rate))))
        if edge > 1:
            fade = np.linspace(0.0, 1.0, edge, endpoint=False)
            taper[:edge] = fade
            taper[-edge:] = fade[::-1]
        cleaned[start:end] = local - taper * tone
    return cleaned


def _frame_positions(length: int, frame: int, hop: int) -> list[int]:
    if length < frame:
        return []
    return list(range(0, length - frame + 1, hop))


def _periodicity(frame: np.ndarray, sample_rate: int) -> float:
    x = frame.astype(np.float64, copy=False)
    x = x - float(np.mean(x))
    energy = float(np.dot(x, x))
    if energy < 1e-12:
        return 0.0
    min_lag = max(1, int(sample_rate / 500.0))
    max_lag = min(len(x) - 2, int(sample_rate / 70.0))
    if max_lag <= min_lag:
        return 0.0
    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        left = x[:-lag]
        right = x[lag:]
        denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
        if denom > 1e-12:
            best = max(best, float(np.dot(left, right) / denom))
    return max(0.0, min(1.0, best))


def _frame_features(
    samples: np.ndarray,
    sample_rate: int,
    start_ms: float,
    end_ms: float,
    frame_ms: float = 14.0,
    hop_ms: float = 3.0,
) -> dict[str, np.ndarray]:
    start = max(0, int(round(start_ms * sample_rate / 1000.0)))
    end = min(len(samples), int(round(end_ms * sample_rate / 1000.0)))
    region = samples[start:end]
    frame = max(128, int(round(frame_ms * sample_rate / 1000.0)))
    hop = max(1, int(round(hop_ms * sample_rate / 1000.0)))
    positions = _frame_positions(len(region), frame, hop)
    if len(positions) < 8:
        raise TargetAlignmentError("target search window is too short")

    centers: list[float] = []
    rms_db: list[float] = []
    high_db: list[float] = []
    flatness: list[float] = []
    periodicity: list[float] = []
    zcr: list[float] = []

    for position in positions:
        chunk = region[position : position + frame].astype(np.float64, copy=False)
        chunk = chunk - float(np.mean(chunk))
        rms = float(np.sqrt(np.mean(chunk * chunk)) + 1e-12)
        power = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)))) ** 2 + 1e-18
        freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sample_rate)
        valid = (freqs >= 250.0) & (freqs <= min(10000.0, 0.45 * sample_rate))
        band = power[valid]
        band_freqs = freqs[valid]
        total = float(np.sum(band)) if len(band) else 1e-18
        high = float(np.sum(band[band_freqs >= 2500.0])) if len(band) else 1e-18
        centers.append(start_ms + 1000.0 * (position + frame / 2.0) / sample_rate)
        rms_db.append(20.0 * math.log10(rms))
        high_db.append(10.0 * math.log10(high + 1e-18))
        flatness.append(float(np.exp(np.mean(np.log(band))) / np.mean(band)) if len(band) else 0.0)
        periodicity.append(_periodicity(chunk, sample_rate))
        zcr.append(float(np.mean((chunk[:-1] >= 0.0) != (chunk[1:] >= 0.0))))

    return {
        "centers": np.asarray(centers, dtype=np.float64),
        "rms_db": np.asarray(rms_db, dtype=np.float64),
        "high_db": np.asarray(high_db, dtype=np.float64),
        "flatness": np.asarray(flatness, dtype=np.float64),
        "periodicity": np.asarray(periodicity, dtype=np.float64),
        "zcr": np.asarray(zcr, dtype=np.float64),
    }


def _robust_z(values: np.ndarray) -> np.ndarray:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = max(1e-6, 1.4826 * mad)
    return (values - center) / scale


def _first_sustained(
    score: np.ndarray,
    threshold: float,
    minimum_frames: int,
) -> int | None:
    count = 0
    for index, value in enumerate(score):
        if value >= threshold:
            count += 1
            if count >= minimum_frames:
                return index - minimum_frames + 1
        else:
            count = 0
    return None


def _detect_noise_anchor(
    samples: np.ndarray,
    sample_rate: int,
    cue_ms: float,
    class_name: str,
) -> tuple[float, float]:
    start_ms = cue_ms + _SEARCH_START_MS
    end_ms = cue_ms + _SEARCH_END_MS
    f = _frame_features(samples, sample_rate, start_ms, end_ms)
    centers = f["centers"]
    high_z = _robust_z(f["high_db"])
    flat_z = _robust_z(f["flatness"])
    zcr_z = _robust_z(f["zcr"])
    rms_z = _robust_z(f["rms_db"])
    periodicity = f["periodicity"]

    if class_name == "affricate":
        derivative = np.diff(high_z, prepend=high_z[0])
        score = 1.25 * high_z + 0.55 * flat_z + 0.35 * zcr_z + 0.45 * derivative - 0.20 * periodicity
        best = int(np.argmax(score))
        backward = max(0, best - 10)
        local = score[backward : best + 1]
        threshold = max(0.15, float(np.percentile(local, 45.0)))
        candidate = _first_sustained(local, threshold, 2)
        chosen = best if candidate is None else backward + candidate
    else:
        if class_name == "fricative":
            score = 1.05 * high_z + 0.65 * flat_z + 0.35 * zcr_z + 0.20 * rms_z - 0.30 * periodicity
        else:
            score = 0.75 * high_z + 0.45 * flat_z + 0.25 * zcr_z + 0.35 * rms_z - 0.15 * periodicity
        high_threshold = float(np.percentile(score, 72.0))
        candidate = _first_sustained(score, high_threshold, 3)
        if candidate is None:
            chosen = int(np.argmax(score))
        else:
            chosen = candidate
    strength = float(score[chosen] - np.median(score))
    return float(centers[chosen] - 7.0), strength


def _detect_sonorant_anchor(
    samples: np.ndarray,
    sample_rate: int,
    cue_ms: float,
    class_name: str,
) -> tuple[float, float]:
    start_ms = cue_ms + _SEARCH_START_MS
    end_ms = cue_ms + _SEARCH_END_MS
    f = _frame_features(samples, sample_rate, start_ms, end_ms, frame_ms=18.0, hop_ms=3.0)
    centers = f["centers"]
    rms_z = _robust_z(f["rms_db"])
    periodicity = f["periodicity"]
    flatness_z = _robust_z(f["flatness"])

    if class_name == "rhotic":
        score = 0.75 * rms_z + 1.10 * periodicity - 0.20 * flatness_z
        threshold = max(0.25, float(np.percentile(score, 58.0)))
    else:
        score = 0.85 * rms_z + 1.35 * periodicity - 0.25 * flatness_z
        threshold = max(0.35, float(np.percentile(score, 55.0)))
    candidate = _first_sustained(score, threshold, 3)
    if candidate is None:
        chosen = int(np.argmax(score))
    else:
        chosen = candidate
    strength = float(score[chosen] - np.median(score))
    return float(centers[chosen] - 9.0), strength


def _detect_anchor(
    samples: np.ndarray,
    sample_rate: int,
    cue_ms: float,
    class_name: str,
) -> tuple[float, str, float]:
    search_start = cue_ms + _SEARCH_START_MS
    search_end = cue_ms + _SEARCH_END_MS
    if class_name == "stop":
        try:
            anchor_ms, strength = _detect_release(
                samples,
                sample_rate,
                search_start,
                search_end,
            )
        except AudioReadError as exc:
            raise TargetAlignmentError(str(exc)) from exc
        return anchor_ms, "release", strength
    if class_name == "affricate":
        anchor_ms, strength = _detect_noise_anchor(samples, sample_rate, cue_ms, class_name)
        return anchor_ms, "frication_release", strength
    if class_name == "fricative":
        anchor_ms, strength = _detect_noise_anchor(samples, sample_rate, cue_ms, class_name)
        return anchor_ms, "frication_onset", strength
    if class_name in {"nasal", "lateral", "rhotic"}:
        anchor_ms, strength = _detect_sonorant_anchor(samples, sample_rate, cue_ms, class_name)
        return anchor_ms, "sonorant_onset", strength
    anchor_ms, strength = _detect_noise_anchor(samples, sample_rate, cue_ms, "other")
    return anchor_ms, "acoustic_onset", strength


def _qc_for_anchors(anchors: list[TargetAnchor]) -> tuple[float | None, str]:
    if len(anchors) < 2:
        return None, "insufficient_repeats"
    latencies = [anchor.cue_to_anchor_ms for anchor in anchors]
    spread = float(max(latencies) - min(latencies))
    boundary = min(anchor.boundary_distance_ms for anchor in anchors)
    weak = min(anchor.strength for anchor in anchors)
    if boundary < 20.0:
        return spread, "boundary_review"
    if spread > _REPEAT_REVIEW_MS:
        return spread, "repeat_timing_review"
    if spread > _REPEAT_GOOD_MS or weak < -0.15:
        return spread, "usable_with_review"
    return spread, "usable"


def _prompt_alignment(
    session_dir: Path,
    index: int,
    prompt: dict[str, object],
    recording: dict[str, object],
) -> TargetPromptAlignment:
    tokens = prompt.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        tokens = str(prompt.get("spoken_pattern", "")).split()
    syllable = str(prompt.get("syllable", ""))
    target_indices = [i for i, token in enumerate(tokens) if str(token) == syllable]
    if not target_indices:
        raise TargetAlignmentError("target token not found")

    wav_name = str(recording["wav"])
    wav_path = session_dir / "recordings" / wav_name
    try:
        samples, sample_rate = read_wav(wav_path)
    except AudioReadError as exc:
        raise TargetAlignmentError(str(exc)) from exc

    sequence = detect_beep_sequence(samples, sample_rate, len(tokens))
    cleaned = _suppress_beeps(samples, sample_rate, sequence.events)
    base_unit = str(prompt.get("base_unit", ""))
    class_name = _ONSET_CLASS.get(base_unit, "other")

    anchors: list[TargetAnchor] = []
    for occurrence, token_index in enumerate(target_indices, start=1):
        cue = sequence.events[token_index + 1]
        anchor_ms, anchor_type, strength = _detect_anchor(
            cleaned,
            sample_rate,
            cue.time_ms,
            class_name,
        )
        latency = anchor_ms - cue.time_ms
        lower = abs(latency - _SEARCH_START_MS)
        upper = abs(_SEARCH_END_MS - latency)
        anchors.append(
            TargetAnchor(
                occurrence=occurrence,
                token_index=token_index,
                cue_ms=cue.time_ms,
                anchor_ms=anchor_ms,
                cue_to_anchor_ms=latency,
                anchor_type=anchor_type,
                strength=strength,
                boundary_distance_ms=min(lower, upper),
            )
        )

    repeat_spread, qc = _qc_for_anchors(anchors)
    return TargetPromptAlignment(
        prompt_index=index,
        base_unit=base_unit,
        class_name=class_name,
        context_family=str(prompt.get("context_family", "")),
        syllable=syllable,
        wav=wav_name,
        anchors=tuple(anchors),
        repeat_spread_ms=repeat_spread,
        qc=qc,
    )


def analyze_target_alignment(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    protocol = session["protocol"]
    recordings = session["recordings"]
    if not isinstance(protocol, dict) or not isinstance(recordings, dict):
        raise ValueError("invalid calibration session")
    prompts = protocol.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("calibration protocol is missing prompts")

    rows: list[TargetPromptAlignment] = []
    failed: list[dict[str, object]] = []
    for index in sorted(recordings):
        if index < 0 or index >= len(prompts):
            continue
        prompt = prompts[index]
        recording = recordings[index]
        if not isinstance(prompt, dict) or not isinstance(recording, dict):
            continue
        try:
            rows.append(_prompt_alignment(session_dir, index, prompt, recording))
        except TargetAlignmentError as exc:
            failed.append({"prompt_index": index, "reason": str(exc)})

    spreads = [row.repeat_spread_ms for row in rows if row.repeat_spread_ms is not None]
    latencies = [anchor.cue_to_anchor_ms for row in rows for anchor in row.anchors]
    qc_counts: dict[str, int] = {}
    for row in rows:
        qc_counts[row.qc] = qc_counts.get(row.qc, 0) + 1

    payload: dict[str, object] = {
        "analysis": "calibration_target_alignment",
        "version": "0.3",
        "session_id": session["session_id"],
        "session_dir": str(session_dir),
        "summary": {
            "recordings": len(recordings),
            "aligned": len(rows),
            "failed": len(failed),
            "observations": sum(len(row.anchors) for row in rows),
            "median_cue_to_anchor_ms": None if not latencies else round(float(np.median(latencies)), 3),
            "p90_repeat_spread_ms": None if not spreads else round(float(np.percentile(spreads, 90.0)), 3),
            "max_repeat_spread_ms": None if not spreads else round(float(np.max(spreads)), 3),
            "qc_counts": qc_counts,
        },
        "alignments": [
            {
                "prompt_index": row.prompt_index,
                "base_unit": row.base_unit,
                "class_name": row.class_name,
                "context_family": row.context_family,
                "syllable": row.syllable,
                "wav": row.wav,
                "repeat_spread_ms": None if row.repeat_spread_ms is None else round(row.repeat_spread_ms, 3),
                "qc": row.qc,
                "anchors": [
                    {
                        "occurrence": anchor.occurrence,
                        "token_index": anchor.token_index,
                        "cue_ms": round(anchor.cue_ms, 3),
                        "anchor_ms": round(anchor.anchor_ms, 3),
                        "cue_to_anchor_ms": round(anchor.cue_to_anchor_ms, 3),
                        "anchor_type": anchor.anchor_type,
                        "strength": round(anchor.strength, 6),
                        "boundary_distance_ms": round(anchor.boundary_distance_ms, 3),
                    }
                    for anchor in row.anchors
                ],
            }
            for row in rows
        ],
        "failed": failed,
    }

    output_dir = session_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "calibration_target_alignment_v0.3.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
