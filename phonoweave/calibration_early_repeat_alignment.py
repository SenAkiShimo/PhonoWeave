from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, read_wav
from .calibration_beep_alignment import BeepEvent, detect_beep_sequence
from .calibration_io import load_calibration_session


WINDOW_START_MS = 105.0
WINDOW_END_MS = 430.0
FRAME_MS = 18.0
HOP_MS = 4.0
MAX_LAG_MS = 160.0
MIN_OVERLAP_MS = 185.0

_BEEP_SUBTRACT_START_MS = -12.0
_BEEP_SUBTRACT_END_MS = 112.0

_CONSENSUS_REVIEW_MS = 52.0
_EDGE_MARGIN_MS = 12.0
_SHAPE_SCORE_LOW = 0.12
_EVENT_SCORE_LOW = 0.10


_ONSET_CLASS = {
    **{item: "stop" for item in ("b", "p", "d", "t", "g", "k")},
    **{item: "affricate" for item in ("zh", "ch", "z", "c", "j", "q")},
    **{item: "fricative" for item in ("f", "h", "x", "sh", "s")},
    **{item: "sonorant" for item in ("m", "n", "l", "r")},
}


@dataclass(frozen=True)
class LagEstimate:
    lag_ms: float
    score: float
    score_margin: float
    edge_hit: bool


@dataclass(frozen=True)
class EarlyRepeatAlignment:
    prompt_index: int
    base_unit: str
    class_name: str
    context_family: str
    syllable: str
    wav: str
    occurrence_1_cue_ms: float
    occurrence_2_cue_ms: float
    shape: LagEstimate
    event: LagEstimate
    lag_disagreement_ms: float
    consensus_lag_ms: float
    qc: str


class EarlyRepeatAlignmentError(RuntimeError):
    pass


def _suppress_beeps(
    samples: np.ndarray,
    sample_rate: int,
    events: tuple[BeepEvent, ...],
) -> np.ndarray:
    cleaned = samples.astype(np.float64, copy=True)
    for event in events:
        start_ms = event.time_ms + _BEEP_SUBTRACT_START_MS
        end_ms = event.time_ms + _BEEP_SUBTRACT_END_MS
        start = max(0, int(round(start_ms * sample_rate / 1000.0)))
        end = min(len(cleaned), int(round(end_ms * sample_rate / 1000.0)))
        if end - start < 32:
            continue
        local = cleaned[start:end].copy()
        t = np.arange(start, end, dtype=np.float64) / sample_rate
        frequency = float(event.expected_frequency_hz)
        design = np.column_stack(
            [
                np.sin(2.0 * np.pi * frequency * t),
                np.cos(2.0 * np.pi * frequency * t),
            ]
        )
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


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if values.ndim == 1:
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        scale = max(1e-6, 1.4826 * mad)
        return (values - center) / scale
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    scale = np.maximum(1e-6, 1.4826 * mad)
    return (values - center) / scale


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


def _frame_tracks(
    samples: np.ndarray,
    sample_rate: int,
    cue_ms: float,
    class_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    start_ms = cue_ms + WINDOW_START_MS
    end_ms = cue_ms + WINDOW_END_MS
    start = max(0, int(round(start_ms * sample_rate / 1000.0)))
    end = min(len(samples), int(round(end_ms * sample_rate / 1000.0)))
    region = samples[start:end]

    frame = max(128, int(round(FRAME_MS * sample_rate / 1000.0)))
    hop = max(1, int(round(HOP_MS * sample_rate / 1000.0)))
    if len(region) < frame:
        raise EarlyRepeatAlignmentError("early repeat window is too short")
    positions = list(range(0, len(region) - frame + 1, hop))
    if len(positions) < 30:
        raise EarlyRepeatAlignmentError("early repeat window has too few frames")

    rows: list[dict[str, float]] = []
    previous_log_power: np.ndarray | None = None
    for position in positions:
        chunk = region[position : position + frame].astype(np.float64, copy=False)
        chunk = chunk - float(np.mean(chunk))
        windowed = chunk * np.hanning(len(chunk))
        power = np.abs(np.fft.rfft(windowed)) ** 2 + 1e-18
        freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sample_rate)

        total_mask = (freqs >= 180.0) & (freqs <= min(9000.0, 0.45 * sample_rate))
        low_mask = (freqs >= 180.0) & (freqs < 1200.0)
        mid_mask = (freqs >= 1200.0) & (freqs < 3500.0)
        high_mask = (freqs >= 3500.0) & (freqs <= min(9000.0, 0.45 * sample_rate))
        total = float(np.sum(power[total_mask])) + 1e-18
        low = float(np.sum(power[low_mask])) + 1e-18
        mid = float(np.sum(power[mid_mask])) + 1e-18
        high = float(np.sum(power[high_mask])) + 1e-18
        centroid = float(np.sum(freqs[total_mask] * power[total_mask]) / total)
        flatness = float(
            np.exp(np.mean(np.log(power[total_mask]))) / np.mean(power[total_mask])
        )
        rms = float(np.sqrt(np.mean(chunk * chunk)) + 1e-12)
        zcr = float(np.mean((chunk[:-1] >= 0.0) != (chunk[1:] >= 0.0)))
        periodicity = _periodicity(chunk, sample_rate)

        log_power = np.log(power[total_mask])
        if previous_log_power is None or len(previous_log_power) != len(log_power):
            flux = 0.0
        else:
            positive = np.maximum(log_power - previous_log_power, 0.0)
            flux = float(np.sqrt(np.mean(positive * positive)))
        previous_log_power = log_power

        rows.append(
            {
                "rms": 20.0 * math.log10(rms),
                "low": math.log(low),
                "mid": math.log(mid),
                "high": math.log(high),
                "centroid": centroid,
                "flatness": flatness,
                "zcr": zcr,
                "periodicity": periodicity,
                "flux": flux,
            }
        )

    names = tuple(rows[0])
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64)
    columns = {name: _robust_z(matrix[:, i]) for i, name in enumerate(names)}

    if class_name == "stop":
        shape = np.column_stack(
            [columns["rms"], columns["high"], columns["flux"], columns["periodicity"]]
        )
        event = (
            1.10 * columns["flux"]
            + 0.75 * np.maximum(np.diff(columns["rms"], prepend=columns["rms"][0]), 0.0)
            + 0.55 * columns["high"]
            - 0.25 * columns["periodicity"]
        )
    elif class_name == "affricate":
        shape = np.column_stack(
            [columns["high"], columns["mid"], columns["centroid"], columns["flatness"], columns["zcr"]]
        )
        event = (
            0.85 * columns["high"]
            + 0.55 * columns["zcr"]
            + 0.45 * columns["flatness"]
            + 0.55 * columns["flux"]
        )
    elif class_name == "fricative":
        shape = np.column_stack(
            [columns["high"], columns["mid"], columns["centroid"], columns["flatness"], columns["zcr"]]
        )
        event = (
            0.95 * columns["high"]
            + 0.65 * columns["zcr"]
            + 0.45 * columns["flatness"]
            + 0.20 * columns["rms"]
        )
    else:
        shape = np.column_stack(
            [columns["low"], columns["rms"], columns["periodicity"], columns["flatness"]]
        )
        event = (
            0.80 * columns["low"]
            + 0.65 * columns["rms"]
            + 0.95 * columns["periodicity"]
            - 0.30 * columns["flatness"]
        )

    return _robust_z(shape), _robust_z(event)


def _lag_overlap(
    left: np.ndarray,
    right: np.ndarray,
    lag_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lag_frames >= 0:
        length = min(len(left), len(right) - lag_frames)
        if length <= 0:
            return left[:0], right[:0]
        return left[:length], right[lag_frames : lag_frames + length]
    shift = -lag_frames
    length = min(len(left) - shift, len(right))
    if length <= 0:
        return left[:0], right[:0]
    return left[shift : shift + length], right[:length]


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.size == 0:
        return -1.0
    a = left.reshape(-1).astype(np.float64, copy=False)
    b = right.reshape(-1).astype(np.float64, copy=False)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom < 1e-12:
        return -1.0
    return float(np.dot(a, b) / denom)


def _best_lag(left: np.ndarray, right: np.ndarray) -> LagEstimate:
    length = min(len(left), len(right))
    left = left[:length]
    right = right[:length]
    max_lag_frames = max(1, int(round(MAX_LAG_MS / HOP_MS)))
    min_overlap_frames = max(8, int(round(MIN_OVERLAP_MS / HOP_MS)))

    scored: list[tuple[float, int]] = []
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        a, b = _lag_overlap(left, right, lag)
        if len(a) < min_overlap_frames:
            continue
        similarity = _similarity(a, b)
        lag_ms = lag * HOP_MS
        timing_penalty = 0.07 * (lag_ms / MAX_LAG_MS) ** 2
        overlap_penalty = 0.05 * (1.0 - len(a) / max(length, 1))
        scored.append((similarity - timing_penalty - overlap_penalty, lag))

    if not scored:
        raise EarlyRepeatAlignmentError("no legal early-repeat lag")
    scored.sort(reverse=True)
    best_score, best_lag = scored[0]
    separated = [
        item for item in scored[1:] if abs(item[1] - best_lag) >= max(2, int(round(20.0 / HOP_MS)))
    ]
    second_best = separated[0][0] if separated else scored[min(1, len(scored) - 1)][0]
    lag_ms = float(best_lag * HOP_MS)
    edge_hit = abs(abs(lag_ms) - MAX_LAG_MS) <= _EDGE_MARGIN_MS
    return LagEstimate(
        lag_ms=lag_ms,
        score=float(best_score),
        score_margin=float(best_score - second_best),
        edge_hit=edge_hit,
    )


def _qc(shape: LagEstimate, event: LagEstimate) -> tuple[float, float, str]:
    disagreement = abs(shape.lag_ms - event.lag_ms)
    consensus = float((shape.lag_ms + event.lag_ms) / 2.0)
    if shape.edge_hit or event.edge_hit:
        return disagreement, consensus, "boundary_review"
    if shape.score < _SHAPE_SCORE_LOW or event.score < _EVENT_SCORE_LOW:
        return disagreement, consensus, "low_similarity"
    if disagreement > _CONSENSUS_REVIEW_MS:
        return disagreement, consensus, "lag_disagreement"
    if min(shape.score_margin, event.score_margin) < 0.010:
        return disagreement, consensus, "ambiguous_peak"
    return disagreement, consensus, "usable"


def _prompt_alignment(
    session_dir: Path,
    index: int,
    prompt: dict[str, object],
    recording: dict[str, object],
) -> EarlyRepeatAlignment:
    tokens = prompt.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        tokens = str(prompt.get("spoken_pattern", "")).split()
    syllable = str(prompt.get("syllable", ""))
    target_indices = [i for i, token in enumerate(tokens) if str(token) == syllable]
    if len(target_indices) != 2:
        raise EarlyRepeatAlignmentError("early repeat alignment requires exactly two targets")

    wav_name = str(recording["wav"])
    wav_path = session_dir / "recordings" / wav_name
    try:
        samples, sample_rate = read_wav(wav_path)
    except AudioReadError as exc:
        raise EarlyRepeatAlignmentError(str(exc)) from exc

    sequence = detect_beep_sequence(samples, sample_rate, len(tokens))
    cleaned = _suppress_beeps(samples, sample_rate, sequence.events)
    first_cue = sequence.events[target_indices[0] + 1].time_ms
    second_cue = sequence.events[target_indices[1] + 1].time_ms

    base_unit = str(prompt.get("base_unit", ""))
    class_name = _ONSET_CLASS.get(base_unit, "other")
    first_shape, first_event = _frame_tracks(cleaned, sample_rate, first_cue, class_name)
    second_shape, second_event = _frame_tracks(cleaned, sample_rate, second_cue, class_name)

    shape = _best_lag(first_shape, second_shape)
    event = _best_lag(first_event, second_event)
    disagreement, consensus, qc = _qc(shape, event)

    return EarlyRepeatAlignment(
        prompt_index=index,
        base_unit=base_unit,
        class_name=class_name,
        context_family=str(prompt.get("context_family", "")),
        syllable=syllable,
        wav=wav_name,
        occurrence_1_cue_ms=first_cue,
        occurrence_2_cue_ms=second_cue,
        shape=shape,
        event=event,
        lag_disagreement_ms=disagreement,
        consensus_lag_ms=consensus,
        qc=qc,
    )


def analyze_early_repeat_alignment(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    protocol = session["protocol"]
    recordings = session["recordings"]
    if not isinstance(protocol, dict) or not isinstance(recordings, dict):
        raise ValueError("invalid calibration session")
    prompts = protocol.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("calibration protocol is missing prompts")

    rows: list[EarlyRepeatAlignment] = []
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
        except EarlyRepeatAlignmentError as exc:
            failed.append({"prompt_index": index, "reason": str(exc)})

    qc_counts: dict[str, int] = {}
    for row in rows:
        qc_counts[row.qc] = qc_counts.get(row.qc, 0) + 1

    shape_scores = [row.shape.score for row in rows]
    event_scores = [row.event.score for row in rows]
    disagreements = [row.lag_disagreement_ms for row in rows]
    consensus = [row.consensus_lag_ms for row in rows]

    payload: dict[str, object] = {
        "analysis": "calibration_early_repeat_alignment",
        "version": "0.3.2",
        "session_id": session["session_id"],
        "session_dir": str(session_dir),
        "development_note": (
            "This session has been used during method development. Results are diagnostic, "
            "not an independent confirmatory validation of the frozen method."
        ),
        "window": {
            "start_ms_after_cue": WINDOW_START_MS,
            "end_ms_after_cue": WINDOW_END_MS,
            "frame_ms": FRAME_MS,
            "hop_ms": HOP_MS,
            "max_lag_ms": MAX_LAG_MS,
        },
        "summary": {
            "recordings": len(recordings),
            "aligned": len(rows),
            "failed": len(failed),
            "median_shape_score": None if not shape_scores else round(float(np.median(shape_scores)), 6),
            "median_event_score": None if not event_scores else round(float(np.median(event_scores)), 6),
            "p90_lag_disagreement_ms": None if not disagreements else round(float(np.percentile(disagreements, 90.0)), 3),
            "median_consensus_lag_ms": None if not consensus else round(float(np.median(consensus)), 3),
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
                "occurrence_1_cue_ms": round(row.occurrence_1_cue_ms, 3),
                "occurrence_2_cue_ms": round(row.occurrence_2_cue_ms, 3),
                "shape_lag_ms": round(row.shape.lag_ms, 3),
                "shape_score": round(row.shape.score, 6),
                "shape_score_margin": round(row.shape.score_margin, 6),
                "shape_edge_hit": row.shape.edge_hit,
                "event_lag_ms": round(row.event.lag_ms, 3),
                "event_score": round(row.event.score, 6),
                "event_score_margin": round(row.event.score_margin, 6),
                "event_edge_hit": row.event.edge_hit,
                "lag_disagreement_ms": round(row.lag_disagreement_ms, 3),
                "consensus_lag_ms": round(row.consensus_lag_ms, 3),
                "qc": row.qc,
            }
            for row in rows
        ],
        "failed": failed,
    }

    output_dir = session_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "calibration_early_repeat_alignment_v0.3.2.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
