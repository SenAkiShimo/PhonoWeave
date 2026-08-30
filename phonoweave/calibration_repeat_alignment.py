from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, read_wav
from .calibration_beep_alignment import BeepEvent, detect_beep_sequence
from .calibration_io import load_calibration_session


WINDOW_START_MS = 90.0
WINDOW_END_MS = 650.0
FRAME_MS = 24.0
HOP_MS = 4.0
MAX_LAG_MS = 300.0
MIN_OVERLAP_MS = 260.0

_BEEP_SUBTRACT_START_MS = -12.0
_BEEP_SUBTRACT_END_MS = 110.0

_SPECTRAL_SCORE_LOW = 0.22
_ACTIVITY_SCORE_LOW = 0.12
_LAG_DISAGREEMENT_REVIEW_MS = 84.0
_EDGE_MARGIN_FRAMES = 2


@dataclass(frozen=True)
class LagEstimate:
    lag_ms: float
    score: float
    second_best_score: float
    score_margin: float
    edge_hit: bool


@dataclass(frozen=True)
class RepeatAlignment:
    prompt_index: int
    base_unit: str
    class_name: str
    context_family: str
    syllable: str
    wav: str
    occurrence_1_cue_ms: float
    occurrence_2_cue_ms: float
    spectral: LagEstimate
    activity: LagEstimate
    lag_disagreement_ms: float
    qc: str


class RepeatAlignmentError(RuntimeError):
    pass


_ONSET_CLASS = {
    **{item: "stop" for item in ("b", "p", "d", "t", "g", "k")},
    **{item: "affricate" for item in ("zh", "ch", "z", "c", "j", "q")},
    **{item: "fricative" for item in ("f", "h", "x", "sh", "s")},
    **{item: "nasal" for item in ("m", "n")},
    "l": "lateral",
    "r": "rhotic",
}


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


def _robust_standardize(values: np.ndarray) -> np.ndarray:
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


def _feature_tracks(
    samples: np.ndarray,
    sample_rate: int,
    cue_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    start_ms = cue_ms + WINDOW_START_MS
    end_ms = cue_ms + WINDOW_END_MS
    start = max(0, int(round(start_ms * sample_rate / 1000.0)))
    end = min(len(samples), int(round(end_ms * sample_rate / 1000.0)))
    region = samples[start:end]

    frame = max(128, int(round(FRAME_MS * sample_rate / 1000.0)))
    hop = max(1, int(round(HOP_MS * sample_rate / 1000.0)))
    if len(region) < frame:
        raise RepeatAlignmentError("repeat-alignment window is too short")
    positions = list(range(0, len(region) - frame + 1, hop))
    if len(positions) < 40:
        raise RepeatAlignmentError("repeat-alignment window has too few frames")

    nyquist_limit = min(9000.0, 0.45 * sample_rate)
    if nyquist_limit <= 700.0:
        raise RepeatAlignmentError("sample rate is too low for repeat alignment")
    band_edges = np.geomspace(260.0, nyquist_limit, 13)

    spectral_rows: list[list[float]] = []
    rms_db: list[float] = []
    periodicity: list[float] = []
    zcr: list[float] = []
    previous_log_spectrum: np.ndarray | None = None
    flux: list[float] = []

    for position in positions:
        chunk = region[position : position + frame].astype(np.float64, copy=False)
        chunk = chunk - float(np.mean(chunk))
        windowed = chunk * np.hanning(len(chunk))
        power = np.abs(np.fft.rfft(windowed)) ** 2 + 1e-18
        freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sample_rate)

        row: list[float] = []
        for left, right in zip(band_edges[:-1], band_edges[1:], strict=True):
            mask = (freqs >= left) & (freqs < right)
            energy = float(np.sum(power[mask])) if np.any(mask) else 1e-18
            row.append(math.log(energy + 1e-18))
        spectral_rows.append(row)

        rms = float(np.sqrt(np.mean(chunk * chunk)) + 1e-12)
        rms_db.append(20.0 * math.log10(rms))
        periodicity.append(_periodicity(chunk, sample_rate))
        zcr.append(float(np.mean((chunk[:-1] >= 0.0) != (chunk[1:] >= 0.0))))

        valid = (freqs >= 260.0) & (freqs <= nyquist_limit)
        log_spectrum = np.log(power[valid])
        if previous_log_spectrum is None or len(log_spectrum) != len(previous_log_spectrum):
            flux.append(0.0)
        else:
            delta = log_spectrum - previous_log_spectrum
            flux.append(float(np.sqrt(np.mean(delta * delta))))
        previous_log_spectrum = log_spectrum

    spectral = _robust_standardize(np.asarray(spectral_rows, dtype=np.float64))
    rms_z = _robust_standardize(np.asarray(rms_db, dtype=np.float64))
    flux_z = _robust_standardize(np.asarray(flux, dtype=np.float64))
    zcr_z = _robust_standardize(np.asarray(zcr, dtype=np.float64))
    periodicity_array = np.asarray(periodicity, dtype=np.float64)

    activity = (
        0.95 * rms_z
        + 0.60 * flux_z
        + 0.30 * zcr_z
        + 0.55 * _robust_standardize(periodicity_array)
    )
    activity = _robust_standardize(activity)
    return spectral, activity


def _normalized_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or left.shape != right.shape:
        return -1.0
    a = left.reshape(-1).astype(np.float64, copy=False)
    b = right.reshape(-1).astype(np.float64, copy=False)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom < 1e-12:
        return -1.0
    return float(np.dot(a, b) / denom)


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


def _best_lag(
    left: np.ndarray,
    right: np.ndarray,
    hop_ms: float = HOP_MS,
    max_lag_ms: float = MAX_LAG_MS,
    min_overlap_ms: float = MIN_OVERLAP_MS,
) -> LagEstimate:
    if len(left) != len(right):
        length = min(len(left), len(right))
        left = left[:length]
        right = right[:length]
    max_lag_frames = max(1, int(round(max_lag_ms / hop_ms)))
    min_overlap_frames = max(8, int(round(min_overlap_ms / hop_ms)))

    scored: list[tuple[float, int]] = []
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        a, b = _lag_overlap(left, right, lag)
        if len(a) < min_overlap_frames:
            continue
        similarity = _normalized_similarity(a, b)
        overlap_fraction = len(a) / max(len(left), 1)
        score = similarity - 0.08 * (1.0 - overlap_fraction)
        scored.append((score, lag))

    if not scored:
        raise RepeatAlignmentError("no legal lag overlap")
    scored.sort(reverse=True)
    best_score, best_lag = scored[0]

    separated = [
        item
        for item in scored[1:]
        if abs(item[1] - best_lag) >= max(2, int(round(24.0 / hop_ms)))
    ]
    second_best = separated[0][0] if separated else scored[min(1, len(scored) - 1)][0]
    edge_hit = abs(best_lag) >= max_lag_frames - _EDGE_MARGIN_FRAMES
    return LagEstimate(
        lag_ms=float(best_lag * hop_ms),
        score=float(best_score),
        second_best_score=float(second_best),
        score_margin=float(best_score - second_best),
        edge_hit=edge_hit,
    )


def _qc(spectral: LagEstimate, activity: LagEstimate) -> tuple[float, str]:
    disagreement = abs(spectral.lag_ms - activity.lag_ms)
    if spectral.edge_hit or activity.edge_hit:
        return disagreement, "boundary_review"
    if spectral.score < _SPECTRAL_SCORE_LOW or activity.score < _ACTIVITY_SCORE_LOW:
        return disagreement, "low_similarity"
    if disagreement > _LAG_DISAGREEMENT_REVIEW_MS:
        return disagreement, "lag_disagreement"
    if spectral.score_margin < 0.015:
        return disagreement, "ambiguous_peak"
    return disagreement, "usable"


def _prompt_alignment(
    session_dir: Path,
    index: int,
    prompt: dict[str, object],
    recording: dict[str, object],
) -> RepeatAlignment:
    tokens = prompt.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        tokens = str(prompt.get("spoken_pattern", "")).split()
    syllable = str(prompt.get("syllable", ""))
    target_indices = [i for i, token in enumerate(tokens) if str(token) == syllable]
    if len(target_indices) != 2:
        raise RepeatAlignmentError("repeat alignment requires exactly two target occurrences")

    wav_name = str(recording["wav"])
    wav_path = session_dir / "recordings" / wav_name
    try:
        samples, sample_rate = read_wav(wav_path)
    except AudioReadError as exc:
        raise RepeatAlignmentError(str(exc)) from exc

    sequence = detect_beep_sequence(samples, sample_rate, len(tokens))
    cleaned = _suppress_beeps(samples, sample_rate, sequence.events)

    first_cue = sequence.events[target_indices[0] + 1].time_ms
    second_cue = sequence.events[target_indices[1] + 1].time_ms
    first_spectral, first_activity = _feature_tracks(cleaned, sample_rate, first_cue)
    second_spectral, second_activity = _feature_tracks(cleaned, sample_rate, second_cue)

    spectral = _best_lag(first_spectral, second_spectral)
    activity = _best_lag(first_activity, second_activity)
    disagreement, qc = _qc(spectral, activity)

    base_unit = str(prompt.get("base_unit", ""))
    return RepeatAlignment(
        prompt_index=index,
        base_unit=base_unit,
        class_name=_ONSET_CLASS.get(base_unit, "other"),
        context_family=str(prompt.get("context_family", "")),
        syllable=syllable,
        wav=wav_name,
        occurrence_1_cue_ms=first_cue,
        occurrence_2_cue_ms=second_cue,
        spectral=spectral,
        activity=activity,
        lag_disagreement_ms=disagreement,
        qc=qc,
    )


def analyze_repeat_alignment(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    protocol = session["protocol"]
    recordings = session["recordings"]
    if not isinstance(protocol, dict) or not isinstance(recordings, dict):
        raise ValueError("invalid calibration session")
    prompts = protocol.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("calibration protocol is missing prompts")

    rows: list[RepeatAlignment] = []
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
        except RepeatAlignmentError as exc:
            failed.append({"prompt_index": index, "reason": str(exc)})

    qc_counts: dict[str, int] = {}
    for row in rows:
        qc_counts[row.qc] = qc_counts.get(row.qc, 0) + 1

    spectral_scores = [row.spectral.score for row in rows]
    activity_scores = [row.activity.score for row in rows]
    disagreements = [row.lag_disagreement_ms for row in rows]
    lags = [row.spectral.lag_ms for row in rows]

    payload: dict[str, object] = {
        "analysis": "calibration_repeat_alignment",
        "version": "0.3.1",
        "session_id": session["session_id"],
        "session_dir": str(session_dir),
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
            "median_spectral_score": None if not spectral_scores else round(float(np.median(spectral_scores)), 6),
            "median_activity_score": None if not activity_scores else round(float(np.median(activity_scores)), 6),
            "p90_lag_disagreement_ms": None if not disagreements else round(float(np.percentile(disagreements, 90.0)), 3),
            "median_pair_lag_ms": None if not lags else round(float(np.median(lags)), 3),
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
                "spectral_lag_ms": round(row.spectral.lag_ms, 3),
                "spectral_score": round(row.spectral.score, 6),
                "spectral_score_margin": round(row.spectral.score_margin, 6),
                "spectral_edge_hit": row.spectral.edge_hit,
                "activity_lag_ms": round(row.activity.lag_ms, 3),
                "activity_score": round(row.activity.score, 6),
                "activity_score_margin": round(row.activity.score_margin, 6),
                "activity_edge_hit": row.activity.edge_hit,
                "lag_disagreement_ms": round(row.lag_disagreement_ms, 3),
                "qc": row.qc,
            }
            for row in rows
        ],
        "failed": failed,
    }

    output_dir = session_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "calibration_repeat_alignment_v0.3.1.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
