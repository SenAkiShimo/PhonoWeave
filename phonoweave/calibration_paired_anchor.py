from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, read_wav
from .calibration_beep_alignment import detect_beep_sequence
from .calibration_early_repeat_alignment import (
    FRAME_MS,
    HOP_MS,
    WINDOW_START_MS,
    _ONSET_CLASS,
    _best_lag,
    _frame_tracks,
    _suppress_beeps,
)
from .calibration_io import load_calibration_session


_REPEAT_ANCHOR_REVIEW_MS = 52.0
_BOUNDARY_MARGIN_MS = 18.0
_MIN_EVENT_SUPPORT = -0.35


@dataclass(frozen=True)
class PairedAnchor:
    prompt_index: int
    base_unit: str
    class_name: str
    context_family: str
    syllable: str
    wav: str
    consensus_lag_ms: float
    alignment_qc: str
    anchor_type: str
    shared_anchor_ms_after_cue: float
    occurrence_1_anchor_ms_after_cue: float
    occurrence_2_anchor_ms_after_cue: float
    repeat_anchor_spread_ms: float
    event_support: float
    boundary_distance_ms: float
    qc: str


class PairedAnchorError(RuntimeError):
    pass


def _first_sustained(values: np.ndarray, threshold: float, frames: int) -> int | None:
    count = 0
    for index, value in enumerate(values):
        if value >= threshold:
            count += 1
            if count >= frames:
                return index - frames + 1
        else:
            count = 0
    return None


def _local_peak(values: np.ndarray, start: int, stop: int) -> int:
    start = max(0, start)
    stop = min(len(values), stop)
    if stop <= start:
        return max(0, min(len(values) - 1, start))
    return start + int(np.argmax(values[start:stop]))


def _anchor_index(values: np.ndarray, class_name: str) -> int:
    if not len(values):
        raise PairedAnchorError("empty paired event track")

    if class_name == "stop":
        threshold = float(np.percentile(values, 72.0))
        candidates = np.flatnonzero(values >= threshold)
        if not len(candidates):
            return int(np.argmax(values))
        best = int(candidates[0])
        for candidate in candidates:
            local = _local_peak(values, int(candidate), int(candidate) + 5)
            if values[local] >= threshold:
                best = local
                break
        return best

    if class_name in {"affricate", "fricative"}:
        threshold = float(np.percentile(values, 62.0))
        found = _first_sustained(values, threshold, 3)
        if found is not None:
            return found
        return int(np.argmax(values))

    threshold = float(np.percentile(values, 58.0))
    found = _first_sustained(values, threshold, 3)
    if found is not None:
        return found
    return int(np.argmax(values))


def _aligned_overlap(
    first: np.ndarray,
    second: np.ndarray,
    lag_frames: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if lag_frames >= 0:
        length = min(len(first), len(second) - lag_frames)
        if length <= 0:
            raise PairedAnchorError("paired alignment has no overlap")
        return first[:length], second[lag_frames : lag_frames + length], 0, lag_frames

    shift = -lag_frames
    length = min(len(first) - shift, len(second))
    if length <= 0:
        raise PairedAnchorError("paired alignment has no overlap")
    return first[shift : shift + length], second[:length], shift, 0


def _anchor_type(class_name: str) -> str:
    if class_name == "stop":
        return "paired_release_anchor"
    if class_name == "affricate":
        return "paired_frication_release_anchor"
    if class_name == "fricative":
        return "paired_frication_onset_anchor"
    if class_name == "sonorant":
        return "paired_sonorant_onset_anchor"
    return "paired_acoustic_anchor"


def _alignment_qc(shape_score: float, event_score: float, disagreement_ms: float, edge_hit: bool) -> str:
    if edge_hit:
        return "boundary_review"
    if shape_score < 0.12 or event_score < 0.10:
        return "low_similarity"
    if disagreement_ms > 52.0:
        return "lag_disagreement"
    return "usable"


def _paired_qc(
    alignment_qc: str,
    spread_ms: float,
    event_support: float,
    boundary_distance_ms: float,
) -> str:
    if alignment_qc != "usable":
        return "alignment_review"
    if boundary_distance_ms < _BOUNDARY_MARGIN_MS:
        return "boundary_review"
    if spread_ms > _REPEAT_ANCHOR_REVIEW_MS:
        return "repeat_anchor_review"
    if event_support < _MIN_EVENT_SUPPORT:
        return "weak_event"
    return "usable"


def _prompt_anchor(
    session_dir: Path,
    index: int,
    prompt: dict[str, object],
    recording: dict[str, object],
) -> PairedAnchor:
    tokens = prompt.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        tokens = str(prompt.get("spoken_pattern", "")).split()
    syllable = str(prompt.get("syllable", ""))
    target_indices = [i for i, token in enumerate(tokens) if str(token) == syllable]
    if len(target_indices) != 2:
        raise PairedAnchorError("paired anchor requires exactly two target occurrences")

    wav_name = str(recording["wav"])
    wav_path = session_dir / "recordings" / wav_name
    try:
        samples, sample_rate = read_wav(wav_path)
    except AudioReadError as exc:
        raise PairedAnchorError(str(exc)) from exc

    sequence = detect_beep_sequence(samples, sample_rate, len(tokens))
    cleaned = _suppress_beeps(samples, sample_rate, sequence.events)
    first_cue = sequence.events[target_indices[0] + 1].time_ms
    second_cue = sequence.events[target_indices[1] + 1].time_ms

    base_unit = str(prompt.get("base_unit", ""))
    class_name = _ONSET_CLASS.get(base_unit, "other")
    first_shape, first_event = _frame_tracks(cleaned, sample_rate, first_cue, class_name)
    second_shape, second_event = _frame_tracks(cleaned, sample_rate, second_cue, class_name)

    shape_lag = _best_lag(first_shape, second_shape)
    event_lag = _best_lag(first_event, second_event)
    disagreement_ms = abs(shape_lag.lag_ms - event_lag.lag_ms)
    consensus_lag_ms = float((shape_lag.lag_ms + event_lag.lag_ms) / 2.0)
    consensus_frames = int(round(consensus_lag_ms / HOP_MS))
    alignment_qc = _alignment_qc(
        shape_lag.score,
        event_lag.score,
        disagreement_ms,
        shape_lag.edge_hit or event_lag.edge_hit,
    )

    first_overlap, second_overlap, first_start, second_start = _aligned_overlap(
        first_event,
        second_event,
        consensus_frames,
    )
    pooled = 0.5 * (first_overlap + second_overlap)
    shared_local_index = _anchor_index(pooled, class_name)

    first_anchor_index = first_start + shared_local_index
    second_anchor_index = second_start + shared_local_index
    search_radius = max(2, int(round(28.0 / HOP_MS)))
    first_individual = _local_peak(
        first_event,
        first_anchor_index - search_radius,
        first_anchor_index + search_radius + 1,
    )
    second_individual = _local_peak(
        second_event,
        second_anchor_index - search_radius,
        second_anchor_index + search_radius + 1,
    )

    frame_center_ms = FRAME_MS / 2.0
    first_latency = WINDOW_START_MS + frame_center_ms + first_individual * HOP_MS
    second_latency = WINDOW_START_MS + frame_center_ms + second_individual * HOP_MS
    shared_first_latency = WINDOW_START_MS + frame_center_ms + first_anchor_index * HOP_MS
    shared_second_latency = WINDOW_START_MS + frame_center_ms + second_anchor_index * HOP_MS
    shared_latency = float((shared_first_latency + shared_second_latency) / 2.0)
    repeat_spread = abs(first_latency - second_latency)

    support = float(min(first_event[first_individual], second_event[second_individual]))
    window_end_latency = WINDOW_START_MS + frame_center_ms + (len(first_event) - 1) * HOP_MS
    boundary_distance = min(
        shared_latency - (WINDOW_START_MS + frame_center_ms),
        window_end_latency - shared_latency,
    )
    qc = _paired_qc(alignment_qc, repeat_spread, support, boundary_distance)

    return PairedAnchor(
        prompt_index=index,
        base_unit=base_unit,
        class_name=class_name,
        context_family=str(prompt.get("context_family", "")),
        syllable=syllable,
        wav=wav_name,
        consensus_lag_ms=consensus_lag_ms,
        alignment_qc=alignment_qc,
        anchor_type=_anchor_type(class_name),
        shared_anchor_ms_after_cue=shared_latency,
        occurrence_1_anchor_ms_after_cue=float(first_latency),
        occurrence_2_anchor_ms_after_cue=float(second_latency),
        repeat_anchor_spread_ms=float(repeat_spread),
        event_support=support,
        boundary_distance_ms=float(boundary_distance),
        qc=qc,
    )


def analyze_paired_anchors(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    protocol = session["protocol"]
    recordings = session["recordings"]
    if not isinstance(protocol, dict) or not isinstance(recordings, dict):
        raise ValueError("invalid calibration session")
    prompts = protocol.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("calibration protocol is missing prompts")

    rows: list[PairedAnchor] = []
    failed: list[dict[str, object]] = []
    for index in sorted(recordings):
        if index < 0 or index >= len(prompts):
            continue
        prompt = prompts[index]
        recording = recordings[index]
        if not isinstance(prompt, dict) or not isinstance(recording, dict):
            continue
        try:
            rows.append(_prompt_anchor(session_dir, index, prompt, recording))
        except PairedAnchorError as exc:
            failed.append({"prompt_index": index, "reason": str(exc)})

    qc_counts: dict[str, int] = {}
    for row in rows:
        qc_counts[row.qc] = qc_counts.get(row.qc, 0) + 1
    spreads = [row.repeat_anchor_spread_ms for row in rows]
    anchors = [row.shared_anchor_ms_after_cue for row in rows]
    usable_anchors = [row.shared_anchor_ms_after_cue for row in rows if row.qc == "usable"]

    payload: dict[str, object] = {
        "analysis": "calibration_paired_anchor",
        "version": "0.3.3",
        "session_id": session["session_id"],
        "session_dir": str(session_dir),
        "development_note": (
            "This session has been used during method development. Paired anchors are "
            "development diagnostics, not independent confirmatory validation."
        ),
        "anchor_definition": (
            "A repeat-supported synthesis-analysis anchor used for consistent acoustic slicing; "
            "it is not asserted to be a phonological or manually verified phonetic boundary."
        ),
        "summary": {
            "recordings": len(recordings),
            "anchored": len(rows),
            "failed": len(failed),
            "median_repeat_anchor_spread_ms": None if not spreads else round(float(np.median(spreads)), 3),
            "p90_repeat_anchor_spread_ms": None if not spreads else round(float(np.percentile(spreads, 90.0)), 3),
            "median_shared_anchor_ms_after_cue": None if not anchors else round(float(np.median(anchors)), 3),
            "median_usable_anchor_ms_after_cue": None if not usable_anchors else round(float(np.median(usable_anchors)), 3),
            "qc_counts": qc_counts,
        },
        "anchors": [
            {
                "prompt_index": row.prompt_index,
                "base_unit": row.base_unit,
                "class_name": row.class_name,
                "context_family": row.context_family,
                "syllable": row.syllable,
                "wav": row.wav,
                "consensus_lag_ms": round(row.consensus_lag_ms, 3),
                "alignment_qc": row.alignment_qc,
                "anchor_type": row.anchor_type,
                "shared_anchor_ms_after_cue": round(row.shared_anchor_ms_after_cue, 3),
                "occurrence_1_anchor_ms_after_cue": round(row.occurrence_1_anchor_ms_after_cue, 3),
                "occurrence_2_anchor_ms_after_cue": round(row.occurrence_2_anchor_ms_after_cue, 3),
                "repeat_anchor_spread_ms": round(row.repeat_anchor_spread_ms, 3),
                "event_support": round(row.event_support, 6),
                "boundary_distance_ms": round(row.boundary_distance_ms, 3),
                "qc": row.qc,
            }
            for row in rows
        ],
        "failed": failed,
    }

    output_dir = session_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "calibration_paired_anchor_v0.3.3.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
