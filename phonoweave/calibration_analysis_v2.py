from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .audio import AudioReadError, read_wav
from .calibration_analysis import (
    _EXPECTED_MANDARIN_ONSETS,
    _FEATURE_NAMES,
    _ONSET_CLASS,
    _WINDOW_MS,
    _cue_suppressed,
    _detect_onset,
    _features,
    _token_expected_ms,
    BEAT_MS,
    COUNT_IN_BEATS,
    PRE_ROLL_MS,
    TEMPO_BPM,
)
from .calibration_io import load_calibration_session


_TIMING_REPEAT_REVIEW_MS = 180.0
_ALIGNMENT_LOWER_BOUND_MS = -55.0
_ALIGNMENT_UPPER_BOUND_MS = 270.0
_ALIGNMENT_BOUNDARY_MARGIN_MS = 12.0


def _extract_observations(
    session_dir: Path,
    session: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    protocol = session["protocol"]
    recordings = session["recordings"]
    if not isinstance(protocol, dict) or not isinstance(recordings, dict):
        raise ValueError("invalid calibration session")
    prompts = protocol.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("calibration protocol is missing prompts")

    observations: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, dict) or index not in recordings:
            continue
        item = recordings[index]
        if not isinstance(item, dict):
            continue
        wav_path = session_dir / "recordings" / str(item["wav"])
        try:
            samples, sample_rate = read_wav(wav_path)
        except AudioReadError as exc:
            skipped.append({"prompt_index": index, "reason": str(exc)})
            continue

        tokens = prompt.get("tokens")
        if not isinstance(tokens, list):
            tokens = str(prompt.get("spoken_pattern", "")).strip().split()
        syllable = str(prompt.get("syllable", ""))
        target_indices = [i for i, token in enumerate(tokens) if str(token) == syllable]
        if not target_indices:
            skipped.append({"prompt_index": index, "reason": "target token not found in pattern"})
            continue

        cleaned = _cue_suppressed(samples, sample_rate, len(tokens))
        detected: list[tuple[int, int, float, float, float]] = []
        for occurrence, token_index in enumerate(target_indices, start=1):
            expected_ms = _token_expected_ms(token_index)
            try:
                detected_ms, alignment_db = _detect_onset(cleaned, sample_rate, expected_ms)
            except (AudioReadError, ValueError) as exc:
                skipped.append(
                    {
                        "prompt_index": index,
                        "occurrence": occurrence,
                        "reason": str(exc),
                    }
                )
                continue
            detected.append(
                (
                    occurrence,
                    token_index,
                    expected_ms,
                    detected_ms - expected_ms,
                    alignment_db,
                )
            )

        if not detected:
            continue

        offsets = np.asarray([row[3] for row in detected], dtype=np.float64)
        shared_offset_ms = float(np.median(offsets))
        repeat_spread_ms = float(np.max(offsets) - np.min(offsets)) if len(offsets) > 1 else 0.0
        class_name = _ONSET_CLASS.get(str(prompt.get("base_unit", "")), "other")
        window_ms = _WINDOW_MS.get(class_name, 95.0)

        for occurrence, token_index, expected_ms, detected_offset_ms, alignment_db in detected:
            anchored_onset_ms = max(0.0, expected_ms + shared_offset_ms)
            start = max(0, int(round(anchored_onset_ms * sample_rate / 1000.0)))
            end = min(
                len(cleaned),
                int(round((anchored_onset_ms + window_ms) * sample_rate / 1000.0)),
            )
            try:
                feature_values = _features(cleaned[start:end], sample_rate)
            except (AudioReadError, ValueError) as exc:
                skipped.append(
                    {
                        "prompt_index": index,
                        "occurrence": occurrence,
                        "reason": str(exc),
                    }
                )
                continue

            observations.append(
                {
                    "prompt_index": index,
                    "base_unit": str(prompt.get("base_unit", "")),
                    "class_name": class_name,
                    "context_family": str(prompt.get("context_family", "")),
                    "syllable": syllable,
                    "occurrence": occurrence,
                    "token_index": token_index,
                    "expected_ms": round(expected_ms, 3),
                    "detected_offset_ms": round(detected_offset_ms, 3),
                    "alignment_offset_ms": round(shared_offset_ms, 3),
                    "timing_repeat_spread_ms": round(repeat_spread_ms, 3),
                    "alignment_db": round(alignment_db, 3),
                    "onset_ms": round(anchored_onset_ms, 3),
                    "window_ms": window_ms,
                    "features": feature_values,
                }
            )

    return observations, skipped


def _feature_matrix(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray(
        [
            [float(row["features"][name]) for name in _FEATURE_NAMES]  # type: ignore[index]
            for row in rows
        ],
        dtype=np.float64,
    )


def _timing_qc(rows: list[dict[str, object]]) -> tuple[float | None, bool]:
    if not rows:
        return None, True
    spreads = [float(row["timing_repeat_spread_ms"]) for row in rows]
    max_spread = max(spreads)
    detected_offsets = [float(row["detected_offset_ms"]) for row in rows]
    boundary_hit = any(
        offset <= _ALIGNMENT_LOWER_BOUND_MS + _ALIGNMENT_BOUNDARY_MARGIN_MS
        or offset >= _ALIGNMENT_UPPER_BOUND_MS - _ALIGNMENT_BOUNDARY_MARGIN_MS
        for offset in detected_offsets
    )
    return max_spread, max_spread > _TIMING_REPEAT_REVIEW_MS or boundary_hit


def _contrast_rows(observations: list[dict[str, object]]) -> list[dict[str, object]]:
    by_base: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in observations:
        by_base[str(row["base_unit"])].append(row)

    results: list[dict[str, object]] = []
    for base_unit in sorted(by_base):
        rows = by_base[base_unit]
        matrix = _feature_matrix(rows)
        scale = np.std(matrix, axis=0, ddof=1) if len(matrix) > 1 else np.ones(len(_FEATURE_NAMES))
        scale = np.where(scale < 1e-9, 1.0, scale)
        contexts: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            contexts[str(row["context_family"])].append(row)

        context_names = sorted(contexts)
        pairs: list[dict[str, object]] = []
        for left_index in range(len(context_names)):
            for right_index in range(left_index + 1, len(context_names)):
                left_name = context_names[left_index]
                right_name = context_names[right_index]
                left_rows = contexts[left_name]
                right_rows = contexts[right_name]
                left = _feature_matrix(left_rows)
                right = _feature_matrix(right_rows)
                effects = (np.mean(right, axis=0) - np.mean(left, axis=0)) / scale
                between = float(np.linalg.norm(effects) / math.sqrt(len(_FEATURE_NAMES)))

                repeat_distances: list[float] = []
                for group in (left, right):
                    if len(group) >= 2:
                        for a in range(len(group)):
                            for b in range(a + 1, len(group)):
                                delta = (group[a] - group[b]) / scale
                                repeat_distances.append(
                                    float(np.linalg.norm(delta) / math.sqrt(len(_FEATURE_NAMES)))
                                )
                repeat_noise = float(np.mean(repeat_distances)) if repeat_distances else None
                ratio = None if repeat_noise is None else between / max(repeat_noise, 0.15)
                max_timing_spread, timing_review = _timing_qc(left_rows + right_rows)
                min_alignment_db = min(
                    float(row["alignment_db"]) for row in left_rows + right_rows
                )

                if timing_review:
                    label = "alignment_review"
                elif ratio is not None and between >= 0.90 and ratio >= 1.80:
                    label = "stable_candidate"
                elif ratio is not None and between >= 0.60 and ratio >= 1.20:
                    label = "possible_candidate"
                else:
                    label = "overlap_or_weak"

                pairs.append(
                    {
                        "context_a": left_name,
                        "context_b": right_name,
                        "count_a": len(left_rows),
                        "count_b": len(right_rows),
                        "standardized_distance": round(between, 4),
                        "within_context_repeat_distance": None
                        if repeat_noise is None
                        else round(repeat_noise, 4),
                        "separation_to_repeat_ratio": None if ratio is None else round(ratio, 4),
                        "maximum_timing_repeat_spread_ms": None
                        if max_timing_spread is None
                        else round(max_timing_spread, 3),
                        "minimum_alignment_db_diagnostic_only": round(min_alignment_db, 3),
                        "screening_label": label,
                        "standardized_effects": {
                            name: round(float(value), 4)
                            for name, value in zip(_FEATURE_NAMES, effects, strict=True)
                        },
                    }
                )

        results.append(
            {
                "base_unit": base_unit,
                "class_name": str(rows[0]["class_name"]),
                "contexts": {
                    name: {
                        "observations": len(contexts[name]),
                        "mean_alignment_offset_ms": round(
                            float(np.mean([float(row["alignment_offset_ms"]) for row in contexts[name]])),
                            3,
                        ),
                        "timing_repeat_spread_ms": round(
                            max(float(row["timing_repeat_spread_ms"]) for row in contexts[name]),
                            3,
                        ),
                    }
                    for name in context_names
                },
                "pairwise": pairs,
            }
        )
    return results


def _observation_csv(observations: list[dict[str, object]]) -> str:
    output = io.StringIO()
    fieldnames = [
        "prompt_index",
        "base_unit",
        "class_name",
        "context_family",
        "syllable",
        "occurrence",
        "token_index",
        "expected_ms",
        "detected_offset_ms",
        "alignment_offset_ms",
        "timing_repeat_spread_ms",
        "alignment_db",
        "onset_ms",
        "window_ms",
        *_FEATURE_NAMES,
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in observations:
        flat = {key: row.get(key) for key in fieldnames if key not in _FEATURE_NAMES}
        features = row["features"]
        if isinstance(features, dict):
            for name in _FEATURE_NAMES:
                flat[name] = features.get(name)
        writer.writerow(flat)
    return output.getvalue()


def analyze_calibration_session_v2(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    observations, skipped = _extract_observations(session_dir, session)
    base_results = _contrast_rows(observations)
    tested = {str(row["base_unit"]) for row in observations}
    untested = [base for base in _EXPECTED_MANDARIN_ONSETS if base not in tested]

    timing_spreads = [float(row["timing_repeat_spread_ms"]) for row in observations]
    detected_offsets = [float(row["detected_offset_ms"]) for row in observations]
    payload: dict[str, object] = {
        "analysis": "calibration_screening",
        "version": "0.2",
        "session_id": session["session_id"],
        "session_dir": str(session_dir),
        "timing": {
            "tempo_bpm": TEMPO_BPM,
            "pre_roll_ms": PRE_ROLL_MS,
            "count_in_beats": COUNT_IN_BEATS,
            "beat_ms": BEAT_MS,
            "repeat_review_threshold_ms": _TIMING_REPEAT_REVIEW_MS,
        },
        "summary": {
            "recorded_prompts": len(session["completed_indices"]),  # type: ignore[arg-type]
            "protocol_prompts": len(session["protocol"]["prompts"]),  # type: ignore[index]
            "observations": len(observations),
            "skipped_observations": len(skipped),
            "tested_onsets": len(tested),
            "untested_onsets": untested,
            "median_detected_offset_ms": None
            if not detected_offsets
            else round(float(np.median(detected_offsets)), 3),
            "p90_timing_repeat_spread_ms": None
            if not timing_spreads
            else round(float(np.percentile(timing_spreads, 90.0)), 3),
            "maximum_timing_repeat_spread_ms": None
            if not timing_spreads
            else round(float(np.max(timing_spreads)), 3),
        },
        "bases": base_results,
        "observations": observations,
        "skipped": skipped,
        "interpretation": "screening_only_not_split_merge_decision",
    }

    output_dir = session_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "calibration_screening_v0.2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "calibration_observations_v0.2.csv").write_text(
        _observation_csv(observations),
        encoding="utf-8",
    )
    return payload
