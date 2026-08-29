from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .audio import AudioReadError, read_wav
from .calibration_io import load_calibration_session


TEMPO_BPM = 72.0
BEAT_MS = 60000.0 / TEMPO_BPM
PRE_ROLL_MS = 250.0
COUNT_IN_BEATS = 1

_FEATURE_NAMES = (
    "rms_db",
    "zero_crossing_rate",
    "centroid_hz",
    "spread_hz",
    "flatness",
    "high_band_ratio",
    "periodicity",
)

_ONSET_CLASS = {
    **{item: "stop" for item in ("b", "p", "d", "t", "g", "k")},
    **{item: "affricate" for item in ("zh", "ch", "z", "c", "j", "q")},
    **{item: "fricative" for item in ("f", "h", "x", "sh", "s")},
    **{item: "nasal" for item in ("m", "n")},
    "l": "lateral",
    "r": "rhotic",
}

_WINDOW_MS = {
    "stop": 70.0,
    "affricate": 110.0,
    "fricative": 125.0,
    "nasal": 90.0,
    "lateral": 90.0,
    "rhotic": 100.0,
}

_EXPECTED_MANDARIN_ONSETS = (
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s",
)


def _token_expected_ms(token_index: int) -> float:
    return PRE_ROLL_MS + BEAT_MS * (COUNT_IN_BEATS + token_index)


def _cue_suppressed(samples: np.ndarray, sample_rate: int, token_count: int) -> np.ndarray:
    cleaned = samples.astype(np.float64, copy=True)
    for token_index in range(token_count):
        expected_ms = _token_expected_ms(token_index)
        start = max(0, int(round((expected_ms - 15.0) * sample_rate / 1000.0)))
        end = min(len(cleaned), int(round((expected_ms + 105.0) * sample_rate / 1000.0)))
        if end - start < 32:
            continue
        local = cleaned[start:end]
        t = np.arange(start, end, dtype=np.float64) / sample_rate
        taper = np.hanning(len(local))
        for frequency in (700.0, 1000.0):
            sin = np.sin(2.0 * np.pi * frequency * t)
            cos = np.cos(2.0 * np.pi * frequency * t)
            a = float(np.dot(local, sin) / max(np.dot(sin, sin), 1e-12))
            b = float(np.dot(local, cos) / max(np.dot(cos, cos), 1e-12))
            local = local - taper * (a * sin + b * cos)
        cleaned[start:end] = local
    return cleaned


def _frame_rms_db(
    samples: np.ndarray,
    sample_rate: int,
    start_ms: float,
    end_ms: float,
    frame_ms: float = 12.0,
    hop_ms: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    start = max(0, int(round(start_ms * sample_rate / 1000.0)))
    end = min(len(samples), int(round(end_ms * sample_rate / 1000.0)))
    frame = max(32, int(round(frame_ms * sample_rate / 1000.0)))
    hop = max(1, int(round(hop_ms * sample_rate / 1000.0)))
    if end - start < frame:
        raise AudioReadError("calibration alignment window is too short")
    positions = np.arange(start, end - frame + 1, hop, dtype=np.int64)
    values = np.empty(len(positions), dtype=np.float64)
    centers = np.empty(len(positions), dtype=np.float64)
    for row, position in enumerate(positions):
        frame_samples = samples[position : position + frame]
        rms = float(np.sqrt(np.mean(frame_samples * frame_samples)) + 1e-12)
        values[row] = 20.0 * np.log10(rms)
        centers[row] = 1000.0 * (position + frame / 2.0) / sample_rate
    return centers, values


def _detect_onset(samples: np.ndarray, sample_rate: int, expected_ms: float) -> tuple[float, float]:
    centers, rms_db = _frame_rms_db(
        samples,
        sample_rate,
        max(0.0, expected_ms - 180.0),
        expected_ms + 330.0,
    )
    smooth = np.convolve(rms_db, np.ones(3, dtype=np.float64) / 3.0, mode="same")
    derivative = np.diff(smooth, prepend=smooth[0])
    baseline = float(np.percentile(smooth, 20.0))
    candidate = (
        (centers >= expected_ms - 55.0)
        & (centers <= expected_ms + 270.0)
        & (smooth >= baseline + 6.0)
        & (derivative >= 0.8)
    )
    indices = np.flatnonzero(candidate)
    if len(indices):
        chosen = int(indices[0])
    else:
        allowed = np.flatnonzero(
            (centers >= expected_ms - 55.0) & (centers <= expected_ms + 270.0)
        )
        if not len(allowed):
            raise AudioReadError("no calibration alignment interval")
        score = smooth + 1.5 * derivative - 0.010 * np.abs(centers - expected_ms)
        chosen = int(allowed[np.argmax(score[allowed])])
    onset_ms = max(0.0, float(centers[chosen] - 6.0))
    alignment_db = float(smooth[chosen] - baseline)
    return onset_ms, alignment_db


def _periodicity(samples: np.ndarray, sample_rate: int) -> float:
    centered = samples.astype(np.float64, copy=False) - float(np.mean(samples))
    energy = float(np.dot(centered, centered))
    if energy <= 1e-12:
        return 0.0
    min_lag = max(1, int(sample_rate / 500.0))
    max_lag = min(len(centered) - 2, int(sample_rate / 70.0))
    if max_lag <= min_lag:
        return 0.0
    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
        if denom > 1e-12:
            best = max(best, float(np.dot(left, right) / denom))
    return max(0.0, min(best, 1.0))


def _features(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    x = samples.astype(np.float64, copy=False)
    if len(x) < 64:
        raise AudioReadError("calibration feature window is too short")
    x = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
    zcr = float(np.mean((x[:-1] >= 0.0) != (x[1:] >= 0.0)))

    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    valid = (freqs >= 250.0) & (freqs <= min(10000.0, 0.45 * sample_rate))
    if not np.any(valid):
        raise AudioReadError("calibration feature window has no usable spectrum")
    band_power = power[valid]
    band_freqs = freqs[valid]
    total = float(np.sum(band_power))
    weights = band_power / total
    centroid = float(np.sum(band_freqs * weights))
    spread = math.sqrt(max(float(np.sum(((band_freqs - centroid) ** 2) * weights)), 0.0))
    flatness = float(np.exp(np.mean(np.log(band_power))) / np.mean(band_power))
    high_threshold = min(4000.0, 0.20 * sample_rate)
    high_ratio = float(np.sum(band_power[band_freqs >= high_threshold]) / total)

    return {
        "rms_db": 20.0 * math.log10(rms),
        "zero_crossing_rate": zcr,
        "centroid_hz": centroid,
        "spread_hz": spread,
        "flatness": flatness,
        "high_band_ratio": high_ratio,
        "periodicity": _periodicity(x, sample_rate),
    }


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
            pattern = str(prompt.get("spoken_pattern", "")).strip()
            tokens = pattern.split()
        syllable = str(prompt.get("syllable", ""))
        target_indices = [i for i, token in enumerate(tokens) if str(token) == syllable]
        if not target_indices:
            skipped.append({"prompt_index": index, "reason": "target token not found in pattern"})
            continue

        cleaned = _cue_suppressed(samples, sample_rate, len(tokens))
        class_name = _ONSET_CLASS.get(str(prompt.get("base_unit", "")), "other")
        window_ms = _WINDOW_MS.get(class_name, 95.0)
        for occurrence, token_index in enumerate(target_indices, start=1):
            expected_ms = _token_expected_ms(token_index)
            try:
                onset_ms, alignment_db = _detect_onset(cleaned, sample_rate, expected_ms)
                start = max(0, int(round(onset_ms * sample_rate / 1000.0)))
                end = min(
                    len(cleaned),
                    int(round((onset_ms + window_ms) * sample_rate / 1000.0)),
                )
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
                    "onset_ms": round(onset_ms, 3),
                    "alignment_offset_ms": round(onset_ms - expected_ms, 3),
                    "alignment_db": round(alignment_db, 3),
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
                min_alignment = min(
                    float(row["alignment_db"]) for row in left_rows + right_rows
                )
                if min_alignment < 4.0:
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
                        "minimum_alignment_db": round(min_alignment, 3),
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
                            float(
                                np.mean(
                                    [float(row["alignment_offset_ms"]) for row in contexts[name]]
                                )
                            ),
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
        "onset_ms",
        "alignment_offset_ms",
        "alignment_db",
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


def analyze_calibration_session(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    observations, skipped = _extract_observations(session_dir, session)
    base_results = _contrast_rows(observations)
    tested = {str(row["base_unit"]) for row in observations}
    untested = [base for base in _EXPECTED_MANDARIN_ONSETS if base not in tested]

    payload: dict[str, object] = {
        "analysis": "calibration_screening",
        "version": "0.1",
        "session_id": session["session_id"],
        "session_dir": str(session_dir),
        "timing": {
            "tempo_bpm": TEMPO_BPM,
            "pre_roll_ms": PRE_ROLL_MS,
            "count_in_beats": COUNT_IN_BEATS,
            "beat_ms": BEAT_MS,
        },
        "summary": {
            "recorded_prompts": len(session["completed_indices"]),  # type: ignore[arg-type]
            "protocol_prompts": len(session["protocol"]["prompts"]),  # type: ignore[index]
            "observations": len(observations),
            "skipped_observations": len(skipped),
            "tested_onsets": len(tested),
            "untested_onsets": untested,
        },
        "bases": base_results,
        "observations": observations,
        "skipped": skipped,
        "interpretation": "screening_only_not_split_merge_decision",
    }

    output_dir = session_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "calibration_screening_v0.1.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "calibration_observations_v0.1.csv").write_text(
        _observation_csv(observations),
        encoding="utf-8",
    )
    return payload
