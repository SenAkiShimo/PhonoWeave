"""Frozen early-frication shadow audit across synthetic noise realizations.

The baseline paired CV detector determines acceptance. Early-frication
backtrace only proposes an alternative onset for already paired candidates.
All manual anchors are evaluation-only; no detector is fitted here.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from .audio import read_wav
from .calibration_cv_transition import cv_tracks
from .calibration_dual_labels import load_dual_labels
from .calibration_early_frication import trace_early_frication
from .calibration_early_frication_cli import TARGETS
from .calibration_excess_fallback import detect_excess_fallback
from .calibration_io import load_calibration_session
from .calibration_multiseed_noise import (
    NOISE_TYPES, SEEDS, SNR_DB, _recording_seed, add_calibrated_noise,
)
from .calibration_negative_controls import CATEGORIES, CORPUS_DIR
from .calibration_transition_onset import acoustic_tracks, nominal_target_cue_ms

REPORT_NAME = "calibration_multiseed_early_frication_v2.8.json"
ERROR_LIMIT_MS = 40.0


def summarize_onsets(rows: list[dict], method: str) -> dict:
    """Matched-only metrics with explicit eligibility and coverage."""
    if method not in ("baseline", "shadow"):
        raise ValueError("unsupported method")
    eligible = [r for r in rows if r["manual_onset_ms"] is not None]
    matched = [
        r for r in eligible if r[f"{method}_error_ms"] is not None
    ]
    errors = np.asarray([r[f"{method}_error_ms"] for r in matched], dtype=float)
    ae = np.abs(errors)
    return {
        "eligible": len(eligible),
        "matched": len(matched),
        "coverage": round(len(matched) / len(eligible), 4) if eligible else None,
        "conditional_mae_ms": round(float(np.mean(ae)), 3) if ae.size else None,
        "conditional_p90_ms": round(float(np.percentile(ae, 90)), 3) if ae.size else None,
        "median_signed_error_ms": round(float(np.median(errors)), 3) if ae.size else None,
        "errors_over_40ms": int(np.sum(ae > ERROR_LIMIT_MS)),
    }


def compare_paired_rows(rows: list[dict]) -> dict:
    """Pair by utterance, preventing mean-only evaluation of early shifts."""
    common = [
        r for r in rows
        if r["manual_onset_ms"] is not None
        and r["baseline_error_ms"] is not None
        and r["shadow_error_ms"] is not None
    ]
    improvements = [
        abs(r["baseline_error_ms"]) - abs(r["shadow_error_ms"])
        for r in common
    ]
    eps = 1e-3
    shifts = [
        r["backtrace_ms"] for r in common if r["backtrace_ms"] is not None
    ]
    return {
        "common_pairs": len(common),
        "improved": sum(v > eps for v in improvements),
        "worsened": sum(v < -eps for v in improvements),
        "unchanged": sum(abs(v) <= eps for v in improvements),
        "median_error_improvement_ms": (
            round(float(np.median(improvements)), 3) if improvements else None
        ),
        "median_backtrace_ms": (
            round(float(np.median(shifts)), 3) if shifts else None
        ),
        "max_backtrace_ms": (
            round(float(np.max(shifts)), 3) if shifts else None
        ),
    }


def audit_early_frication_multiseed(
    session_dir: Path,
    *, seeds: tuple[int, ...] = SEEDS,
) -> dict:
    """Apply fixed baseline/backtrace to clean and 24 deterministic noise trials."""
    if not seeds or any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("seeds must be a nonempty tuple of nonnegative integers")
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    labels = load_dual_labels(session_dir)["labels"]
    prepared = []
    for index, base, context in TARGETS:
        prompt = session["protocol"]["prompts"][index]
        if (prompt["base_unit"], prompt["context_family"]) != (base, context):
            raise ValueError(f"protocol mismatch at index {index}")
        recording = session["recordings"].get(index)
        if recording is None:
            raise ValueError(f"recording missing at index {index}")
        filename = str(recording["wav"])
        audio, rate = read_wav(session_dir / "recordings" / filename)
        tokens = prompt.get("tokens") or str(prompt["spoken_pattern"]).split()
        indices = [i for i, token in enumerate(tokens) if token == prompt["syllable"]]
        if len(indices) != 2:
            raise ValueError("unexpected number of target tokens")
        references = []
        for occurrence in (1, 2):
            entry = (
                labels.get(str(index), {}).get("occurrences", {})
                .get(str(occurrence), {}).get("consonant_onset")
            )
            eligible = (
                isinstance(entry, dict)
                and entry.get("status") == "ok"
                and entry.get("source") == "manual_v0.3"
                and not entry.get("outside_expected_slot", False)
            )
            references.append(float(entry["absolute_anchor_ms"]) if eligible else None)
        prepared.append((index, base, context, filename, audio, rate, indices, references))

    negatives = []
    folder = (session_dir.parent / CORPUS_DIR).resolve()
    if folder.is_dir():
        for metadata_path in sorted(folder.glob("*.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("schema") != "negative_control_v1":
                continue
            if metadata.get("category") not in CATEGORIES:
                raise ValueError(f"invalid negative category in {metadata_path.name}")
            audio_path = (folder / str(metadata["wav"])).resolve()
            if audio_path.parent != folder:
                raise ValueError("negative recording path outside corpus")
            audio, rate = read_wav(audio_path)
            negatives.append((str(metadata["id"]), metadata["category"], audio, rate))

    conditions = [("original", None, None, None)]
    conditions.extend(
        (f"{kind}_{snr:g}db_seed{seed}", kind, snr, seed)
        for kind in NOISE_TYPES for snr in SNR_DB for seed in seeds
    )
    positive_rows, negative_rows, trials = [], [], []
    for name, noise_type, snr_db, seed in conditions:
        current = []
        for index, base, context, filename, samples, rate, tokens, references in prepared:
            altered = (
                samples if noise_type is None else add_calibrated_noise(
                    samples, rate, float(snr_db), noise_type,
                    _recording_seed(seed, filename, noise_type),
                )
            )
            features = acoustic_tracks(altered, rate)
            vowel_features = cv_tracks(altered, rate)
            for occurrence, (token_index, gold) in enumerate(zip(tokens, references), 1):
                cue = nominal_target_cue_ms(token_index)
                baseline = detect_excess_fallback(features, vowel_features, cue)
                is_paired = baseline.status == "paired_candidate"
                original = baseline.consonant_onset_ms if is_paired else None
                shadow = trace_early_frication(features, original, cue)
                proposed = (
                    shadow["onset_ms"] if shadow["status"] == "shadow_candidate"
                    else None
                )
                row = {
                    "trial": name, "noise_type": noise_type or "none",
                    "snr_db": snr_db, "seed": seed,
                    "prompt_index": index, "base_unit": base, "context": context,
                    "occurrence": occurrence, "manual_onset_ms": gold,
                    "paired_status": baseline.status,
                    "baseline_method": baseline.method,
                    "cv_boundary_ms": baseline.cv_boundary_ms,
                    "baseline_onset_ms": original,
                    "shadow_onset_ms": proposed,
                    "backtrace_ms": shadow.get("backtrace_ms"),
                    "baseline_error_ms": (
                        round(original - gold, 3)
                        if gold is not None and original is not None else None
                    ),
                    "shadow_error_ms": (
                        round(proposed - gold, 3)
                        if gold is not None and proposed is not None else None
                    ),
                }
                current.append(row)
        positive_rows.extend(current)

        current_neg = []
        for recording_id, category, samples, rate in negatives:
            altered = (
                samples if noise_type is None else add_calibrated_noise(
                    samples, rate, float(snr_db), noise_type,
                    _recording_seed(seed, recording_id, noise_type),
                )
            )
            features = acoustic_tracks(altered, rate)
            vowels = cv_tracks(altered, rate)
            duration_ms = 1000.0 * len(altered) / rate
            cues = np.arange(350.0, duration_ms - 500.0 + 0.001, 200.0)
            paired = 0
            shifted = 0
            for cue in cues:
                base = detect_excess_fallback(features, vowels, float(cue))
                if base.status != "paired_candidate":
                    continue
                paired += 1
                probe = trace_early_frication(
                    features, base.consonant_onset_ms, float(cue)
                )
                shifted += int(
                    probe.get("backtrace_ms") is not None
                    and probe["backtrace_ms"] > 0
                )
            current_neg.append({
                "trial": name, "id": recording_id, "category": category,
                "windows_checked": len(cues), "paired_windows": paired,
                "paired_recording": paired > 0,
                "paired_windows_with_backtrace": shifted,
            })
        negative_rows.extend(current_neg)

        baseline_metrics = summarize_onsets(current, "baseline")
        shadow_metrics = summarize_onsets(current, "shadow")
        trials.append({
            "trial": name, "noise_type": noise_type or "none",
            "snr_db": snr_db, "seed": seed,
            "positive_total": len(current),
            "positive_paired": sum(r["paired_status"] == "paired_candidate" for r in current),
            "baseline": baseline_metrics, "shadow": shadow_metrics,
            "paired_comparison": compare_paired_rows(current),
            "negative_recordings": len(current_neg),
            "negative_paired_recordings": sum(r["paired_recording"] for r in current_neg),
        })

    aggregate = {}
    for kind in NOISE_TYPES:
        for snr in SNR_DB:
            chosen = [
                t for t in trials if t["noise_type"] == kind and t["snr_db"] == snr
            ]
            def median_metric(method, metric):
                data = [t[method][metric] for t in chosen if t[method][metric] is not None]
                return round(float(np.median(data)), 3) if data else None
            baseline_mae = median_metric("baseline", "conditional_mae_ms")
            shadow_mae = median_metric("shadow", "conditional_mae_ms")
            aggregate[f"{kind}_{snr:g}db"] = {
                "trials": len(chosen),
                "targets": len(prepared) * 2,
                "eligible_per_trial": chosen[0]["baseline"]["eligible"],
                "baseline_matched_min": min(t["baseline"]["matched"] for t in chosen),
                "baseline_matched_max": max(t["baseline"]["matched"] for t in chosen),
                "shadow_matched_min": min(t["shadow"]["matched"] for t in chosen),
                "shadow_matched_max": max(t["shadow"]["matched"] for t in chosen),
                "baseline_median_mae_ms": baseline_mae,
                "shadow_median_mae_ms": shadow_mae,
                "shadow_better_trial_count": sum(
                    t["baseline"]["conditional_mae_ms"] is not None
                    and t["shadow"]["conditional_mae_ms"] is not None
                    and t["shadow"]["conditional_mae_ms"] <
                        t["baseline"]["conditional_mae_ms"]
                    for t in chosen
                ),
                "baseline_errors_over_40ms_total": sum(
                    t["baseline"]["errors_over_40ms"] for t in chosen
                ),
                "shadow_errors_over_40ms_total": sum(
                    t["shadow"]["errors_over_40ms"] for t in chosen
                ),
                "negative_trials_with_paired_recording": sum(
                    t["negative_paired_recordings"] > 0 for t in chosen
                ),
            }
    report = {
        "analysis": "early_frication_multiseed_shadow_audit",
        "version": "2.8-development", "session_id": session_dir.name,
        "seeds": list(seeds), "noise_types": list(NOISE_TYPES),
        "snr_db": list(SNR_DB),
        "clean": trials[0],
        "aggregate": aggregate, "trials": trials,
        "positives": positive_rows, "negatives": negative_rows,
        "limitations": [
            "All randomized noise conditions reuse four development WAV recordings, with eight labeled target occurrences.",
            "Noise seeds are not independent speaker, microphone or recording sessions.",
            "The baseline determines whether the CV pair exists. Backtracing cannot recover missing CV pairs.",
            "Negative screening uses overlapping windows and nine self-labeled recordings; zero detections is not validated specificity.",
            "Backtrace could include non-speech noise even when paired CV evidence exists.",
            "No backtrace parameters were changed or fitted to the evaluation marks.",
            "Clean recordings and original predictions remain unchanged.",
        ],
    }
    destination = session_dir / "analysis" / REPORT_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
