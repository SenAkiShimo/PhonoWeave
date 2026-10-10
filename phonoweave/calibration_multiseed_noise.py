"""Frozen multi-seed perturbation audit of strict and fallback CV detection.

No detector thresholds are modified. Noise levels are defined relative to
original-recording active-speech RMS; for negative audio without active speech,
the full-recording RMS provides the reference and may be extremely low.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import zlib

import numpy as np

from .audio import read_wav
from .calibration_cv_consistency import classify_syllable
from .calibration_cv_transition import cv_tracks
from .calibration_dual_labels import load_dual_labels
from .calibration_excess_fallback import detect_excess_fallback
from .calibration_io import load_calibration_session
from .calibration_negative_controls import CATEGORIES, CORPUS_DIR
from .calibration_sibilant_onset import SIBILANTS
from .calibration_transition_onset import acoustic_tracks, nominal_target_cue_ms

SEEDS = (103, 211, 307, 419, 523, 631)
SNR_DB = (20.0, 35.0)
NOISE_TYPES = ("white", "pink")
REPORT_FILENAME = "calibration_multiseed_noise_stability_v2.6.json"


def signal_reference_rms(samples: np.ndarray) -> float:
    """Match the reference convention used by existing stress probes."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("invalid input audio")
    active = x[np.abs(x) >= 0.01]
    reference = active if len(active) else x
    return float(np.sqrt(np.mean(reference * reference)))


def add_calibrated_noise(
    samples: np.ndarray, sample_rate: int, snr_db: float,
    noise_type: str, seed: int,
) -> np.ndarray:
    """Reproducible white/pink additive noise with equal full-band RMS.

    Pink noise has 1/f power spectral density above 30 Hz. These are
    synthetic channel stressors, not observed environmental recordings.
    """
    x = np.asarray(samples, dtype=np.float64)
    if sample_rate < 8000 or not np.isfinite(snr_db) or noise_type not in NOISE_TYPES:
        raise ValueError("invalid noise specification")
    target_rms = signal_reference_rms(x) / (10.0 ** (snr_db / 20.0))
    if target_rms == 0.0:
        return x.copy()
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(len(x))
    if noise_type == "pink":
        frequencies = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
        # Amplitude ~ f^-1/2 implies power spectral density ~ 1/f.
        weight = np.sqrt(30.0 / np.maximum(frequencies, 30.0))
        weight[0] = 0.0
        noise = np.fft.irfft(np.fft.rfft(noise) * weight, n=len(x))
    noise = noise - float(np.mean(noise))
    norm = float(np.sqrt(np.mean(noise * noise)))
    if norm == 0.0:
        raise ValueError("degenerate noise")
    return x + noise * (target_rms / norm)


def _recording_seed(seed: int, recording_id: str, noise_type: str) -> int:
    return (seed + zlib.crc32((recording_id + ":" + noise_type).encode("utf-8"))) % (2**32)


def evaluate_multiseed(session_dir: Path, seeds: tuple[int, ...] = SEEDS) -> dict:
    """Measure seed-to-seed sensitivity, not independent-sample accuracy."""
    if not seeds or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    labels = load_dual_labels(session_dir)["labels"]
    pos_records = []
    failures = []
    for index, rec in sorted(session["recordings"].items()):
        prompt = session["protocol"]["prompts"][index]
        if prompt.get("base_unit") not in SIBILANTS:
            continue
        tokens = prompt.get("tokens") or str(prompt.get("spoken_pattern", "")).split()
        slots = [j for j, token in enumerate(tokens) if token == prompt.get("syllable")]
        if len(slots) != 2:
            failures.append({"prompt_index": index, "reason": "target_token_count"})
            continue
        try:
            audio, sr = read_wav(session_dir / "recordings" / rec["wav"])
            gold = []
            for occurrence in (1, 2):
                mark = (labels.get(str(index), {}).get("occurrences", {})
                        .get(str(occurrence), {}).get("consonant_onset"))
                valid = (
                    isinstance(mark, dict) and mark.get("status") == "ok"
                    and not mark.get("outside_expected_slot", False)
                    and mark.get("source") == "manual_v0.3"
                )
                gold.append(float(mark["absolute_anchor_ms"]) if valid else None)
            pos_records.append((index, prompt, rec["wav"], audio, sr, slots, gold))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"prompt_index": index, "reason": str(exc)})

    negative_records = []
    negative_dir = (session_dir.parent / CORPUS_DIR).resolve()
    if negative_dir.is_dir():
        for meta_path in sorted(negative_dir.glob("*.json")):
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                if m.get("schema") != "negative_control_v1":
                    continue
                if m.get("category") not in CATEGORIES:
                    raise ValueError("unsupported category")
                wav = (negative_dir / str(m["wav"])).resolve()
                if wav.parent != negative_dir:
                    raise ValueError("WAV path outside corpus")
                audio, sr = read_wav(wav)
                negative_records.append((m["id"], m["category"], audio, sr))
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                failures.append({"metadata": meta_path.name, "reason": str(exc)})

    rows = []
    negative_rows = []
    trials = []
    for noise_type in NOISE_TYPES:
        for snr_db in SNR_DB:
            for seed in seeds:
                trial_id = f"{noise_type}_{snr_db:g}db_seed{seed}"
                for index, prompt, filename, audio, sr, slots, gold in pos_records:
                    y = add_calibrated_noise(
                        audio, sr, snr_db, noise_type,
                        _recording_seed(seed, str(filename), noise_type)
                    )
                    onset_tracks, vowel_tracks = acoustic_tracks(y, sr), cv_tracks(y, sr)
                    for occurrence, (token_idx, reference) in enumerate(zip(slots, gold), 1):
                        cue = nominal_target_cue_ms(token_idx)
                        strict = classify_syllable(onset_tracks, vowel_tracks, cue)
                        fallback = detect_excess_fallback(onset_tracks, vowel_tracks, cue)
                        rows.append({
                            "trial": trial_id, "noise_type": noise_type,
                            "snr_db": snr_db, "seed": seed,
                            "prompt_index": index, "base_unit": prompt["base_unit"],
                            "context": prompt.get("context_family"),
                            "occurrence": occurrence, "manual_onset_ms": reference,
                            "strict_status": strict.status,
                            "fallback_status": fallback.status,
                            "fallback_method": fallback.method,
                            "strict_error_ms": (
                                round(strict.consonant_onset_ms - reference, 3)
                                if reference is not None and strict.status == "paired_candidate"
                                else None
                            ),
                            "fallback_error_ms": (
                                round(fallback.consonant_onset_ms - reference, 3)
                                if reference is not None and fallback.status == "paired_candidate"
                                else None
                            ),
                        })
                for recording_id, category, audio, sr in negative_records:
                    y = add_calibrated_noise(
                        audio, sr, snr_db, noise_type,
                        _recording_seed(seed, recording_id, noise_type)
                    )
                    onset_tracks, vowel_tracks = acoustic_tracks(y, sr), cv_tracks(y, sr)
                    length_ms = len(y) * 1000.0 / sr
                    cues = np.arange(350.0, length_ms - 500.0 + 0.001, 200.0)
                    strict_trigger = fallback_trigger = False
                    for cue in cues:
                        strict = classify_syllable(onset_tracks, vowel_tracks, float(cue))
                        fallback = detect_excess_fallback(onset_tracks, vowel_tracks, float(cue))
                        strict_trigger |= strict.status == "paired_candidate"
                        fallback_trigger |= fallback.status == "paired_candidate"
                    negative_rows.append({
                        "trial": trial_id, "noise_type": noise_type,
                        "snr_db": snr_db, "seed": seed, "id": recording_id,
                        "category": category, "windows_checked": len(cues),
                        "strict_trigger": strict_trigger,
                        "fallback_trigger": fallback_trigger,
                    })
                current = [r for r in rows if r["trial"] == trial_id]
                current_neg = [r for r in negative_rows if r["trial"] == trial_id]
                annotated = [r for r in current if r["manual_onset_ms"] is not None]
                summary = {
                    "trial": trial_id, "noise_type": noise_type,
                    "snr_db": snr_db, "seed": seed,
                    "positive_total": len(current), "annotated_total": len(annotated),
                    "negative_total": len(current_neg),
                }
                for method in ("strict", "fallback"):
                    matched = [r for r in annotated if r[f"{method}_error_ms"] is not None]
                    absolute = np.asarray([abs(r[f"{method}_error_ms"]) for r in matched])
                    summary[method] = {
                        "positive_paired": sum(r[f"{method}_status"] == "paired_candidate" for r in current),
                        "annotated_matched": len(matched),
                        "conditional_mae_ms": round(float(np.mean(absolute)), 3) if len(absolute) else None,
                        "conditional_p90_ms": round(float(np.percentile(absolute, 90)), 3) if len(absolute) else None,
                        "annotated_errors_over_40ms": int(np.count_nonzero(absolute > 40.0)),
                        "negative_recordings_triggered": sum(r[f"{method}_trigger"] for r in current_neg),
                    }
                trials.append(summary)

    aggregate = {}
    for noise_type in NOISE_TYPES:
        for snr_db in SNR_DB:
            selected = [r for r in trials if r["noise_type"] == noise_type and r["snr_db"] == snr_db]
            if not selected:
                continue
            summary = {"replicates": len(selected), "distinct_positive_recordings": len(pos_records),
                       "distinct_negative_recordings": len(negative_records)}
            for method in ("strict", "fallback"):
                paired = [r[method]["positive_paired"] for r in selected]
                matched = [r[method]["annotated_matched"] for r in selected]
                negative = [r[method]["negative_recordings_triggered"] for r in selected]
                summary[method] = {
                    "paired_mean": round(float(np.mean(paired)), 3),
                    "paired_min": int(min(paired)),
                    "paired_max": int(max(paired)),
                    "annotated_matched_min": int(min(matched)),
                    "annotated_matched_max": int(max(matched)),
                    "trials_with_any_negative_trigger": sum(n > 0 for n in negative),
                    "total_negative_recording_triggers_across_repeats": int(sum(negative)),
                }
            aggregate[f"{noise_type}_{snr_db:g}db"] = summary

    report = {
        "analysis": "multiseed_synthetic_noise_stability",
        "version": "2.6-development", "session_id": session_dir.name,
        "seeds": list(seeds), "noise_types": list(NOISE_TYPES), "snr_db": list(SNR_DB),
        "aggregate": aggregate, "trials": trials,
        "positives": rows, "negatives": negative_rows, "failures": failures,
        "limitations": [
            "Perturbed replicas of the same recordings are statistically dependent.",
            "SNR is referenced to the original active-signal RMS, not a measured environmental SNR.",
            "For silence/near-silence negatives, active-signal SNR is poorly defined.",
            "Pink and white noise have equal full-band RMS, not equal high-band SNR.",
            "Category labels are self-reported; candidate triggers are not adjudicated false positives.",
            "Overlapping windows are correlated and may omit recording edges.",
            "No thresholds were tuned or changed. All outcomes are development-only."
        ]
    }
    path = session_dir / "analysis" / REPORT_FILENAME
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
