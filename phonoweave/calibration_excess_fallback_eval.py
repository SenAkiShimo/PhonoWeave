"""Paired comparison of strict and experimental noise-excess onset fallback."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import numpy as np

from .audio import read_wav
from .calibration_cv_consistency import classify_syllable
from .calibration_cv_transition import cv_tracks
from .calibration_dual_labels import load_dual_labels
from .calibration_excess_fallback import detect_excess_fallback
from .calibration_io import load_calibration_session
from .calibration_negative_controls import CATEGORIES, CORPUS_DIR
from .calibration_robustness import SCENARIOS, perturb
from .calibration_sibilant_onset import SIBILANTS
from .calibration_transition_onset import acoustic_tracks, nominal_target_cue_ms

REPORT_NAME = "calibration_excess_fallback_comparison_v2.5.json"


def evaluate_excess_fallback(session_dir: Path) -> dict:
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    labels = load_dual_labels(session_dir)["labels"]
    positive = []
    negative = []
    failures = []

    for index, record in sorted(session["recordings"].items()):
        prompt = session["protocol"]["prompts"][index]
        if prompt.get("base_unit") not in SIBILANTS:
            continue
        tokens = prompt.get("tokens") or str(prompt.get("spoken_pattern", "")).split()
        positions = [k for k, tok in enumerate(tokens) if tok == prompt.get("syllable")]
        if len(positions) != 2:
            failures.append({"prompt_index": index, "reason": "target_token_count"})
            continue
        try:
            audio, sr = read_wav(session_dir / "recordings" / record["wav"])
            for scenario_index, scenario in enumerate(SCENARIOS):
                modified = perturb(
                    np.asarray(audio, dtype=float), scenario,
                    np.random.default_rng(8500 + index * 11 + scenario_index)
                )
                tr, cvtr = acoustic_tracks(modified, sr), cv_tracks(modified, sr)
                for occurrence, token_index in enumerate(positions, 1):
                    cue = nominal_target_cue_ms(token_index)
                    strict = classify_syllable(tr, cvtr, cue)
                    proposed = detect_excess_fallback(tr, cvtr, cue)
                    mark = (labels.get(str(index), {}).get("occurrences", {})
                            .get(str(occurrence), {}).get("consonant_onset"))
                    eligible = (
                        isinstance(mark, dict) and mark.get("status") == "ok"
                        and not mark.get("outside_expected_slot", False)
                        and mark.get("source") == "manual_v0.3"
                    )
                    gold = float(mark["absolute_anchor_ms"]) if eligible else None
                    positive.append({
                        "prompt_index": index, "base_unit": prompt["base_unit"],
                        "context": prompt.get("context_family"),
                        "occurrence": occurrence, "scenario": scenario,
                        "manual_onset_ms": gold,
                        "strict_status": strict.status,
                        "strict_onset_ms": strict.consonant_onset_ms,
                        "fallback_status": proposed.status,
                        "fallback_method": proposed.method,
                        "fallback_onset_ms": proposed.consonant_onset_ms,
                        "fallback_cv_ms": proposed.cv_boundary_ms,
                        "fallback_threshold_dbfs": proposed.threshold_dbfs,
                        "strict_signed_error_ms": (
                            round(strict.consonant_onset_ms - gold, 3)
                            if gold is not None and strict.status == "paired_candidate" else None
                        ),
                        "fallback_signed_error_ms": (
                            round(proposed.consonant_onset_ms - gold, 3)
                            if gold is not None and proposed.status == "paired_candidate" else None
                        ),
                    })
        except (ValueError, OSError, KeyError, TypeError) as exc:
            failures.append({"prompt_index": index, "reason": str(exc)})

    folder = (session_dir.parent / CORPUS_DIR).resolve()
    if folder.is_dir():
        for meta_file in sorted(folder.glob("*.json")):
            try:
                m = json.loads(meta_file.read_text(encoding="utf-8"))
                if m.get("schema") != "negative_control_v1":
                    continue
                if m.get("category") not in CATEGORIES:
                    raise ValueError("invalid category")
                audio_file = (folder / str(m["wav"])).resolve()
                if audio_file.parent != folder:
                    raise ValueError("WAV outside corpus")
                audio, sr = read_wav(audio_file)
                duration = 1000 * len(audio) / sr
                cues = np.arange(350.0, duration - 500.0 + 0.001, 200.0)
                for scenario_index, scenario in enumerate(SCENARIOS):
                    modified = perturb(
                        np.asarray(audio, dtype=float), scenario,
                        np.random.default_rng(9300 + scenario_index * 100 + sum(map(ord, str(m["id"]))))
                    )
                    tr, cvtr = acoustic_tracks(modified, sr), cv_tracks(modified, sr)
                    for window_type, positions in (
                        ("fixed_1250ms", [1250.0]),
                        ("overlapping_200ms", cues),
                    ):
                        strict_hits = 0
                        proposed_hits = 0
                        recovered = 0
                        for cue in positions:
                            strict = classify_syllable(tr, cvtr, float(cue))
                            proposed = detect_excess_fallback(tr, cvtr, float(cue))
                            strict_ok = strict.status == "paired_candidate"
                            proposed_ok = proposed.status == "paired_candidate"
                            strict_hits += int(strict_ok)
                            proposed_hits += int(proposed_ok)
                            recovered += int(proposed_ok and not strict_ok)
                        negative.append({
                            "id": m["id"], "category": m["category"],
                            "scenario": scenario, "window_type": window_type,
                            "windows_checked": len(positions),
                            "strict_paired_windows": strict_hits,
                            "fallback_paired_windows": proposed_hits,
                            "new_paired_windows": recovered,
                            "strict_recording_trigger": strict_hits > 0,
                            "fallback_recording_trigger": proposed_hits > 0,
                        })
            except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                failures.append({"metadata": meta_file.name, "reason": str(exc)})

    summary = {}
    for scenario in SCENARIOS:
        pos = [r for r in positive if r["scenario"] == scenario]
        gold = [r for r in pos if r["manual_onset_ms"] is not None]
        row = {"positive_occurrences": len(pos), "eligible_gold": len(gold)}
        for method in ("strict", "fallback"):
            ok = [r for r in pos if r[f"{method}_status"] == "paired_candidate"]
            errors = [abs(r[f"{method}_signed_error_ms"]) for r in gold
                      if r[f"{method}_signed_error_ms"] is not None]
            row[method] = {
                "positive_paired": len(ok),
                "gold_matched": len(errors),
                "conditional_mae_ms": round(float(np.mean(errors)), 3) if errors else None,
                "gold_coverage": round(len(errors) / len(gold), 4) if gold else None,
            }
        row["new_positive_paired"] = sum(
            r["strict_status"] != "paired_candidate"
            and r["fallback_status"] == "paired_candidate" for r in pos
        )
        for window_type in ("fixed_1250ms", "overlapping_200ms"):
            neg = [r for r in negative
                   if r["scenario"] == scenario and r["window_type"] == window_type]
            row[window_type] = {
                "recordings": len(neg),
                "strict_recording_triggers": sum(r["strict_recording_trigger"] for r in neg),
                "fallback_recording_triggers": sum(r["fallback_recording_trigger"] for r in neg),
                "new_fallback_recording_triggers": sum(
                    r["fallback_recording_trigger"] and not r["strict_recording_trigger"]
                    for r in neg
                ),
                "fallback_trigger_categories": dict(Counter(
                    r["category"] for r in neg if r["fallback_recording_trigger"]
                )),
            }
        summary[scenario] = row
    report = {
        "analysis": "experimental_excess_power_fallback_comparison",
        "version": "2.5-development",
        "session_id": session_dir.name,
        "summary": summary,
        "positives": positive, "negatives": negative, "failures": failures,
        "cautions": [
            "Fallback decisions are exploratory and are not the production detector.",
            "CV-confirmed recovery does not guarantee correct consonant ownership.",
            "The same development audio, self-reported negative labels, and simulated noise are reused.",
            "Matched-only MAE must always be reported together with positive coverage.",
            "Overlapping negative windows are not independent events and do not cover edges.",
            "Audio cue synchronization remains nominal, not sample-accurate."
        ],
    }
    output = session_dir / "analysis" / REPORT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
