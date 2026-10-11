"""Frozen v2.9 early-onset evaluation on a separately recorded session."""
from __future__ import annotations

import hashlib
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
from .calibration_selective_early_onset import select_early_onset, score_rows, pairwise_score
from .calibration_transition_onset import acoustic_tracks, nominal_target_cue_ms

REPORT_FILE = "calibration_independent_session_v3.0.json"
DEV_SESSION_ID = "20261009_225541_655813"
ALLOWED_DEVICES = ("phone", "built_in", "external_usb", "other")
ALLOWED_ENVIRONMENTS = ("quiet", "indoor_noise", "outdoor_noise", "other")


def evaluate_independent_session(
    session_dir: Path, development_dir: Path,
    *, device: str, environment: str,
) -> dict:
    if device not in ALLOWED_DEVICES or environment not in ALLOWED_ENVIRONMENTS:
        raise ValueError("device and environment must use declared categories")
    session_dir = session_dir.expanduser().resolve()
    development_dir = development_dir.expanduser().resolve()
    if session_dir == development_dir or session_dir.name == DEV_SESSION_ID:
        raise ValueError("development session cannot be an independent evaluation")
    dev = load_calibration_session(development_dir)
    session = load_calibration_session(session_dir)
    if dev["session_id"] == session["session_id"]:
        raise ValueError("evaluation and development session IDs are identical")
    seen = {}
    for index, _, _ in TARGETS:
        rec = dev["recordings"].get(index)
        if not rec:
            raise ValueError("development recording missing")
        path = development_dir / "recordings" / rec["wav"]
        seen[hashlib.sha256(path.read_bytes()).hexdigest()] = path.name

    marks = load_dual_labels(session_dir)["labels"]
    rows = []
    file_fingerprints = {}
    for index, base, context in TARGETS:
        prompt = session["protocol"]["prompts"][index]
        if (prompt["base_unit"],prompt["context_family"]) != (base,context):
            raise ValueError(f"protocol mismatch at {index}")
        rec = session["recordings"].get(index)
        if not rec:
            raise ValueError(f"missing independent recording at {index}")
        path = (session_dir / "recordings" / rec["wav"]).resolve()
        if path.parent != (session_dir / "recordings").resolve():
            raise ValueError("recording path outside session")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError(f"audio overlaps development corpus: {path.name}")
        if digest in file_fingerprints.values():
            raise ValueError("duplicate recording inside independent session")
        file_fingerprints[path.name] = digest
        samples, rate = read_wav(path)
        tracks, vowel = acoustic_tracks(samples,rate), cv_tracks(samples,rate)
        tokens = prompt.get("tokens") or str(prompt["spoken_pattern"]).split()
        positions = [i for i, token in enumerate(tokens) if token == prompt["syllable"]]
        if len(positions) != 2:
            raise ValueError("unexpected target token count")
        for occurrence, position in enumerate(positions, 1):
            cue = nominal_target_cue_ms(position)
            result = detect_excess_fallback(tracks,vowel,cue)
            paired = result.status == "paired_candidate"
            baseline = result.consonant_onset_ms if paired else None
            trace = trace_early_frication(tracks,baseline,cue)
            shadow = trace.get("onset_ms") if trace["status"] == "shadow_candidate" else None
            selection = select_early_onset(
                paired_status=result.status, baseline_onset_ms=baseline,
                shadow_onset_ms=shadow, cv_boundary_ms=result.cv_boundary_ms,
            )
            label = (marks.get(str(index),{}).get("occurrences",{})
                     .get(str(occurrence),{}).get("consonant_onset"))
            valid = (isinstance(label,dict) and label.get("status")=="ok"
                     and label.get("source")=="manual_v0.3"
                     and not label.get("outside_expected_slot",False))
            gold = float(label["absolute_anchor_ms"]) if valid else None
            cv_label = (marks.get(str(index),{}).get("occurrences",{})
                        .get(str(occurrence),{}).get("cv_boundary"))
            cv_valid = (isinstance(cv_label,dict) and cv_label.get("status")=="ok"
                        and cv_label.get("source")=="manual_v0.3"
                        and not cv_label.get("outside_expected_slot",False))
            gold_cv = float(cv_label["absolute_anchor_ms"]) if cv_valid else None
            def error(estimate):
                return round(float(estimate)-gold,3) if estimate is not None and gold is not None else None
            rows.append({
                "prompt_index": index, "base_unit": base, "context":context,
                "occurrence":occurrence, "status":result.status,
                "manual_onset_ms":gold, "manual_cv_ms":gold_cv,
                "baseline_onset_ms":baseline, "shadow_onset_ms":shadow,
                "selected_onset_ms":selection["onset_ms"],
                "cv_boundary_ms":result.cv_boundary_ms,
                "decision_applied":selection["applied"],"decision_reason":selection["reason"],
                "baseline_error_ms":error(baseline),
                "shadow_error_ms":error(shadow),
                "selected_error_ms":error(selection["onset_ms"]),
                "cv_error_ms":round(result.cv_boundary_ms-gold_cv,3)
                    if paired and gold_cv is not None else None,
            })
    # Preserve missing-label cases as explicit missing coverage rather than
    # replacing absent references with nominal beat times.
    results = {
        "analysis":"independent_session_frozen_onset",
        "version":"3.0-evaluation",
        "session_id":session["session_id"],
        "development_session_id":dev["session_id"],
        "capture":{"device":device,"environment":environment},
        "recording_sha256":file_fingerprints,
        "targets":len(rows),
        "annotated_onsets":sum(r["manual_onset_ms"] is not None for r in rows),
        "annotated_cv_boundaries":sum(r["manual_cv_ms"] is not None for r in rows),
        "paired_candidates":sum(r["status"]=="paired_candidate" for r in rows),
        "baseline":score_rows(rows,"baseline_error_ms"),
        "always_backtrace":score_rows(rows,"shadow_error_ms"),
        "selective":score_rows(rows,"selected_error_ms"),
        "cv_boundary": {
            "eligible":sum(r["manual_cv_ms"] is not None for r in rows),
            "matched":sum(r["cv_error_ms"] is not None for r in rows),
            "conditional_mae_ms":round(float(np.mean([
                abs(r["cv_error_ms"]) for r in rows if r["cv_error_ms"] is not None
            ])),3) if any(r["cv_error_ms"] is not None for r in rows) else None,
        },
        "paired_comparison":pairwise_score(rows),
        "rows":rows,
        "limitations":[
            "Recording SHA-256 detects exact copies, not acoustically transformed duplicates.",
            "Capture device and environment are declared rather than independently instrumented.",
            "One new session is not cross-speaker, cross-device, or cross-environment generalization.",
            "Nominal beat cue timestamps are not sample-accurate synchronization.",
            "Reference onsets and CV boundaries must be independently annotated before interpreting accuracy.",
            "Frozen v2.9 rule and existing v2.6 detection thresholds are not modified.",
        ],
    }
    out=session_dir/"analysis"/REPORT_FILE
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return results
