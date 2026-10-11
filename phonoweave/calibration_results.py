"""Versioned, research-only calibration result export for downstream consumers.

Exports reproducible unlabeled acoustic candidates, not validated speaker
parameters or instructions to alter an inventory. No human annotations are
read, and no voicebank file is modified.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

from .audio import read_wav
from .calibration_cv_transition import cv_tracks
from .calibration_early_frication import trace_early_frication
from .calibration_excess_fallback import detect_excess_fallback
from .calibration_io import load_calibration_session
from .calibration_selective_early_onset import select_early_onset
from .calibration_transition_onset import acoustic_tracks, nominal_target_cue_ms

SCHEMA = "phonoweave.calibration_results"
SCHEMA_VERSION = "1.0.0"
OUTPUT_FILENAME = "calibration_results_v1.json"
SUPPORTED_BASES = frozenset({"x", "sh"})
MODEL_VERSION = "experimental_frication_cv_v2.6_selective_v2.9"


def _optional_time(value: object) -> float | None:
    """Canonicalize a candidate timestamp and reject invalid numeric data."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("candidate timestamp is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("candidate timestamp is negative or non-finite")
    return round(result, 3)


def serialize_candidate(
    *,
    prompt_index: int,
    prompt: dict,
    occurrence: int,
    nominal_cue_ms: float,
    prediction: object,
    shoulder: dict,
    selection: dict,
) -> dict:
    """Serialize one prediction without consulting manual reference anchors."""
    status = prediction.status
    if status not in {"paired_candidate", "review", "unresolved"}:
        raise ValueError("unrecognized candidate status")
    onset = _optional_time(prediction.consonant_onset_ms)
    cv = _optional_time(prediction.cv_boundary_ms)
    if status == "paired_candidate" and (
        onset is None or cv is None or not onset < cv
    ):
        raise ValueError("paired candidate must contain ordered onset and CV boundary")

    shadow = (
        _optional_time(shoulder.get("onset_ms"))
        if shoulder.get("status") == "shadow_candidate" else None
    )
    selected = _optional_time(selection.get("onset_ms"))
    applied = selection.get("applied")
    if not isinstance(applied, bool):
        raise ValueError("invalid selection flag")
    if status != "paired_candidate":
        if applied or selected is not None:
            raise ValueError("unpaired candidate cannot provide a selected onset")
        shadow = None
    else:
        if selected is None or selected > onset or selected >= cv:
            raise ValueError("selected onset is missing or inconsistent")
        if applied and shadow != selected:
            raise ValueError("selected onset does not match the shadow estimate")
        if not applied and selected != onset:
            raise ValueError("rejected refinement must retain the original onset")

    return {
        "prompt_index": prompt_index,
        "base_unit": prompt["base_unit"],
        "context_family": prompt["context_family"],
        "syllable": prompt["syllable"],
        "occurrence": occurrence,
        "nominal_cue_ms": _optional_time(nominal_cue_ms),
        "detector": {
            "status": status,
            "reason": prediction.reason,
            "method": prediction.method,
            "consonant_onset_ms": onset,
            "cv_boundary_ms": cv,
        },
        "experimental_refinement": {
            "shadow_onset_ms": shadow,
            "selected_onset_ms": selected,
            "applied": applied,
            "reason": selection["reason"],
        },
        "review_required": True,
    }


def export_calibration_results(
    session_dir: Path, *, output: Path | None = None
) -> dict:
    """Write an explicit research-only contract from recorded x/sh prompts.

    Missing recordings are recorded in coverage but are never fabricated.
    Other consonants are explicitly unsupported, not classified as negative.
    """
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    prompts = session["protocol"]["prompts"]
    recordings_dir = (session_dir / "recordings").resolve()
    results = []
    sources = []
    unsupported = []
    for index, recording in sorted(session["recordings"].items()):
        prompt = prompts[index]
        base_unit = prompt["base_unit"]
        if base_unit not in SUPPORTED_BASES:
            unsupported.append({
                "prompt_index": index, "base_unit": base_unit,
                "context_family": prompt["context_family"],
                "reason": "no_validated_model_for_class",
            })
            continue
        path = (recordings_dir / str(recording["wav"])).resolve()
        if path.parent != recordings_dir or not path.is_file():
            raise ValueError("calibration recording path is invalid")
        audio, sample_rate = read_wav(path)
        if sample_rate <= 0:
            raise ValueError("invalid recording sample rate")
        features = acoustic_tracks(audio, sample_rate)
        vowel_features = cv_tracks(audio, sample_rate)
        tokens = prompt.get("tokens") or str(prompt["spoken_pattern"]).split()
        target_positions = [
            position for position, token in enumerate(tokens)
            if token == prompt["syllable"]
        ]
        expected = int(prompt.get("target_occurrences", len(target_positions)))
        if not target_positions or expected != len(target_positions):
            raise ValueError(f"invalid target sequence at prompt {index}")
        sources.append({
            "prompt_index": index,
            "recording": f"recordings/{path.name}",
            "sample_rate": sample_rate,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        })
        for occurrence, position in enumerate(target_positions, 1):
            cue = nominal_target_cue_ms(position)
            candidate = detect_excess_fallback(features, vowel_features, cue)
            raw_onset = (
                candidate.consonant_onset_ms
                if candidate.status == "paired_candidate" else None
            )
            trace = trace_early_frication(features, raw_onset, cue)
            shadow = (
                trace["onset_ms"] if trace["status"] == "shadow_candidate"
                else None
            )
            selection = select_early_onset(
                paired_status=candidate.status,
                baseline_onset_ms=raw_onset,
                shadow_onset_ms=shadow,
                cv_boundary_ms=candidate.cv_boundary_ms,
            )
            results.append(serialize_candidate(
                prompt_index=index, prompt=prompt,
                occurrence=occurrence, nominal_cue_ms=cue,
                prediction=candidate, shoulder=trace, selection=selection,
            ))
    summary = {
        "protocol_prompts": len(prompts),
        "recorded_prompts": len(session["recordings"]),
        "supported_recorded_prompts": len(sources),
        "unsupported_recorded_prompts": len(unsupported),
        "target_occurrences": len(results),
        "paired_candidates": sum(
            row["detector"]["status"] == "paired_candidate" for row in results
        ),
        "review": sum(row["detector"]["status"] == "review" for row in results),
        "unresolved": sum(row["detector"]["status"] == "unresolved" for row in results),
        "selective_refinements": sum(
            row["experimental_refinement"]["applied"] for row in results
        ),
    }
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "session_id": session["session_id"],
        "protocol_name": session["protocol"].get("name"),
        "protocol_version": session["protocol"].get("version"),
        "model": {
            "identifier": MODEL_VERSION,
            "maturity": "research_only",
            "supported_bases": sorted(SUPPORTED_BASES),
            "time_unit": "ms_from_recording_start",
            "cue_alignment": "nominal",
            "default_voicebank_decisions_modified": False,
        },
        "summary": summary,
        "recording_sources": sources,
        "unsupported_recordings": unsupported,
        "predictions": results,
        "limitations": [
            "Paired candidates are unverified hypotheses; all require human review.",
            "Only elicited Mandarin x/sh prompts are modeled, not the full consonant inventory.",
            "Cue alignment is nominal and cannot substitute sample-accurate timing.",
            "Neither manual reference labels nor speaker-profile decisions are read.",
            "Selective backtrace is experimental and must not be treated as a trusted boundary.",
        ],
    }
    destination = (
        output.expanduser().resolve() if output is not None
        else session_dir / "analysis" / OUTPUT_FILENAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
