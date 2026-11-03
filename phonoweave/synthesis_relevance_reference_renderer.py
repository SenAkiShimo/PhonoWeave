"""Executable S01-only reference splice renderer for the frozen /sh/ pilot.

This is a deliberately simple *signal-processing reference renderer*, not
OpenUtau, DDSP, a singing synthesizer, or proof of perceptual improvement.
It produces real PCM WAVs and an internally audited external-render receipt.
No S02 waveform is opened or used as a donor. The frozen corpus is rehashed,
not rewritten. Both arms share the same S01-derived right-hand vowel sample.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import wave

import numpy as np

from .audio import read_wav
from .synthesis_relevance_controlled_corpus import FINALS, PAIRS, REPEATS
from .synthesis_relevance_rendered_intake import (
    PLAN_SCHEMA, _canonical, _check_receipt, _digest,
    make_receipt_template, verify_frozen_corpus,
)

SCHEMA = "phonoweave.synthesis_relevance.reference_splice_renderer.v1"
RATE = 48000
FRICATION_MS = 180
VOWEL_MS = 620
CROSSFADE_MS = 8
CONFIG = {
    "schema": SCHEMA,
    "render_sample_rate_hz": RATE,
    "frication_ms": FRICATION_MS,
    "vowel_ms": VOWEL_MS,
    "crossfade_ms": CROSSFADE_MS,
    "peak_guard": 0.90,
    "donor_selection": "deterministic_first_S01_take_opposite_final_vs_same_final",
    "score_type": "fixed_reference_syllable_no_pitch_transposition",
}


def _canonical_take_path(root: Path, take_id: str) -> Path:
    expected = {
        f"S01_sh{final}_{repeat:02d}":
        f"audio/S01/sh{final}_{repeat:02d}.wav"
        for final in FINALS for repeat in range(1, REPEATS + 1)
    }
    rel = expected.get(take_id)
    if rel is None:
        raise ValueError("attempt to render with a non-S01 donor")
    p = (root / rel).resolve(strict=True)
    if not p.is_relative_to(root) or not p.is_file():
        raise ValueError("S01 donor WAV path escapes capture corpus")
    return p


def _stretch(segment: np.ndarray, frames: int) -> np.ndarray:
    if frames < 2 or len(segment) < 2:
        raise ValueError("reference splice segment has insufficient samples")
    samples = np.asarray(segment, dtype=np.float64)
    if not np.all(np.isfinite(samples)):
        raise ValueError("nonfinite donor samples")
    if len(samples) == frames:
        return samples.copy()
    return np.interp(
        np.linspace(0, len(samples) - 1, frames),
        np.arange(len(samples)), samples
    )


def _render(reference: np.ndarray, frication: np.ndarray, rate: int) -> np.ndarray:
    """Crossfade a donor frication to an identical shared S01 vowel tail."""
    left_frames = round(FRICATION_MS * RATE / 1000)
    right_frames = round(VOWEL_MS * RATE / 1000)
    fade = round(CROSSFADE_MS * RATE / 1000)
    a = _stretch(frication, left_frames)
    b = _stretch(reference, right_frames)
    if len(a) <= fade or len(b) <= fade:
        raise ValueError("invalid crossfade")
    mix = np.arange(fade, dtype=np.float64) / (fade - 1)
    overlap = a[-fade:] * np.cos(mix * np.pi / 2) + b[:fade] * np.sin(mix * np.pi / 2)
    joined = np.concatenate((a[:-fade], overlap, b[fade:]))
    # One deterministic fixed peak guard for both arms; never independent
    # RMS normalization (which would confound the A/B loudness comparison).
    return joined


def _write_pcm(path: Path, audio: np.ndarray) -> None:
    if not len(audio) or not np.all(np.isfinite(audio)):
        raise ValueError("invalid reference render signal")
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-5:
        raise ValueError("render is silent")
    if peak > 1.0:
        raise ValueError("render exceeds full-scale")
    pcm = np.round(np.clip(audio, -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            wav.writeframes(pcm.tobytes())


def _source_segment(
    root: Path, take_id: str, annotations: dict,
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if take_id in cache:
        return cache[take_id]
    path = _canonical_take_path(root, take_id)
    entry = annotations.get(take_id)
    if not isinstance(entry, dict):
        raise ValueError(f"missing frozen S01 annotation: {take_id}")
    onset, boundary = entry.get("frication_onset_ms"), entry.get("cv_boundary_ms")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and np.isfinite(v) for v in (onset, boundary)):
        raise ValueError("S01 onset/CV boundary annotation is required")
    samples, rate = read_wav(path)
    start = int(round(onset * rate / 1000))
    middle = int(round(boundary * rate / 1000))
    tail_end = min(len(samples), middle + round(0.65 * rate))
    if not (0 <= start < middle <= len(samples) and
            middle - start >= round(0.04 * rate) and
            tail_end - middle >= round(0.15 * rate)):
        raise ValueError(f"insufficient annotated S01 frication or vowel: {take_id}")
    frication = np.asarray(samples[start:middle], dtype=np.float64)
    vowel = np.asarray(samples[middle:tail_end], dtype=np.float64)
    if (np.sqrt(np.mean(frication ** 2)) < 1e-5
            or np.sqrt(np.mean(vowel ** 2)) < 1e-5):
        raise ValueError(f"silent S01 donor segment: {take_id}")
    cache[take_id] = (frication, vowel)
    return cache[take_id]


def render_reference_plan(
    plan: dict, manifest: dict, corpus: Path, output: Path,
) -> dict:
    """Render and atomically publish all 48 arms plus exact provenance receipt."""
    corpus = corpus.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if output == corpus or corpus in output.parents:
        raise ValueError("render output cannot be inside original capture corpus")
    if output.exists():
        raise FileExistsError("refusing to overwrite existing render output")
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("task_count") != 24
            or not isinstance(plan.get("tasks"), list)
            or len(plan["tasks"]) != 24):
        raise ValueError("invalid frozen 24-task render plan")
    if (manifest.get("speaker_id") != plan.get("speaker_id")
            or manifest.get("pitch_subbank") != plan.get("pitch_subbank")):
        raise ValueError("render plan speaker identity mismatch")
    frozen = sha256(json.dumps(
        manifest, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    if frozen != plan.get("source_manifest_sha256"):
        raise ValueError("render plan manifest hash is stale")
    verify_frozen_corpus(plan, corpus)
    annotations = {
        row["take_id"]: row for row in manifest.get("recordings", [])
        if isinstance(row, dict) and row.get("session_id") == "S01"
    }
    if len(annotations) != 24:
        raise ValueError("expected 24 distinct S01 annotations")

    # Restrict donor selection to the named S01 IDs. Pooled condition chooses
    # the opposite final within the predeclared pair for a controlled contrast,
    # while split chooses the target-final donor. Both share the same vowel
    # samples from the target-final S01 donor; S02 audio is never decoded.
    tasks = plan["tasks"]
    receipt = make_receipt_template(plan)
    receipt.update({
        "engine_name": "PhonoWeave Reference Splice",
        "engine_version": "1",
        "resampler_name": "NumPy linear duration remap",
        "resampler_version": "1",
        "score_template_sha256": _canonical({
            "syllable": "frozen_same_duration_reference", "target_ms":
            FRICATION_MS + VOWEL_MS - CROSSFADE_MS,
            "no_pitch_transposition": True,
        }),
        "render_configuration_sha256": _canonical(CONFIG),
        "same_score_and_render_settings_between_arms_confirmed": True,
        "S02_audio_not_used_as_render_donor_or_conditioning_confirmed": True,
    })
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".pw-reference-render-", dir=output.parent,
    ) as temporary:
        stage = Path(temporary) / "renders"
        stage.mkdir()
        for task, row in zip(tasks, receipt["trial_receipts"]):
            target_final = task["target_final"]
            pair = next(p for p in PAIRS if target_final in p)
            opposite_final = pair[0] if pair[1] == target_final else pair[1]
            split_donor = f"S01_sh{target_final}_01"
            pooled_donor = f"S01_sh{opposite_final}_01"
            if (split_donor not in task["allowed_S01_split_donor_take_ids"]
                    or pooled_donor not in task["allowed_S01_pooled_donor_take_ids"]):
                raise ValueError("proposed donor not in frozen S01 pool")
            _, common_vowel = _source_segment(corpus, split_donor, annotations, cache)
            row["target_take_id"] = task["target_take_id"]
            for arm, donor in (("pooled", pooled_donor), ("split", split_donor)):
                frication, _ = _source_segment(corpus, donor, annotations, cache)
                rendered = _render(common_vowel, frication, RATE)
                path = stage / task["render_wavs"][arm]
                _write_pcm(path, rendered)
                row[f"{arm}_donor_take_ids_used"] = [donor]
                row[f"{arm}_wav_sha256"] = _digest(path)
        # Validate the exact same receipt and media API used by the blind pack.
        _check_receipt(plan, receipt, stage)
        receipt_path = stage / "renderer_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "schema": SCHEMA, "status": "reference_splice_audio_written",
            "source_manifest_sha256": frozen,
            "render_plan_sha256": _canonical(plan),
            "render_receipt_sha256": _canonical(receipt),
            "reference_only": True, "actual_singing_resampler_used": False,
            "S02_wav_waveforms_used_for_rendering": False,
            "source_bank_modified": False,
            "rendered_pairs": len(tasks),
            "rendered_wavs": len(tasks) * 2,
            "limitations": [
                "A simple S01-only splice-and-time-remap baseline, not DDSP, OpenUtau or singing synthesis.",
                "Neither arm performs pitch transposition or note-level phonemizer routing.",
                "Pooled arm deliberately takes a contrasting final's frication; "
                "it is not a pooled-donor estimator and must not be labeled as one in efficacy results.",
                "Both arms share a target-final S01 vowel, not an S02 reference vowel.",
                "The training source lacks independently evaluated rendering fidelity.",
            ],
        }
        (stage / "reference_render_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError("destination appeared during reference rendering")
        os.rename(stage, output)
    return metadata
