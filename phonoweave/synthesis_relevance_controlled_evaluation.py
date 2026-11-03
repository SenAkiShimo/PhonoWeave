"""Frozen two-session /sh/ pilot evaluation using annotated frication-only WAVs.

S01 is the only source of donors and policy ranking; S02 is *always* held out.
This is a spectral-centroid acoustic proxy, NEVER rendered singing or a
perceptual outcome. Refuse to score incomplete or altered capture manifests.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .audio import read_wav
from .synthesis_relevance_controlled_corpus import (
    FINALS, PAIRS, ROLES, REPEATS, SCHEMA as CORPUS_SCHEMA, audit_manifest,
)
from .synthesis_relevance_unit_merging_cost import _distance, _signature, WINDOW_MS

SCHEMA = "phonoweave.synthesis_relevance.controlled_session_proxy.v1"
PAIR_NAMES = tuple("/".join(pair) for pair in PAIRS)
# Prespecified, not selected from S02.
LINGUISTIC_HEURISTIC = ("a/ua", "an/uan", "ai/uai", "ang/uang")
WINDOW_MARGIN_MS = WINDOW_MS / 2 + 2.0


def _read_checked_signature(root: Path, entry: dict) -> tuple[float, ...]:
    rel = entry["relative_wav"]
    path = (root / rel).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("audio path leaves pilot directory")
    samples, rate = read_wav(path)
    onset, cv = float(entry["frication_onset_ms"]), float(entry["cv_boundary_ms"])
    # Choose a centered 20-ms interval strictly within the labeled frication
    # region. This avoids a waveform window in the vowel by construction.
    if cv - onset < 2 * WINDOW_MARGIN_MS:
        raise ValueError(f"frication interval too short for analysis: {entry['take_id']}")
    middle = (onset + cv) / 2
    if (not np.isfinite(middle) or middle - WINDOW_MS / 2 < onset
            or middle + WINDOW_MS / 2 > cv):
        raise ValueError("invalid center for frication-only feature")
    spectrum, _rms = _signature(samples, rate, middle)
    if len(spectrum) != 48 or not all(math.isfinite(x) for x in spectrum):
        raise ValueError("non-finite or incomplete frication signature")
    return spectrum


def _mean(rows: list[tuple[float, ...]]) -> tuple[float, ...]:
    if not rows:
        raise ValueError("missing required training signatures")
    return tuple(np.mean(np.asarray(rows, dtype=float), axis=0).tolist())


def _average_losses(
    train: dict[str, list[tuple[float, ...]]],
    test: dict[str, list[tuple[float, ...]]],
) -> tuple[list[dict], float]:
    """Return per-target S02 losses and S01-only split utility ranking."""
    pair_losses, train_utilities = [], []
    for pair in PAIRS:
        left, right = pair
        for f in pair:
            if len(train[f]) != REPEATS or len(test[f]) != REPEATS:
                raise ValueError("pilot final replication mismatch")
        donors = {f: _mean(train[f]) for f in pair}
        pool = _mean(train[left] + train[right])
        train_delta = np.mean([
            _distance(sig, pool) - _distance(sig, donors[f])
            for f in pair for sig in train[f]
        ])
        train_utilities.append(float(train_delta))
        for final in pair:
            for index, sig in enumerate(test[final], start=1):
                pair_losses.append({
                    "pair": "/".join(pair), "final": final,
                    "heldout_repeat": index,
                    "pooled_loss": float(_distance(sig, pool)),
                    "split_loss": float(_distance(sig, donors[final])),
                })
    return pair_losses, dict(zip(PAIR_NAMES, train_utilities))


def evaluate_controlled_manifest(data: dict, root: Path) -> dict:
    """Only emit metrics when all preregistered integrity requirements pass."""
    if data.get("schema") != CORPUS_SCHEMA:
        raise ValueError("unsupported frozen controlled-corpus schema")
    root = root.expanduser().resolve(strict=True)
    audit = audit_manifest(data, root)
    if not audit["analysis_ready"]:
        return {
            "schema": SCHEMA,
            "status": "insufficient",
            "audit": audit,
            "budget_curves": [],
            "heldout_losses": [],
            "outcome_status": "not_evaluated",
            "production_ready": False,
        }
    # Require immutable frozen ids/roles/paths, not just 48 valid files.
    actual = {}
    for row in data["recordings"]:
        sid, final, repeat = row["session_id"], row["final"], row["repeat"]
        expected_id = f"{sid}_sh{final}_{repeat:02d}"
        expected_path = f"audio/{sid}/sh{final}_{repeat:02d}.wav"
        if row["take_id"] != expected_id or row["relative_wav"] != expected_path:
            raise ValueError("noncanonical pilot target identity or path")
        key = (sid, final, repeat)
        if key in actual:
            raise ValueError("duplicate frozen capture cell")
        actual[key] = row

    signatures: dict[str, dict[str, list[tuple[float, ...]]]] = {
        "S01": defaultdict(list), "S02": defaultdict(list)
    }
    audio_hashes = {}
    for sid, _role in ROLES:
        for final in FINALS:
            for repeat in range(1, REPEATS + 1):
                item = actual[(sid, final, repeat)]
                signature = _read_checked_signature(root, item)
                signatures[sid][final].append(signature)
                audio_hashes[item["take_id"]] = item["wav_sha256"]
    losses, train_gain = _average_losses(signatures["S01"], signatures["S02"])
    rankings = {
        "fixed_lexicographic": tuple(sorted(PAIR_NAMES)),
        "prespecified_linguistic_heuristic": LINGUISTIC_HEURISTIC,
        "speaker_adaptive_S01_only": tuple(sorted(
            PAIR_NAMES, key=lambda x: (-train_gain[x], x)
        )),
    }
    # The same 24 independent-session S02 WAVs are evaluated at every point.
    # Each k selects exactly k additional split pair identities; this is not
    # equivalent to k new capture lines or equal-duration recording budgets.
    curves = []
    for k in range(len(PAIR_NAMES) + 1):
        for name, ranking in rankings.items():
            chosen = set(ranking[:k])
            errors = [
                row["split_loss"] if row["pair"] in chosen else row["pooled_loss"]
                for row in losses
            ]
            curves.append({
                "budget_split_pairs": k,
                "method": name,
                "selected_pairs": sorted(chosen),
                "n_heldout_WAVs": len(errors),
                "mean_heldout_log_spectral_distance": float(np.mean(errors)),
            })
    # Do not leak local WAV paths or raw audio into the report.
    return {
        "schema": SCHEMA,
        "status": "evaluable_acoustic_proxy",
        "study_id": data["study_id"],
        "speaker_id": data["speaker_id"],
        "pitch_subbank": data["pitch_subbank"],
        "protocol": {
            "train_session": "S01",
            "heldout_session": "S02",
            "sessions_are_distinct_dates": True,
            "frication_window_ms": WINDOW_MS,
            "window_definition": "midpoint_of_human_annotated_frication",
            "budget_unit": "one additional split-final-pair donor identity",
            "frozen_final_pairs": [list(p) for p in PAIRS],
            "heldout_role_used_for_selection": False,
            "repeats_per_final_and_session": REPEATS,
        },
        "source_manifest_sha256": sha256(
            json.dumps(data, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "source_audio_sha256_by_take": dict(sorted(audio_hashes.items())),
        "train_only_utility": {k: train_gain[k] for k in PAIR_NAMES},
        "rankings": {k: list(v) for k, v in rankings.items()},
        "heldout_losses": losses,
        "budget_curves": curves,
        "n_heldout_WAVs": len(losses),
        "outcome_status": "S02_spectral_proxy_only",
        "production_ready": False,
        "production_alias_export": "disabled",
        "source_WAV_OTO_modified": False,
        "limitations": [
            "One speaker, one subbank, two sessions: no population generalization.",
            "S01 training takes are separate from S02 heldout by preregistered session.",
            "Session dates and hashes are declared evidence, not proof of "
            "independent physiological phonation or absence of device confounds.",
            "Frication intervals are human-annotated and not necessarily error-free.",
            "Midpoint spectra are diagnostic proxies, not DDSP, resampler or "
            "OpenUtau rendered-song quality.",
            "Budget is split final-pair identities, NOT measured time or "
            "actual additional recording line costs.",
            "The fixed phonological heuristic is prespecified but not "
            "a universal rule or calibrated strong baseline.",
        ],
    }


def run_controlled_proxy(manifest: Path, output: Path, *, root: Path | None = None) -> dict:
    manifest = manifest.expanduser().resolve(strict=True)
    source = (root or manifest.parent).expanduser().resolve(strict=True)
    result = evaluate_controlled_manifest(
        json.loads(manifest.read_text(encoding="utf-8")), source
    )
    output = output.expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("evaluation output must be outside the capture corpus")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result
