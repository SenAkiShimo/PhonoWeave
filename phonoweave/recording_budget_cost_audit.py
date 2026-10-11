"""Audit OTO-acoustic segment yield and WAV-duration cost at fixed WAV budgets.

No selection policies, holdout partition or distance metric are changed.
Cost is full physical WAV duration including silence and non-target
phonemes. Segment yield counts deduplicated eligible s/sh/x examples
already extracted from OTO, not all OTO aliases or independent phonemes.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, median
import wave

from .mandarin import collect_observations, context_for
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .recording_budget_benchmark import (
    BASES, BUDGETS, POLICIES, RecordingUnit, load_recording_units,
    metadata, select_sequence,
)
from .recording_budget_cross_voicebank import validate_manifest
from .recording_budget_disjoint_cv import disjoint_folds, PARTITION_SEED

VERSION = "0.6-oto-segment-and-wav-duration-audit"


def eligible_wav_durations(
    root: Path, units: tuple[RecordingUnit, ...]
) -> dict[str, float]:
    """Read the entire source WAV length for every eligible SHA-256 group.

    Do not confuse number of OTO labels with number of recording takes.
    """
    entries, _ = load_voicebank(root)
    observations = collect_observations(
        (entry for entry in entries if entry.wav_path.is_file()),
        affix_pairs(load_prefix_maps(root)),
    )
    wav_paths = sorted({
        o.entry.wav_path.resolve()
        for o in observations
        if o.base_unit in BASES
        and context_for(o.base_unit, o.final) in {"plain", "rounded"}
    })
    eligible_ids = {u.digest for u in units}
    durations = {}
    for wav in wav_paths:
        digest = sha256(wav.read_bytes()).hexdigest()
        if digest not in eligible_ids:
            continue
        try:
            with wave.open(str(wav), "rb") as handle:
                if handle.getframerate() <= 0:
                    raise ValueError("invalid WAV sample rate")
                seconds = handle.getnframes() / handle.getframerate()
        except (wave.Error, OSError) as exc:
            raise ValueError("cannot read eligible WAV duration") from exc
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("invalid eligible WAV duration")
        if digest in durations and not math.isclose(
            durations[digest], seconds, rel_tol=0, abs_tol=1e-6
        ):
            raise ValueError("identical WAV hashes have different durations")
        durations[digest] = seconds
    missing = eligible_ids - durations.keys()
    if missing:
        raise ValueError(f"missing durations for {len(missing)} eligible WAV groups")
    return durations


def audit_recording_costs(
    units: tuple[RecordingUnit, ...],
    wav_seconds: dict[str, float],
    *,
    budgets: tuple[float, ...] = BUDGETS,
    seed: int = PARTITION_SEED,
) -> dict:
    """Inspect acoustic evidence obtained by frozen WAV-count acquisitions."""
    if len({u.digest for u in units}) != len(units):
        raise ValueError("duplicate WAV groups")
    if set(wav_seconds) != {u.digest for u in units}:
        raise ValueError("WAV duration index must match eligible recording groups")
    if any(not math.isfinite(v) or v <= 0 for v in wav_seconds.values()):
        raise ValueError("WAV durations must be positive and finite")
    if not budgets or any(not 0 < fraction <= 1 for fraction in budgets):
        raise ValueError("invalid budget fractions")
    groups = disjoint_folds(units, seed=seed)
    rows = []
    for fold, heldout in enumerate(groups):
        heldout_ids = {u.digest for u in heldout}
        train = tuple(u for u in units if u.digest not in heldout_ids)
        lookup = {u.digest: u for u in train}
        available_segments = sum(len(u.examples) for u in train)
        available_seconds = sum(wav_seconds[u.digest] for u in train)
        candidates = metadata(train)
        for policy in POLICIES:
            sequence = select_sequence(
                candidates, lambda digest: lookup[digest].examples,
                policy=policy, seed=seed + 8101 + fold,
            )
            for fraction in budgets:
                limit = max(1, math.ceil(len(train) * fraction))
                chosen = sequence[:limit]
                acquired_segments = sum(len(lookup[digest].examples) for digest in chosen)
                acquired_seconds = sum(wav_seconds[digest] for digest in chosen)
                rows.append({
                    "fold": fold,
                    "policy": policy,
                    "budget_fraction": fraction,
                    "acquired_wavs": len(chosen),
                    "available_wavs": len(train),
                    "acquired_eligible_segments": acquired_segments,
                    "available_eligible_segments": available_segments,
                    "acquired_segment_fraction": round(
                        acquired_segments / available_segments, 6
                    ),
                    "acquired_wav_duration_seconds": round(acquired_seconds, 3),
                    "available_wav_duration_seconds": round(available_seconds, 3),
                    "acquired_duration_fraction": round(
                        acquired_seconds / available_seconds, 6
                    ),
                })
    summaries = []
    for fraction in budgets:
        for policy in POLICIES:
            samples = [
                row for row in rows
                if row["budget_fraction"] == fraction and row["policy"] == policy
            ]
            summaries.append({
                "budget_fraction": fraction,
                "policy": policy,
                "mean_segment_fraction": round(mean(
                    row["acquired_segment_fraction"] for row in samples
                ), 6),
                "mean_duration_fraction": round(mean(
                    row["acquired_duration_fraction"] for row in samples
                ), 6),
                "min_duration_fraction": round(min(
                    row["acquired_duration_fraction"] for row in samples
                ), 6),
                "max_duration_fraction": round(max(
                    row["acquired_duration_fraction"] for row in samples
                ), 6),
                "per_fold": [
                    {"fold": row["fold"],
                     "wav_count": row["acquired_wavs"],
                     "segment_fraction": row["acquired_segment_fraction"],
                     "duration_fraction": row["acquired_duration_fraction"]}
                    for row in samples
                ],
            })
    segment_counts = [len(u.examples) for u in units]
    seconds = list(wav_seconds.values())
    return {
        "version": VERSION,
        "base_protocol": "0.2-disjoint-wav-cv",
        "split_seed": seed,
        "unique_eligible_wavs": len(units),
        "eligible_oto_acoustic_segments": sum(segment_counts),
        "median_eligible_segments_per_wav": median(segment_counts),
        "max_eligible_segments_per_wav": max(segment_counts),
        "wavs_with_multiple_eligible_segments": sum(v > 1 for v in segment_counts),
        "total_eligible_wav_duration_seconds": round(sum(seconds), 3),
        "median_eligible_wav_duration_seconds": round(median(seconds), 3),
        "rows": rows,
        "summary": summaries,
        "limitations": [
            "WAV duration is total physical audio length, including other phones, silence and padding.",
            "This is a cost audit of original equal-WAV-count selections; it is not an equal-duration-budget experiment.",
            "OTO-derived eligible segment counts are deduplicated target acoustic observations, not total phone inventory.",
            "OTO alias counts may contain aliases for the same acoustic event; no assumption of independence is made.",
            "Selection strategies, frozen folds and acoustic scores remain exactly those used in v0.2.",
            "Selected physical WAVs are indivisible: all contained phonetic observations are acquired together.",
        ],
    }


def audit_cost_manifest(manifest_path: Path, output_path: Path) -> dict:
    data = json.loads(manifest_path.expanduser().resolve().read_text(encoding="utf-8"))
    manifest = validate_manifest(data)
    banks = []
    for bank in manifest:
        root = bank["root"]
        result = {"bank_id": bank["bank_id"], "speaker_group_id": bank["speaker_group_id"]}
        try:
            units, extraction = load_recording_units(root)
            if len(units) < 20:
                raise ValueError("fewer than 20 eligible WAVs")
            durations = eligible_wav_durations(root, units)
            result.update(audit_recording_costs(units, durations))
            result["oto_entries"] = extraction["oto_entries"]
            result["status"] = "audited"
        except (ValueError, OSError, UnicodeError) as exc:
            result["status"] = "excluded"
            result["reason"] = f"{type(exc).__name__}: {exc}"
        banks.append(result)
    report = {
        "experiment": "oto_segment_yield_and_wav_duration_cost_audit",
        "version": VERSION,
        "banks": banks,
        "audited_banks": sum(row["status"] == "audited" for row in banks),
        "excluded_banks": sum(row["status"] == "excluded" for row in banks),
        "paths_and_audio_excluded_from_output": True,
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
