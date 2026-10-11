"""Frozen-policy, equal-WAV-duration-ceiling retrospective benchmark.

Each recording is an indivisible OTO-bearing WAV. Policies rank candidates
using the frozen v0.2 logic, without receiving duration information. The
duration budget truncates each *ordered prefix* at its first non-fitting WAV;
later recordings are not searched to fill the leftover time.

This is a post-hoc comparison using realized full-WAV durations, not a
prospective predictor of unknown future recording times or synthesis quality.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from .recording_budget_benchmark import (
    BUDGETS, POLICIES, RecordingUnit, evaluate_heldout,
    load_recording_units, metadata, select_sequence,
)
from .recording_budget_cost_audit import eligible_wav_durations
from .recording_budget_cross_voicebank import validate_manifest
from .recording_budget_disjoint_cv import (
    NUMBER_OF_FOLDS, PARTITION_SEED, disjoint_folds,
)
from .recording_budget_random_sensitivity import SEED_STRIDE

VERSION = "0.7-equal-duration-ceiling"
DEFAULT_RANDOM_REPEATS = 100
TOLERANCE_SECONDS = 1e-8


def duration_limited_prefix(
    order: tuple[str, ...], durations: dict[str, float], ceiling_seconds: float,
) -> tuple[tuple[str, ...], float, float | None]:
    """Select only consecutive recording files fitting within a shared ceiling.

    Return the selected IDs, spent seconds and first blocked WAV length.
    No skip/backfill or fractional-WAV acquisition is permitted.
    """
    if not math.isfinite(ceiling_seconds) or ceiling_seconds < 0:
        raise ValueError("duration ceiling must be finite and nonnegative")
    if len(order) != len(set(order)):
        raise ValueError("duplicate recording in acquisition sequence")
    picked = []
    spent = 0.0
    blocked = None
    for digest in order:
        if digest not in durations:
            raise ValueError("missing recording duration")
        length = float(durations[digest])
        if not math.isfinite(length) or length <= 0:
            raise ValueError("recording duration must be positive and finite")
        if spent + length > ceiling_seconds + TOLERANCE_SECONDS:
            blocked = length
            break
        spent += length
        picked.append(digest)
    if spent > ceiling_seconds + TOLERANCE_SECONDS:
        raise AssertionError("duration ceiling violated")
    return tuple(picked), spent, blocked


def _duration_row(
    order: tuple[str, ...],
    lookup: dict[str, RecordingUnit],
    duration_seconds: dict[str, float],
    heldout: tuple[RecordingUnit, ...],
    ceiling_seconds: float,
    available_seconds: float,
    available_segments: int,
) -> dict:
    acquired, spent, blocked = duration_limited_prefix(
        order, duration_seconds, ceiling_seconds
    )
    selected = tuple(lookup[key] for key in acquired)
    scores = evaluate_heldout(selected, heldout)
    segment_count = sum(len(unit.examples) for unit in selected)
    return {
        "acquired_wavs": len(selected),
        "acquired_eligible_segments": segment_count,
        "available_eligible_segments": available_segments,
        "acquired_segment_fraction": round(segment_count / available_segments, 6),
        "spent_seconds": round(spent, 6),
        "available_seconds": round(available_seconds, 6),
        "spent_fraction": round(spent / available_seconds, 6),
        "budget_unspent_seconds": round(max(0.0, ceiling_seconds - spent), 6),
        "next_blocked_wav_seconds": (
            round(blocked, 6) if blocked is not None else None
        ),
        "heldout_penalized_distance": scores["penalized_acoustic_distance"],
        "heldout_matched_only_distance": scores["matched_only_acoustic_distance"],
        "heldout_cell_coverage": scores["cell_coverage"],
        "heldout_wav_cell_pairs": scores["heldout_wav_cell_pairs"],
        "matched_wav_cell_pairs": scores["matched_wav_cell_pairs"],
    }


def evaluate_duration_budget(
    units: tuple[RecordingUnit, ...],
    duration_seconds: dict[str, float],
    *,
    budgets: tuple[float, ...] = BUDGETS,
    n_random: int = DEFAULT_RANDOM_REPEATS,
    seed: int = PARTITION_SEED,
    n_folds: int = NUMBER_OF_FOLDS,
) -> dict:
    """Run frozen acquisition orders with matched, non-overrunning time caps."""
    if type(n_random) is not int or n_random < 2:
        raise ValueError("random repeats must be an integer >= 2")
    if not budgets or len(set(budgets)) != len(budgets) or any(
        isinstance(b, bool) or not isinstance(b, (int, float))
        or not math.isfinite(b) or not 0 < b <= 1 for b in budgets
    ):
        raise ValueError("invalid or duplicate budget fractions")
    if len({u.digest for u in units}) != len(units):
        raise ValueError("duplicate recording identifiers")
    if set(duration_seconds) != {u.digest for u in units}:
        raise ValueError("duration index must match eligible WAV groups")
    if any(not isinstance(x, (int, float)) or isinstance(x, bool)
           or not math.isfinite(x) or x <= 0 for x in duration_seconds.values()):
        raise ValueError("durations must be finite and positive")

    groups = disjoint_folds(units, seed=seed, n_folds=n_folds)
    fold_reports = []
    for fold, held in enumerate(groups):
        held_ids = {u.digest for u in held}
        train = tuple(u for u in units if u.digest not in held_ids)
        if not train:
            raise ValueError("training fold empty")
        lookup = {u.digest: u for u in train}
        available_seconds = sum(duration_seconds[u.digest] for u in train)
        available_segments = sum(len(u.examples) for u in train)
        candidates = metadata(train)
        acquisition_seed = seed + 8101 + fold
        ordered = {
            policy: select_sequence(
                candidates, lambda digest: lookup[digest].examples,
                policy=policy, seed=acquisition_seed,
            )
            for policy in POLICIES
        }
        random_sequences = [
            select_sequence(
                candidates, lambda digest: lookup[digest].examples,
                policy="stratified_random",
                seed=acquisition_seed + i * SEED_STRIDE,
            )
            for i in range(n_random)
        ]
        if ordered["stratified_random"] != random_sequences[0]:
            raise AssertionError("v0.3 seed-zero random sequence mismatch")
        fold_rows = []
        for fraction in budgets:
            ceiling = available_seconds * fraction
            policy_results = {
                policy: _duration_row(
                    sequence, lookup, duration_seconds, held,
                    ceiling, available_seconds, available_segments,
                )
                for policy, sequence in ordered.items()
            }
            null = [
                _duration_row(
                    sequence, lookup, duration_seconds, held,
                    ceiling, available_seconds, available_segments,
                )
                for sequence in random_sequences
            ]
            if null[0] != policy_results["stratified_random"]:
                raise AssertionError("seed-zero evaluation not reproduced")
            for policy, row in policy_results.items():
                if row["spent_seconds"] > ceiling + 1e-5:
                    raise AssertionError("duration budget exceeded")
                if fraction == 1.0 and (
                    row["acquired_wavs"] != len(train)
                    or row["acquired_eligible_segments"] != available_segments
                ):
                    raise AssertionError("full duration ceiling must acquire all WAVs")
            fold_rows.append({
                "budget_fraction": fraction,
                "budget_ceiling_seconds": round(ceiling, 6),
                "policies": policy_results,
                "random_repetitions": null,
            })
        fold_reports.append({
            "fold": fold,
            "train_unique_wavs": len(train),
            "test_unique_wavs": len(held),
            "training_eligible_segments": available_segments,
            "training_total_wav_seconds": round(available_seconds, 6),
            "test_cells_absent_from_training": sorted(
                {e.cell for u in held for e in u.examples}
                - {e.cell for u in train for e in u.examples}
            ),
            "budgets": fold_rows,
        })

    totals = []
    for fraction in budgets:
        items = [
            next(r for r in fold["budgets"] if r["budget_fraction"] == fraction)
            for fold in fold_reports
        ]
        def metric(policy: str, field: str) -> float:
            return float(mean(row["policies"][policy][field] for row in items))
        fixed_distance = metric("fixed_balanced", "heldout_penalized_distance")
        adaptive_distance = metric("adaptive_observed", "heldout_penalized_distance")
        random_scores = [
            float(mean(
                row["random_repetitions"][index]["heldout_penalized_distance"]
                for row in items
            ))
            for index in range(n_random)
        ]
        random_distances = np.asarray(random_scores, dtype=float)
        better = int(np.sum(random_distances < adaptive_distance - 1e-8))
        equal = int(np.sum(np.abs(random_distances - adaptive_distance) <= 1e-8))
        totals.append({
            "budget_fraction": fraction,
            "fixed_distance": round(fixed_distance, 6),
            "adaptive_distance": round(adaptive_distance, 6),
            "random_seed0_distance": round(random_scores[0], 6),
            "random_mean_distance": round(float(np.mean(random_distances)), 6),
            "random_p05_distance": round(float(np.percentile(random_distances, 5)), 6),
            "random_p95_distance": round(float(np.percentile(random_distances, 95)), 6),
            "random_draws_better_than_adaptive": better,
            "random_draws_tied_to_adaptive": equal,
            "random_draws_worse_than_adaptive": n_random - better - equal,
            "adaptive_minus_fixed": round(adaptive_distance - fixed_distance, 6),
            "adaptive_minus_random_mean": round(
                adaptive_distance - float(np.mean(random_distances)), 6
            ),
            "fixed_mean_spent_fraction": round(metric("fixed_balanced", "spent_fraction"), 6),
            "adaptive_mean_spent_fraction": round(metric("adaptive_observed", "spent_fraction"), 6),
            "random_seed0_mean_spent_fraction": round(metric("stratified_random", "spent_fraction"), 6),
            "adaptive_mean_acquired_segments": round(metric("adaptive_observed", "acquired_eligible_segments"), 3),
            "fixed_mean_acquired_segments": round(metric("fixed_balanced", "acquired_eligible_segments"), 3),
            "per_fold": [
                {
                    "fold": i,
                    "fixed_distance": row["policies"]["fixed_balanced"]["heldout_penalized_distance"],
                    "adaptive_distance": row["policies"]["adaptive_observed"]["heldout_penalized_distance"],
                    "fixed_spent_fraction": row["policies"]["fixed_balanced"]["spent_fraction"],
                    "adaptive_spent_fraction": row["policies"]["adaptive_observed"]["spent_fraction"],
                    "fixed_eligible_segments": row["policies"]["fixed_balanced"]["acquired_eligible_segments"],
                    "adaptive_eligible_segments": row["policies"]["adaptive_observed"]["acquired_eligible_segments"],
                }
                for i, row in enumerate(items)
            ],
        })
    return {
        "experiment": "equal_duration_ceiling_recording_budget",
        "version": VERSION,
        "source_protocol": "0.2-disjoint-wav-cv",
        "selection_policy_version": "0.2-frozen",
        "seed": seed,
        "n_folds": n_folds,
        "random_repeats_per_fold": n_random,
        "budget_fractions": list(budgets),
        "unique_eligible_wavs": len(units),
        "eligible_oto_acoustic_segments": sum(len(u.examples) for u in units),
        "total_full_wav_seconds": round(sum(duration_seconds.values()), 6),
        "budget_mode": "maximum_fraction_of_training_full_WAV_duration",
        "acquisition_rule": "consecutive_prefix_stop_before_first_overrun_no_skipping",
        "runtime_cost_foreknowledge": "retrospective_header_duration_only_not_used_for_ranking",
        "fold_reports": fold_reports,
        "summary": totals,
        "limitations": [
            "Equal ceilings do not imply exactly equal spent seconds; unused time is reported per policy.",
            "Source WAVs are indivisible; the first non-fitting candidate stops acquisition, with no later backfill.",
            "Realized WAV duration is known only retrospectively; future recordings need a separate prospective time predictor.",
            "Full duration includes rests and unrelated phonemes and is not the total human recording labor.",
            "OTO target counts represent extracted acoustic observations, not independent phoneme inventory.",
            "Frozen acoustic coverage score remains an OTO-conditioned spectral proxy, not synthesis quality.",
            "All WAVs from one singer remain within that singer; disjoint test folds still share training recordings.",
            "Cross-bank result interpretation must weight speakers, not raw OTO aliases or WAVs.",
            "Selection ranking never uses unacquired acoustic features or candidate recording duration.",
            "The study was motivated by an examined prior dataset and is exploratory.",
        ],
    }


def run_duration_manifest(manifest: Path, output: Path, *, repeats: int = DEFAULT_RANDOM_REPEATS) -> dict:
    raw = json.loads(manifest.expanduser().resolve().read_text(encoding="utf-8"))
    banks = validate_manifest(raw)
    results = []
    seen_hashes = {}
    for bank in banks:
        record = {
            "bank_id": bank["bank_id"],
            "speaker_group_id": bank["speaker_group_id"],
        }
        try:
            root = bank["root"]
            if not root.is_dir():
                raise ValueError("missing voicebank folder")
            units, extraction = load_recording_units(root)
            if len(units) < 20:
                raise ValueError("fewer than 20 eligible WAV recordings")
            common = sorted({seen_hashes[u.digest] for u in units if u.digest in seen_hashes})
            if common:
                raise ValueError("exact WAV duplicate with a prior bank: " + ", ".join(common))
            duration = eligible_wav_durations(root, units)
            record.update(evaluate_duration_budget(units, duration, n_random=repeats))
            record["oto_entries"] = extraction["oto_entries"]
            record["status"] = "evaluated"
            for unit in units:
                seen_hashes[unit.digest] = bank["bank_id"]
        except (ValueError, OSError, UnicodeError, KeyError, IndexError) as error:
            record["status"] = "excluded"
            record["exclusion_reason"] = f"{type(error).__name__}: {error}"
        results.append(record)
    active = [bank for bank in results if bank["status"] == "evaluated"]
    by_budget = []
    for budget in BUDGETS:
        rows = [next(x for x in bank["summary"] if x["budget_fraction"] == budget)
                for bank in active]
        by_budget.append({
            "budget_fraction": budget,
            "speaker_groups": len(rows),
            "adaptive_better_than_random_mean": sum(
                x["adaptive_minus_random_mean"] < -1e-8 for x in rows
            ),
            "adaptive_worse_than_random_mean": sum(
                x["adaptive_minus_random_mean"] > 1e-8 for x in rows
            ),
            "adaptive_better_than_fixed": sum(
                x["adaptive_minus_fixed"] < -1e-8 for x in rows
            ),
            "mean_equal_speaker_adaptive_minus_random": (
                round(mean(x["adaptive_minus_random_mean"] for x in rows), 6)
                if rows else None
            ),
            "mean_equal_speaker_adaptive_minus_fixed": (
                round(mean(x["adaptive_minus_fixed"] for x in rows), 6)
                if rows else None
            ),
            "mean_adaptive_spent_fraction": (
                round(mean(x["adaptive_mean_spent_fraction"] for x in rows), 6)
                if rows else None
            ),
            "mean_fixed_spent_fraction": (
                round(mean(x["fixed_mean_spent_fraction"] for x in rows), 6)
                if rows else None
            ),
        })
    report = {
        "experiment": "equal_duration_ceiling_cross_voicebank_replication",
        "version": VERSION,
        "requested_banks": len(banks),
        "evaluated_banks": len(active),
        "excluded_banks": len(banks)-len(active),
        "independent_speaker_groups_declared_not_biometrically_verified": True,
        "results": results,
        "equal_speaker_summary_by_budget": by_budget,
        "sufficient_for_cross_bank_comparison": len(active) >= 2,
        "source_audio_paths_written": False,
    }
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
