"""Sham-acoustic mechanism control under frozen v0.8 duration ceilings."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from .recording_budget_benchmark import BUDGETS, load_recording_units, metadata, select_sequence
from .recording_budget_context_control import coverage_first_random_sequence
from .recording_budget_context_fast_eval import PreparedHeldoutDistances
from .recording_budget_cost_audit import eligible_wav_durations
from .recording_budget_cross_voicebank import validate_manifest
from .recording_budget_disjoint_cv import PARTITION_SEED, disjoint_folds
from .recording_budget_random_sensitivity import SEED_STRIDE
from .recording_budget_sham_feedback import (
    incremental_adaptive_sequence, permuted_training_feedback,
)

VERSION = "0.9-within-context-acoustic-permutation"
BASELINE_VERSION = "0.8-context-coverage-random-control"
EPS = 1e-8


def evaluate_sham_feedback(
    units, seconds, reference, *, repeats=100,
):
    if type(repeats) is not int or repeats < 2:
        raise ValueError("requires at least two permutations")
    if reference.get("version") != BASELINE_VERSION:
        raise ValueError("expected v0.8 reference bank")
    if len(units) != reference["unique_eligible_wavs"]:
        raise ValueError("WAV count changed from v0.8")
    if set(seconds) != {u.digest for u in units}:
        raise ValueError("duration index mismatch")
    folds = disjoint_folds(units, seed=PARTITION_SEED)
    if len(folds) != len(reference["folds"]):
        raise ValueError("fold count differs from v0.8")
    out = []
    for fold, held in enumerate(folds):
        test = {u.digest for u in held}
        training = tuple(u for u in units if u.digest not in test)
        lookup = {u.digest: u for u in training}
        meta = metadata(training)
        total_seconds = sum(seconds[u.digest] for u in training)
        total_segments = sum(len(u.examples) for u in training)
        reference_fold = reference["folds"][fold]
        if (reference_fold["fold"] != fold
                or reference_fold["heldout_wavs"] != len(held)
                or reference_fold["training_wavs"] != len(training)):
            raise ValueError("test split differs from v0.8")
        true_sequence = select_sequence(
            meta, lambda identifier: lookup[identifier].examples,
            policy="adaptive_observed", seed=PARTITION_SEED + 8101 + fold,
        )
        cached_sequence = incremental_adaptive_sequence(
            meta, lambda identifier: lookup[identifier].examples
        )
        if true_sequence != cached_sequence:
            raise AssertionError("incremental selector drift from frozen full adaptive")
        evaluator = PreparedHeldoutDistances(lookup, held)
        truth = evaluator.score_duration_budgets(
            true_sequence, seconds, BUDGETS, total_seconds, total_segments
        )
        for b in BUDGETS:
            ref = next(x for x in reference_fold["budgets"] if x["budget_fraction"] == b)
            saved = ref["reference_policies"]["adaptive_observed"]
            current = truth[b]
            for key in ("penalized_distance", "acquired_wavs",
                        "acquired_eligible_segments", "spent_fraction"):
                if not math.isclose(current[key], saved[key], rel_tol=0, abs_tol=2e-6):
                    raise ValueError(f"v0.8 replay mismatch in {key} fold={fold} budget={b}")
        null_scores = {b: [] for b in BUDGETS}
        diagnostics = []
        for repeat in range(repeats):
            seed = PARTITION_SEED + 99001 + fold + repeat * SEED_STRIDE
            permuted, stat = permuted_training_feedback(training, seed=seed)
            sequence = incremental_adaptive_sequence(
                meta, lambda identifier: permuted[identifier]
            )
            scores = evaluator.score_duration_budgets(
                sequence, seconds, BUDGETS, total_seconds, total_segments
            )
            for b in BUDGETS:
                null_scores[b].append(scores[b])
            diagnostics.append({
                "repeat": repeat,
                "seed": seed,
                "same_full_acquisition_order": sequence == true_sequence,
                "changed_feedback_assignments": stat["changed_feature_assignments"],
                "shuffled_training_examples": stat["shuffled_training_examples"],
                "permutable_cells": stat["permutable_cells"],
                "prefix_overlap_by_budget": {
                    str(b): round(
                        len(
                            set(sequence[:scores[b]["acquired_wavs"]])
                            & set(true_sequence[:truth[b]["acquired_wavs"]])
                        ) / max(
                            1,
                            len(
                                set(sequence[:scores[b]["acquired_wavs"]])
                                | set(true_sequence[:truth[b]["acquired_wavs"]])
                            ),
                        ), 6
                    )
                    for b in BUDGETS
                },
            })
        out.append({
            "fold": fold, "heldout_wavs": len(held),
            "training_wavs": len(training),
            "budgets": [
                {
                    "budget_fraction": b,
                    "true_feedback": truth[b],
                    "permuted_feedback_trials": null_scores[b],
                }
                for b in BUDGETS
            ],
            "permutation_diagnostics": diagnostics,
        })
    summary = []
    for b in BUDGETS:
        rows = [next(x for x in fold["budgets"] if x["budget_fraction"] == b) for fold in out]
        true_mean = float(mean(x["true_feedback"]["penalized_distance"] for x in rows))
        draws = np.asarray([
            mean(row["permuted_feedback_trials"][i]["penalized_distance"] for row in rows)
            for i in range(repeats)
        ], dtype=float)
        better = int(np.sum(draws < true_mean - EPS))
        tied = int(np.sum(np.abs(draws - true_mean) <= EPS))
        summary.append({
            "budget_fraction": b,
            "true_mean_distance": round(true_mean, 6),
            "permuted_mean_distance": round(float(np.mean(draws)), 6),
            "permuted_p05": round(float(np.percentile(draws, 5)), 6),
            "permuted_p95": round(float(np.percentile(draws, 95)), 6),
            "true_minus_permuted_mean": round(true_mean - float(np.mean(draws)), 6),
            "permuted_draws_better_than_true": better,
            "permuted_draws_equal_to_true": tied,
            "permuted_draws_worse_than_true": repeats - better - tied,
            "mean_true_spent_fraction": round(mean(
                x["true_feedback"]["spent_fraction"] for x in rows
            ), 6),
            "mean_permuted_spent_fraction": round(mean(
                x["spent_fraction"] for row in rows
                for x in row["permuted_feedback_trials"]
            ), 6),
            "mean_true_missing_penalty": round(mean(
                x["true_feedback"]["missing_penalty_contribution"] for x in rows
            ), 6),
            "mean_permuted_missing_penalty": round(mean(
                x["missing_penalty_contribution"] for row in rows
                for x in row["permuted_feedback_trials"]
            ), 6),
            "identical_full_order_fraction": round(mean(
                x["same_full_acquisition_order"] for fold in out
                for x in fold["permutation_diagnostics"]
            ), 6),
            "mean_changed_feedback_assignments": round(mean(
                x["changed_feedback_assignments"] for fold in out
                for x in fold["permutation_diagnostics"]
            ), 3),
        })
    return {
        "version": VERSION,
        "reference_protocol": BASELINE_VERSION,
        "unique_eligible_wavs": len(units),
        "permutations_per_fold": repeats,
        "folds": out,
        "summary": summary,
        "limitations": [
            "Training-only per-cell feedback permutations are created offline and are not a deployable data collection policy.",
            "Evaluation and held-out features always remain genuine and unchanged.",
            "Each WAV remains an indivisible acquisition unit under frozen full-duration ceilings.",
            "Permutation preserves within-cell feature distributions, metadata and waveform segment counts, not true identity linkage.",
            "The selector observes shuffled feedback only after a WAV is selected.",
            "Selection ranking may be invariant under many permutations; order overlap is separately reported.",
            "Repeated sham draws are not independent singers and these empirical quantiles are not across-speaker confidence intervals.",
            "Audio-duration ceilings do not guarantee exactly equal realized recorded seconds.",
            "Acoustic proxy coverage is not singing-synthesis quality.",
            "Post-hoc exploratory study following examination of the eight-bank v0.8 outcomes.",
        ],
    }


def run_manifest(manifest: Path, reference: Path, output: Path, *, repeats: int = 100) -> dict:
    banks = validate_manifest(json.loads(manifest.expanduser().resolve().read_text("utf-8")))
    old = json.loads(reference.expanduser().resolve().read_text("utf-8"))
    if old.get("version") != BASELINE_VERSION:
        raise ValueError("expected v0.8 reference JSON")
    by_id = {x["bank_id"]: x for x in old["results"]}
    if set(by_id) != {bank["bank_id"] for bank in banks}:
        raise ValueError("manifest and v0.8 reference banks differ")
    rows = []
    for bank in banks:
        units, _ = load_recording_units(bank["root"])
        duration = eligible_wav_durations(bank["root"], units)
        record = evaluate_sham_feedback(units, duration, by_id[bank["bank_id"]], repeats=repeats)
        record["bank_id"] = bank["bank_id"]
        record["speaker_group_id"] = bank["speaker_group_id"]
        rows.append(record)
    macro = []
    for b in BUDGETS:
        records = [next(r for r in bank["summary"] if r["budget_fraction"] == b) for bank in rows]
        macro.append({
            "budget_fraction": b,
            "evaluated_speakers": len(records),
            "real_better_than_permuted_mean": sum(x["true_minus_permuted_mean"] < -EPS for x in records),
            "real_worse_than_permuted_mean": sum(x["true_minus_permuted_mean"] > EPS for x in records),
            "ties": sum(abs(x["true_minus_permuted_mean"]) <= EPS for x in records),
            "equal_speaker_mean_true_minus_permuted": round(mean(
                x["true_minus_permuted_mean"] for x in records
            ), 6),
        })
    report = {
        "experiment": "training_only_within_context_sham_acoustic_feedback",
        "version": VERSION,
        "reference_version": BASELINE_VERSION,
        "speaker_groups": len(rows),
        "permutations_per_fold": repeats,
        "banks": rows,
        "equal_speaker_summary": macro,
        "audio_or_paths_included": False,
    }
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
