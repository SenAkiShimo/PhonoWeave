"""Exploratory random-null sensitivity on a frozen disjoint WAV partition.

All three acquisition policies, acoustic features and evaluation distances
are imported unchanged from the v0.2 experiment. Different random sequences
are replicates of an acquisition policy, NOT independent evaluation sets.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from .recording_budget_benchmark import (
    BUDGETS, RecordingUnit, evaluate_heldout, load_recording_units,
    metadata, select_sequence,
)
from .recording_budget_disjoint_cv import (
    PARTITION_SEED, NUMBER_OF_FOLDS, disjoint_folds,
)

VERSION = "0.3-random-baseline-sensitivity"
N_RANDOM = 100
SEED_STRIDE = 104729


def evaluate_random_sensitivity(
    units: tuple[RecordingUnit, ...],
    *,
    n_random: int = N_RANDOM,
    partition_seed: int = PARTITION_SEED,
    n_folds: int = NUMBER_OF_FOLDS,
    budgets: tuple[float, ...] = BUDGETS,
) -> dict:
    if isinstance(n_random, bool) or not isinstance(n_random, int) or n_random < 2:
        raise ValueError("at least two random acquisitions are required")
    if not budgets or any(not 0 < b <= 1 for b in budgets):
        raise ValueError("invalid budgets")
    groups = disjoint_folds(units, n_folds=n_folds, seed=partition_seed)
    folds = []
    for fold, held in enumerate(groups):
        train_hashes = {unit.digest for unit in held}
        train = tuple(unit for unit in units if unit.digest not in train_hashes)
        mapping = {u.digest: u for u in train}
        candidates = metadata(train)
        original_seed = partition_seed + 8101 + fold
        fixed_sequence = select_sequence(
            candidates, lambda digest: mapping[digest].examples, policy="fixed_balanced",
            seed=original_seed,
        )
        adaptive_sequence = select_sequence(
            candidates, lambda digest: mapping[digest].examples, policy="adaptive_observed",
            seed=original_seed,
        )
        random_sequences = []
        for repeat in range(n_random):
            current_seed = original_seed + repeat * SEED_STRIDE
            sequence = select_sequence(
                candidates, lambda digest: mapping[digest].examples, policy="stratified_random",
                seed=current_seed,
            )
            random_sequences.append((current_seed, sequence))
        scores = []
        for fraction in budgets:
            amount = max(1, math.ceil(fraction * len(train)))
            def evaluate(sequence):
                selected = tuple(mapping[key] for key in sequence[:amount])
                return evaluate_heldout(selected, held)["penalized_acoustic_distance"]
            fixed = evaluate(fixed_sequence)
            adaptive = evaluate(adaptive_sequence)
            null = np.asarray([evaluate(seq) for _, seq in random_sequences], dtype=float)
            if not np.isfinite(null).all():
                raise ValueError("invalid random distance")
            better = int(np.sum(null < adaptive - 1e-8))
            equal = int(np.sum(np.abs(null - adaptive) <= 1e-8))
            worse = int(np.sum(null > adaptive + 1e-8))
            scores.append({
                "budget_fraction": fraction,
                "budget_wav": amount,
                "fixed_distance": fixed,
                "adaptive_distance": adaptive,
                "random_seed0_distance": float(null[0]),
                "random_mean": float(np.mean(null)),
                "random_median": float(np.median(null)),
                "random_p05": float(np.percentile(null, 5)),
                "random_p95": float(np.percentile(null, 95)),
                "random_min": float(np.min(null)),
                "random_max": float(np.max(null)),
                "random_better_than_adaptive": better,
                "random_equal_to_adaptive": equal,
                "random_worse_than_adaptive": worse,
                "random_distances": [float(d) for d in null],
            })
        folds.append({
            "fold": fold,
            "train_unique_wavs": len(train),
            "test_unique_wavs": len(held),
            "heldout_sha256": sorted(train_hashes),
            "random_seed_schedule": [
                original_seed + repeat * SEED_STRIDE for repeat in range(n_random)
            ],
            "budgets": scores,
        })
    summary = []
    for fraction in budgets:
        items = [next(x for x in fold["budgets"] if x["budget_fraction"] == fraction)
                 for fold in folds]
        fixed = np.asarray([x["fixed_distance"] for x in items])
        adaptive = np.asarray([x["adaptive_distance"] for x in items])
        # Macro-average over the same five held-out folds per random repetition.
        random_by_repeat = np.asarray([
            [x["random_distances"][repeat] for x in items]
            for repeat in range(n_random)
        ], dtype=float)
        random_macro = np.mean(random_by_repeat, axis=1)
        adaptive_macro = float(np.mean(adaptive))
        summary.append({
            "budget_fraction": fraction,
            "fixed_macro_mean": float(np.mean(fixed)),
            "adaptive_macro_mean": adaptive_macro,
            "random_macro_mean": float(np.mean(random_macro)),
            "random_macro_median": float(np.median(random_macro)),
            "random_macro_p05": float(np.percentile(random_macro, 5)),
            "random_macro_p95": float(np.percentile(random_macro, 95)),
            "random_macro_min": float(np.min(random_macro)),
            "random_macro_max": float(np.max(random_macro)),
            "random_macro_seed0": float(random_macro[0]),
            "random_repeats_better_than_adaptive": int(np.sum(random_macro < adaptive_macro - 1e-8)),
            "random_repeats_equal_to_adaptive": int(np.sum(np.abs(random_macro - adaptive_macro) <= 1e-8)),
            "random_repeats_worse_than_adaptive": int(np.sum(random_macro > adaptive_macro + 1e-8)),
            "per_fold": [
                {"fold": fold["fold"], "fixed": item["fixed_distance"],
                 "adaptive": item["adaptive_distance"],
                 "random_mean": item["random_mean"],
                 "random_better_than_adaptive": item["random_better_than_adaptive"],
                 "random_equal_to_adaptive": item["random_equal_to_adaptive"],
                 "random_worse_than_adaptive": item["random_worse_than_adaptive"]}
                for fold, item in zip(folds, items, strict=True)
            ],
        })
    return {
        "experiment": "random_acquisition_sensitivity",
        "version": VERSION,
        "base_protocol": "0.2-disjoint-wav-cv",
        "partition_seed": partition_seed,
        "folds": n_folds,
        "random_repetitions_per_fold": n_random,
        "random_seed_stride": SEED_STRIDE,
        "unique_heldout_wavs": len({
            h for fold in folds for h in fold["heldout_sha256"]
        }),
        "random_seed0_reproduces_v0_2": True,
        "fold_results": folds,
        "summary": summary,
        "limitations": [
            "Random repetitions share the same underlying recorded WAVs and held-out folds.",
            "Percentile bounds represent random acquisition variability conditional on this partition, not uncertainty across speakers.",
            "One macro draw combines the same-index random sequence from each fold; these draws do not constitute independent singers.",
            "Selection rules, objective, feature normalization and disjoint folds are frozen from v0.2.",
            "This sensitivity experiment was designed after observing the v0.2 results and is exploratory.",
            "No time-cost weighting, natural fixed-reclist baseline, or synthesized-audio evaluation is included.",
        ],
    }


def run_voicebank_random_sensitivity(root: Path, output: Path, *, n_random: int = N_RANDOM) -> dict:
    units, extraction = load_recording_units(root)
    report = evaluate_random_sensitivity(units, n_random=n_random)
    report["extraction"] = extraction
    out = output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report
