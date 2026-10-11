"""Disjoint-WAV cross-validation of frozen recording-budget policies.

This protocol replaces v0.1's highly overlapping repeated holdouts with
one deterministic metadata-stratified five-fold partition. Every unique WAV
belongs to one and only one held-out fold. Acoustic feature vectors are not
consulted by the partitioning procedure.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .recording_budget_benchmark import (
    BUDGETS, FEATURE_SCALE, MISSING_CELL_PENALTY, POLICIES,
    RecordingUnit, evaluate_heldout, load_recording_units, metadata,
    select_sequence,
)

PROTOCOL_VERSION = "0.2-disjoint-wav-cv"
NUMBER_OF_FOLDS = 5
PARTITION_SEED = 20261011


def disjoint_folds(
    units: tuple[RecordingUnit, ...],
    *, n_folds: int = NUMBER_OF_FOLDS, seed: int = PARTITION_SEED,
) -> tuple[tuple[RecordingUnit, ...], ...]:
    """Partition WAVs once, using metadata rather than evaluation features.

    Rare phonetic cells are allocated early; fold sizes differ by at most
    one. A singleton cell necessarily lacks a training example in the fold
    containing it. Such gaps are retained and counted in evaluation.
    """
    if not 2 <= n_folds <= len(units):
        raise ValueError("need between two and number-of-WAVs folds")
    if len({u.digest for u in units}) != len(units):
        raise ValueError("duplicate WAV SHA-256 groups")
    if any(not u.examples for u in units):
        raise ValueError("WAV group without usable consonant examples")
    rng = np.random.default_rng(seed)
    totals = Counter(cell for unit in units for cell in unit.cells)
    order = list(map(int, rng.permutation(len(units))))
    seed_rank = {idx: n for n, idx in enumerate(order)}
    order.sort(key=lambda i: (
        -sum(1.0 / totals[cell] for cell in units[i].cells),
        -len(units[i].cells), seed_rank[i],
    ))
    fold_rank = list(map(int, rng.permutation(n_folds)))
    rank = {idx: n for n, idx in enumerate(fold_rank)}
    capacities = [
        len(units) // n_folds + (i < len(units) % n_folds)
        for i in range(n_folds)
    ]
    # Randomize which fold receives an extra WAV when the corpus size is
    # not divisible by the fold count.
    capacities = [capacities[rank[f]] for f in range(n_folds)]
    assigned: list[list[int]] = [[] for _ in range(n_folds)]
    counts: list[Counter] = [Counter() for _ in range(n_folds)]
    for index in order:
        unit = units[index]
        candidates = [f for f in range(n_folds) if len(assigned[f]) < capacities[f]]
        if not candidates:
            raise ValueError("partition capacity exhausted")
        # Balance metadata-cell shares first and fold load second.
        # This never reads Example.features.
        target_fold = min(candidates, key=lambda f: (
            sum(counts[f][cell] / totals[cell] for cell in unit.cells),
            len(assigned[f]) / capacities[f],
            rank[f],
        ))
        assigned[target_fold].append(index)
        counts[target_fold].update(unit.cells)
    output = tuple(tuple(units[i] for i in sorted(bucket)) for bucket in assigned)
    combined = [unit.digest for bucket in output for unit in bucket]
    if len(combined) != len(units) or len(set(combined)) != len(units):
        raise AssertionError("WAVs must occur once across disjoint test folds")
    return output


def run_disjoint_benchmark(
    units: tuple[RecordingUnit, ...],
    *,
    n_folds: int = NUMBER_OF_FOLDS,
    seed: int = PARTITION_SEED,
    budgets: tuple[float, ...] = BUDGETS,
) -> dict:
    """Compare unchanged v0.1 selectors on a non-overlapping test partition."""
    if any(not 0 < b <= 1 for b in budgets) or not budgets:
        raise ValueError("invalid budget fractions")
    groups = disjoint_folds(units, n_folds=n_folds, seed=seed)
    split_records = []
    rows = []
    for fold, heldout in enumerate(groups):
        test_hashes = {u.digest for u in heldout}
        train = tuple(unit for unit in units if unit.digest not in test_hashes)
        if not train:
            raise ValueError("fold has no training recordings")
        test_cells = {cell for unit in heldout for cell in unit.cells}
        train_cells = {cell for unit in train for cell in unit.cells}
        split_records.append({
            "fold": fold,
            "training_sha256": sorted(u.digest for u in train),
            "heldout_sha256": sorted(test_hashes),
            "train_unique_wavs": len(train),
            "heldout_unique_wavs": len(heldout),
            "train_cells": sorted(train_cells),
            "heldout_cells": sorted(test_cells),
            "test_cells_absent_from_training": sorted(test_cells - train_cells),
        })
        train_map = {unit.digest: unit for unit in train}
        candidates = metadata(train)
        for policy in POLICIES:
            sequence = select_sequence(
                candidates, lambda digest: train_map[digest].examples,
                policy=policy, seed=seed + 8101 + fold,
            )
            for fraction in budgets:
                count = max(1, math.ceil(fraction * len(train)))
                acquired = tuple(train_map[digest] for digest in sequence[:count])
                scores = evaluate_heldout(acquired, heldout)
                rows.append({
                    "fold": fold,
                    "seed": seed + fold,
                    "policy": policy,
                    "budget_fraction": fraction,
                    "budget_wav": count,
                    "train_available_wav": len(train),
                    **scores,
                })
    comparisons = []
    summary = []
    for fraction in budgets:
        for policy in POLICIES:
            subset = [
                row for row in rows
                if row["policy"] == policy and row["budget_fraction"] == fraction
            ]
            summary.append({
                "policy": policy,
                "budget_fraction": fraction,
                "folds": len(subset),
                "mean_cell_coverage": round(float(np.mean([
                    row["cell_coverage"] for row in subset
                    if row["cell_coverage"] is not None
                ])), 5),
                "mean_penalized_acoustic_distance": round(float(np.mean([
                    row["penalized_acoustic_distance"] for row in subset
                    if row["penalized_acoustic_distance"] is not None
                ])), 5),
            })
        for baseline in ("fixed_balanced", "stratified_random"):
            pairs = []
            for fold in range(n_folds):
                b = next(row for row in rows if
                         row["fold"] == fold and row["budget_fraction"] == fraction
                         and row["policy"] == baseline)
                a = next(row for row in rows if
                         row["fold"] == fold and row["budget_fraction"] == fraction
                         and row["policy"] == "adaptive_observed")
                delta = round(
                    a["penalized_acoustic_distance"] - b["penalized_acoustic_distance"], 5
                )
                pairs.append({"fold": fold, "delta": delta,
                              "baseline_distance": b["penalized_acoustic_distance"],
                              "adaptive_distance": a["penalized_acoustic_distance"]})
            comparisons.append({
                "budget_fraction": fraction, "baseline": baseline,
                "adaptive_wins": sum(p["delta"] < -1e-8 for p in pairs),
                "adaptive_losses": sum(p["delta"] > 1e-8 for p in pairs),
                "ties": sum(abs(p["delta"]) <= 1e-8 for p in pairs),
                "mean_paired_delta": round(float(np.mean([p["delta"] for p in pairs])), 6),
                "per_fold": pairs,
            })
    test_digest_counts = Counter(
        digest for row in split_records for digest in row["heldout_sha256"]
    )
    if len(test_digest_counts) != len(units) or any(n != 1 for n in test_digest_counts.values()):
        raise AssertionError("disjoint test partition violated")
    return {
        "experiment": "recording_budget_vs_heldout_acoustic_coverage",
        "version": PROTOCOL_VERSION,
        "parent_protocol": "0.1-retrospective",
        "source_audio_grouping": "unique_wav_sha256",
        "scope": "Mandarin s/sh/x fricative OTO consonant core",
        "partition": {
            "method": "single_metadata_stratified_disjoint_wav_partition",
            "seed": seed, "folds": n_folds,
            "all_unique_wavs_tested_exactly_once": True,
            "distinct_heldout_wavs": len(test_digest_counts),
            "test_reuse_count": sum(count > 1 for count in test_digest_counts.values()),
            "fold_test_sizes": [len(group) for group in groups],
            "folds_with_missing_training_cells": sum(
                bool(s["test_cells_absent_from_training"]) for s in split_records
            ),
        },
        "budgets": list(budgets),
        "policies": list(POLICIES),
        "fixed_feature_scale": list(FEATURE_SCALE),
        "distance_missing_cell_penalty": MISSING_CELL_PENALTY,
        "splits": split_records,
        "rows": rows,
        "aggregate": summary,
        "paired_comparisons": comparisons,
        "limitations": [
            "Disjoint held-out WAVs reduce cross-fold test reuse but do not establish independent speakers.",
            "Partition uses metadata only; singleton phonetic cells can be absent from training and remain penalized.",
            "v0.2 uses ~20% test folds, so numeric results are not directly comparable to v0.1's 25% overlapping holdouts.",
            "The selector rules and normalized acoustic distance are unchanged from v0.1.",
            "Only acquired WAV acoustic features are visible to the adaptive selector.",
            "WAV-count budgets do not account for recording duration; evaluation is not synthesized-audio quality.",
            "Byte-identical WAV hashes do not detect re-encoded copies or duplicate takes.",
            "All results describe one retrospective voicebank, not cross-singer generalization.",
        ],
    }


def run_voicebank_disjoint_benchmark(root: Path, output: Path) -> dict:
    units, extraction = load_recording_units(root)
    if len(units) < NUMBER_OF_FOLDS:
        raise ValueError("insufficient unique eligible WAV recordings")
    report = run_disjoint_benchmark(units)
    report["extraction"] = extraction
    out = output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    return report
