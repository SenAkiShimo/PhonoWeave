"""Mechanism ablations for the frozen recording-budget adaptive selector.

Ablations disable existing priority terms without modifying their original
weights or the default selector. No oracle or unacquired acoustic values enter
selection. Scores always use the unmodified held-out acoustic metric.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .recording_budget_benchmark import (
    BUDGETS, CandidateMetadata, Example, RecordingUnit,
    acoustic_distance, evaluate_heldout, load_recording_units, metadata,
    select_sequence,
)
from .recording_budget_disjoint_cv import PARTITION_SEED, NUMBER_OF_FOLDS, disjoint_folds

ABLATIONS = (
    "full_adaptive",
    "no_acoustic_dispersion",
    "no_final_novelty",
    "metadata_balancing_only",
)
EPSILON = 1e-8
VERSION = "0.4-mechanism-ablation"


def ablated_sequence(
    candidates: tuple[CandidateMetadata, ...],
    acquire,
    *,
    variant: str,
    seed: int,
) -> tuple[str, ...]:
    """Repeat frozen priority logic with only named terms zeroed.

    Default/full policy delegates to the original algorithm, allowing direct
    bit-for-bit comparison against v0.2.
    """
    if variant not in ABLATIONS:
        raise ValueError("unsupported ablation")
    if variant == "full_adaptive":
        return select_sequence(candidates, acquire, policy="adaptive_observed", seed=seed)
    if len({c.identifier for c in candidates}) != len(candidates):
        raise ValueError("duplicate candidate identifiers")
    remaining = {c.identifier: c for c in candidates}
    acquired: list[Example] = []
    cell_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    selected: list[str] = []
    include_dispersion = variant == "no_final_novelty"
    include_final = variant == "no_acoustic_dispersion"
    while remaining:
        dispersion: dict[str, float] = {}
        if include_dispersion:
            samples_by_cell: dict[str, list[Example]] = defaultdict(list)
            for item in acquired:
                samples_by_cell[item.cell].append(item)
            for cell, samples in samples_by_cell.items():
                if len(samples) >= 2:
                    distances = [
                        acoustic_distance(samples[i], samples[j])
                        for i in range(len(samples))
                        for j in range(i + 1, len(samples))
                    ]
                    dispersion[cell] = float(np.median(distances))
        def priority(candidate: CandidateMetadata) -> float:
            values = []
            for cell in candidate.cells:
                score = 1.0 / (1.0 + cell_counts[cell])
                if include_dispersion:
                    d = dispersion.get(cell, 0.0)
                    score += 0.4 * d / (0.3 + d) if d > 0 else 0.0
                if include_final:
                    score += 0.08 * sum(
                        final_counts[f] == 0
                        for f in candidate.finals if f.startswith(cell + ":")
                    )
                values.append(score)
            return max(values, default=0.0)
        chosen = min(remaining, key=lambda k: (-priority(remaining[k]), k))
        item = remaining.pop(chosen)
        selected.append(chosen)
        acquired.extend(acquire(chosen))
        cell_counts.update(set(item.cells))
        final_counts.update(set(item.finals))
    return tuple(selected)


def evaluate_mechanism_ablations(
    units: tuple[RecordingUnit, ...],
    *,
    seed: int = PARTITION_SEED,
    n_folds: int = NUMBER_OF_FOLDS,
    budgets: tuple[float, ...] = BUDGETS,
) -> dict:
    if not budgets or any(not 0 < b <= 1 for b in budgets):
        raise ValueError("invalid budget fractions")
    folds = disjoint_folds(units, seed=seed, n_folds=n_folds)
    rows = []
    split_ids = []
    for fold, heldout in enumerate(folds):
        excluded = {u.digest for u in heldout}
        train = tuple(u for u in units if u.digest not in excluded)
        lookup = {u.digest: u for u in train}
        candidates = metadata(train)
        split_ids.append({"fold": fold, "heldout_sha256": sorted(excluded)})
        sequences = {}
        for variant in ABLATIONS:
            sequences[variant] = ablated_sequence(
                candidates, lambda digest: lookup[digest].examples,
                variant=variant, seed=seed + 8101 + fold,
            )
        fixed = select_sequence(
            candidates, lambda digest: lookup[digest].examples,
            policy="fixed_balanced", seed=seed + 8101 + fold,
        )
        if sequences["metadata_balancing_only"] != fixed:
            raise AssertionError("metadata-only ablation must match frozen fixed-balanced policy")
        for fraction in budgets:
            amount = max(1, math.ceil(len(train) * fraction))
            for variant in ABLATIONS:
                selected = tuple(lookup[key] for key in sequences[variant][:amount])
                score = evaluate_heldout(selected, heldout)
                rows.append({
                    "fold": fold, "budget_fraction": fraction,
                    "budget_wav": amount, "variant": variant,
                    "distance": score["penalized_acoustic_distance"],
                    "cell_coverage": score["cell_coverage"],
                })
    summary = []
    for fraction in budgets:
        baseline = [
            row["distance"] for row in rows
            if row["budget_fraction"] == fraction and row["variant"] == "full_adaptive"
        ]
        for variant in ABLATIONS:
            subset = [
                row["distance"] for row in rows
                if row["budget_fraction"] == fraction and row["variant"] == variant
            ]
            diffs = [a - b for a, b in zip(subset, baseline, strict=True)]
            summary.append({
                "budget_fraction": fraction, "variant": variant,
                "mean_distance": round(float(np.mean(subset)), 5),
                "minus_full_adaptive": round(float(np.mean(diffs)), 6),
                "ablation_better_folds": sum(d < -EPSILON for d in diffs),
                "ablation_worse_folds": sum(d > EPSILON for d in diffs),
                "ties": sum(abs(d) <= EPSILON for d in diffs),
                "per_fold": [
                    {"fold": fold, "distance": value, "minus_full_adaptive": round(delta, 5)}
                    for fold, (value, delta) in enumerate(zip(subset, diffs, strict=True))
                ],
            })
    return {
        "experiment": "recording_budget_mechanism_ablation",
        "version": VERSION,
        "parent_protocol": "0.2-disjoint-wav-cv",
        "partition_seed": seed,
        "test_folds": n_folds,
        "budgets": list(budgets),
        "variants": list(ABLATIONS),
        "heldout_hashes_by_fold": split_ids,
        "rows": rows,
        "summary": summary,
        "limitations": [
            "Single pre-existing voicebank and unchanged acoustic proxy; not synthetic singing quality.",
            "Metadata-only variant is exactly frozen fixed_balanced: no new baseline is claimed.",
            "Disabling an algorithmic term after observing v0.3 is exploratory, not confirmatory.",
            "The final-novelty term uses candidate phonetic metadata available before recording.",
            "Removing acoustic dispersion also removes all access to acquired waveform features for ranking.",
            "Overlapping training pools and one speaker preclude independent speaker-level inference.",
            "No algorithm weights, feature transforms, or evaluation metrics are tuned.",
        ],
    }


def run_voicebank_mechanism_ablation(root: Path, output: Path) -> dict:
    units, extraction = load_recording_units(root)
    result = evaluate_mechanism_ablations(units)
    result["extraction"] = extraction
    path = output.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
