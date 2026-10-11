"""Paired, leakage-aware diagnostics for a frozen recording-budget experiment.

This module reads a completed v0.1 JSON report. It never loads WAV audio,
reselects recordings, changes acoustic metrics or computes inferential
confidence intervals from correlated folds.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path

REQUIRED_POLICIES = ("fixed_balanced", "stratified_random", "adaptive_observed")
COMPARISONS = ("fixed_balanced", "stratified_random")
ADAPTIVE = "adaptive_observed"
EPSILON = 1e-8


def _finite_score(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not a numeric distance")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} is not a finite nonnegative distance")
    return result


def analyze_budget_report(payload: dict) -> dict:
    """Compute paired fold effects; summarize repeated WAV overlap separately."""
    if payload.get("version") != "0.1-retrospective":
        raise ValueError("expected recording-budget v0.1 report")
    rows, splits = payload.get("rows"), payload.get("splits")
    if not isinstance(rows, list) or not isinstance(splits, list) or not splits:
        raise ValueError("report must contain rows and splits")
    split_map: dict[int, set[str]] = {}
    fold_seed: dict[int, int] = {}
    for split in splits:
        fold, seed = split["fold"], split["seed"]
        if not isinstance(fold, int) or fold in split_map:
            raise ValueError("invalid or duplicate fold")
        train = set(split["training_sha256"])
        held = set(split["heldout_sha256"])
        if not train or not held or len(train) != len(split["training_sha256"]) or len(held) != len(split["heldout_sha256"]):
            raise ValueError("missing or duplicate recording hashes in split")
        if not train.isdisjoint(held):
            raise ValueError("train/test WAV leakage in reported split")
        split_map[fold] = held
        fold_seed[fold] = seed
    by_group: dict[tuple[int, float], dict[str, dict]] = {}
    budgets = set()
    for row in rows:
        fold, budget, policy = row["fold"], row["budget_fraction"], row["policy"]
        if fold not in split_map or row["seed"] != fold_seed[fold]:
            raise ValueError("row references unknown fold or incorrect seed")
        if policy not in REQUIRED_POLICIES or not 0 < budget <= 1:
            raise ValueError("unknown policy or budget")
        _finite_score(row["penalized_acoustic_distance"], "penalized_acoustic_distance")
        group = by_group.setdefault((fold, budget), {})
        if policy in group:
            raise ValueError("duplicate fold/budget/policy")
        group[policy] = row
        budgets.add(budget)
    expected = {(fold, budget) for fold in split_map for budget in budgets}
    if set(by_group) != expected:
        raise ValueError("incomplete fold/budget grid")
    for (fold, budget), group in by_group.items():
        if set(group) != set(REQUIRED_POLICIES):
            raise ValueError("incomplete policy comparison")
        sizes = {(r["budget_wav"], r["train_available_wav"],
                  r["heldout_wav_cell_pairs"]) for r in group.values()}
        if len(sizes) != 1:
            raise ValueError("unequal acquisition or evaluation budgets")
        if group[ADAPTIVE]["budget_wav"] > len(
            splits[[s["fold"] for s in splits].index(fold)]["training_sha256"]
        ):
            raise ValueError("budget exceeds available WAV recordings")
    comparisons = []
    for budget in sorted(budgets):
        for baseline in COMPARISONS:
            paired = []
            for fold in sorted(split_map):
                group = by_group[(fold, budget)]
                source = _finite_score(group[baseline]["penalized_acoustic_distance"], "baseline")
                adaptive = _finite_score(group[ADAPTIVE]["penalized_acoustic_distance"], "adaptive")
                delta = adaptive - source
                paired.append({
                    "fold": fold, "seed": fold_seed[fold],
                    "budget_wav": group[ADAPTIVE]["budget_wav"],
                    "baseline_distance": round(source, 5),
                    "adaptive_distance": round(adaptive, 5),
                    "adaptive_minus_baseline": round(delta, 5),
                    "result": "adaptive_better" if delta < -EPSILON
                              else "adaptive_worse" if delta > EPSILON
                              else "tie",
                })
            mean_baseline = sum(x["baseline_distance"] for x in paired) / len(paired)
            mean_adaptive = sum(x["adaptive_distance"] for x in paired) / len(paired)
            comparisons.append({
                "budget_fraction": budget,
                "baseline": baseline,
                "folds": len(paired),
                "adaptive_wins": sum(x["result"] == "adaptive_better" for x in paired),
                "adaptive_losses": sum(x["result"] == "adaptive_worse" for x in paired),
                "ties": sum(x["result"] == "tie" for x in paired),
                "mean_baseline_distance": round(mean_baseline, 6),
                "mean_adaptive_distance": round(mean_adaptive, 6),
                "mean_paired_delta": round(mean_adaptive - mean_baseline, 6),
                "relative_mean_difference_pct": round(
                    100 * (mean_adaptive - mean_baseline) / mean_baseline, 3
                ) if mean_baseline > 0 else None,
                "per_fold": paired,
            })
    overlaps = []
    for left, right in combinations(sorted(split_map), 2):
        a, b = split_map[left], split_map[right]
        shared = len(a & b)
        overlaps.append({
            "fold_a": left, "fold_b": right, "shared_test_wavs": shared,
            "test_jaccard": round(shared / len(a | b), 5),
        })
    holdout_usage = Counter(hash_ for values in split_map.values() for hash_ in values)
    ceiling = all(r["cell_coverage"] == 1.0 for r in rows)
    return {
        "analysis": "recording_budget_paired_fold_audit",
        "source_version": payload["version"],
        "folds": len(split_map),
        "budgets": sorted(budgets),
        "comparisons": comparisons,
        "test_overlap": {
            "pairwise": overlaps,
            "mean_pairwise_jaccard": round(
                sum(x["test_jaccard"] for x in overlaps) / len(overlaps), 5
            ) if overlaps else None,
            "distinct_heldout_wavs_across_folds": len(holdout_usage),
            "wavs_used_as_test_more_than_once": sum(v > 1 for v in holdout_usage.values()),
        },
        "coarse_cell_coverage_saturated": ceiling,
        "interpretation_limits": [
            "Repeated folds share WAV recordings; paired fold counts are descriptive, not independent Bernoulli trials.",
            "The five deterministic splits cannot establish statistical significance or speaker-level generalization.",
            "A positive distance delta means adaptive allocation underperformed its baseline.",
            "All policies see candidate metadata; only selected WAV acoustic features are revealed to the adaptive algorithm.",
            "Results reflect existing fricative acoustic-feature coverage, not actual synthesizer or recording-time efficiency.",
            "No selection policy, normalization, feature set, or source benchmark metric was changed.",
        ],
    }


def audit_budget_json(source: Path, output: Path | None = None) -> dict:
    source = source.expanduser().resolve()
    report = json.loads(source.read_text(encoding="utf-8"))
    result = analyze_budget_report(report)
    if output is not None:
        destination = output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
