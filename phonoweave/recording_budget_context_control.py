"""Exploratory context-coverage-constrained random control for Experiment 1.

A metadata-only selector greedily prioritizes uncovered phonetic-context
cells and otherwise uses the original v0.2 stratified-random priority weights.
It does not observe unacquired WAV spectra, WAV duration, or evaluation audio.

The frozen v0.7 strict consecutive-prefix duration ceiling remains in force.
Scores are decomposed into a missing-cell penalty and within-covered-cell
distance using exactly the original metric and denominator.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from .recording_budget_benchmark import (
    BUDGETS, MISSING_CELL_PENALTY, CandidateMetadata, RecordingUnit,
    acoustic_distance, evaluate_heldout, load_recording_units, metadata,
    select_sequence,
)
from .recording_budget_cost_audit import eligible_wav_durations
from .recording_budget_cross_voicebank import validate_manifest
from .recording_budget_disjoint_cv import PARTITION_SEED, disjoint_folds
from .recording_budget_duration_comparison import (
    _duration_row, duration_limited_prefix,
)
from .recording_budget_random_sensitivity import SEED_STRIDE

VERSION = "0.8-context-coverage-random-control"
REQUIRED_V07 = "0.7-equal-duration-ceiling"
EPS = 1e-8


def coverage_first_random_sequence(
    candidates: tuple[CandidateMetadata, ...], *, seed: int
) -> tuple[str, ...]:
    """Use only candidate OTO metadata, never acoustic vectors or WAV lengths.

    Before all known training cells are represented, consider only candidates
    attaining the largest number of uncovered cells. Ties are sampled with
    the frozen stratified-random priority weights. After full coverage,
    sample all remaining candidates by those same priority weights.
    """
    if len({c.identifier for c in candidates}) != len(candidates):
        raise ValueError("duplicate candidate identifier")
    if any(not c.cells for c in candidates):
        raise ValueError("candidate without phonetic-context metadata")
    rng = np.random.default_rng(seed)
    remaining = {c.identifier: c for c in candidates}
    uncovered = {cell for c in candidates for cell in c.cells}
    counts: Counter[str] = Counter()
    sequence: list[str] = []
    while remaining:
        eligible = sorted(remaining)
        if uncovered:
            gains = {
                key: len(set(remaining[key].cells) & uncovered) for key in eligible
            }
            maximum = max(gains.values())
            if maximum <= 0:
                raise AssertionError("uncovered metadata cannot be resolved")
            eligible = [key for key in eligible if gains[key] == maximum]
        weights = np.asarray([
            max(1.0 / (1.0 + counts[cell]) for cell in remaining[key].cells)
            for key in eligible
        ], dtype=float)
        if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0:
            raise ValueError("invalid stratified random weights")
        key = eligible[int(rng.choice(len(eligible), p=weights / weights.sum()))]
        chosen = remaining.pop(key)
        sequence.append(key)
        counts.update(set(chosen.cells))
        uncovered.difference_update(chosen.cells)
    return tuple(sequence)


def decompose_distance(
    selected: tuple[RecordingUnit, ...],
    heldout: tuple[RecordingUnit, ...],
) -> dict:
    """Partition v0.2's unchanged penalized mean on the same WAV/cell pairs.

    Matched conditional means vary in denominator when classes are absent.
    For fair accounting the *contributions* to the same total denominator
    are reported separately, including zero matched contribution if none.
    """
    pool = defaultdict(list)
    for unit in selected:
        for example in unit.examples:
            pool[example.cell].append(example)
    totals = []
    conditional = []
    missing = 0
    for unit in heldout:
        per_cell = defaultdict(list)
        for e in unit.examples:
            per_cell[e.cell].append(e)
        for cell, examples in per_cell.items():
            if cell not in pool:
                missing += 1
                totals.append(MISSING_CELL_PENALTY)
            else:
                distance = float(np.mean([
                    min(acoustic_distance(e, c) for c in pool[cell])
                    for e in examples
                ]))
                conditional.append(distance)
                totals.append(distance)
    if not totals:
        raise ValueError("no held-out WAV/cell observations")
    divisor = len(totals)
    within = sum(conditional) / divisor
    penalty = missing * MISSING_CELL_PENALTY / divisor
    score = evaluate_heldout(selected, heldout)
    if abs((within + penalty) - score["penalized_acoustic_distance"]) > 0.000011:
        raise AssertionError("metric decomposition differs from frozen v0.2")
    return {
        "heldout_wav_cell_pairs": divisor,
        "matched_wav_cell_pairs": len(conditional),
        "missing_wav_cell_pairs": missing,
        "missing_penalty_contribution": round(penalty, 6),
        "matched_distance_contribution": round(within, 6),
        "matched_only_distance": score["matched_only_acoustic_distance"],
        "penalized_distance": score["penalized_acoustic_distance"],
        "full_context_coverage": score["cell_coverage"] == 1.0,
        "cell_coverage": score["cell_coverage"],
    }


def _evaluated_sequence(
    sequence: tuple[str, ...],
    train_map: dict[str, RecordingUnit],
    duration_seconds: dict[str, float],
    held: tuple[RecordingUnit, ...],
    limit_seconds: float,
    available_seconds: float,
    available_segments: int,
) -> dict:
    row = _duration_row(
        sequence, train_map, duration_seconds, held,
        limit_seconds, available_seconds, available_segments,
    )
    chosen, _, _ = duration_limited_prefix(sequence, duration_seconds, limit_seconds)
    parts = decompose_distance(tuple(train_map[key] for key in chosen), held)
    if parts["penalized_distance"] != row["heldout_penalized_distance"]:
        raise AssertionError("frozen held-out distance changed")
    return {
        "acquired_wavs": row["acquired_wavs"],
        "acquired_eligible_segments": row["acquired_eligible_segments"],
        "spent_fraction": row["spent_fraction"],
        "unspent_seconds": row["budget_unspent_seconds"],
        **parts,
    }


def _validate_original(
    bank_reference: dict, units: tuple[RecordingUnit, ...],
    folds: tuple, duration: dict[str, float],
) -> None:
    if bank_reference.get("status") != "evaluated":
        raise ValueError("v0.7 reference bank not evaluated")
    if bank_reference.get("version") != REQUIRED_V07:
        raise ValueError("v0.7 reference version mismatch")
    if bank_reference["unique_eligible_wavs"] != len(units):
        raise ValueError("eligible WAV counts differ from v0.7")
    if bank_reference["eligible_oto_acoustic_segments"] != sum(
        len(u.examples) for u in units
    ):
        raise ValueError("OTO segment counts differ from v0.7")
    if not math.isclose(
        bank_reference["total_full_wav_seconds"], sum(duration.values()),
        rel_tol=0, abs_tol=0.000002,
    ):
        raise ValueError("eligible WAV durations differ from v0.7")
    if len(bank_reference["fold_reports"]) != len(folds):
        raise ValueError("v0.7 fold count mismatch")


def evaluate_context_control(
    units: tuple[RecordingUnit, ...],
    wav_seconds: dict[str, float],
    bank_reference: dict,
    *,
    repeats: int = 100,
) -> dict:
    if type(repeats) is not int or repeats < 2:
        raise ValueError("random repeat count must be >= 2")
    if set(wav_seconds) != {u.digest for u in units}:
        raise ValueError("eligible WAV duration index mismatch")
    groups = disjoint_folds(units, seed=PARTITION_SEED)
    _validate_original(bank_reference, units, groups, wav_seconds)
    result_folds = []
    for fold_index, held in enumerate(groups):
        ref_fold = bank_reference["fold_reports"][fold_index]
        test_hashes = {u.digest for u in held}
        lookup = {u.digest: u for u in units if u.digest not in test_hashes}
        available_seconds = sum(wav_seconds[hash_] for hash_ in lookup)
        available_segments = sum(len(u.examples) for u in lookup.values())
        if (
            ref_fold["fold"] != fold_index
            or ref_fold["train_unique_wavs"] != len(lookup)
            or ref_fold["test_unique_wavs"] != len(held)
            or ref_fold["training_eligible_segments"] != available_segments
            or not math.isclose(ref_fold["training_total_wav_seconds"], available_seconds,
                                rel_tol=0, abs_tol=0.000002)
        ):
            raise ValueError("v0.7 train/test partition or cost mismatch")
        candidate_metadata = metadata(tuple(lookup.values()))
        first_seed = PARTITION_SEED + 8101 + fold_index
        original = {
            policy: select_sequence(
                candidate_metadata,
                lambda digest: lookup[digest].examples,
                policy=policy, seed=first_seed,
            )
            for policy in ("fixed_balanced", "adaptive_observed", "stratified_random")
        }
        random_original = select_sequence(
            candidate_metadata,
            lambda digest: lookup[digest].examples,
            policy="stratified_random", seed=first_seed,
        )
        if original["stratified_random"] != random_original:
            raise AssertionError("v0.7 first random draw replay failed")
        trials = [
            coverage_first_random_sequence(
                candidate_metadata, seed=first_seed + i * SEED_STRIDE
            )
            for i in range(repeats)
        ]
        fold_budgets = []
        for fraction in BUDGETS:
            ref_budget = next(
                (row for row in ref_fold["budgets"] if row["budget_fraction"] == fraction),
                None,
            )
            if ref_budget is None:
                raise ValueError("missing v0.7 budget")
            ceiling = available_seconds * fraction
            if not math.isclose(ref_budget["budget_ceiling_seconds"], ceiling,
                                rel_tol=0, abs_tol=0.000002):
                raise ValueError("duration ceiling differs from v0.7")
            anchors = {
                name: _evaluated_sequence(
                    sequence, lookup, wav_seconds, held,
                    ceiling, available_seconds, available_segments,
                )
                for name, sequence in original.items()
            }
            for policy, anchor in anchors.items():
                old = ref_budget["policies"][policy]
                for key, expected in (
                    ("acquired_wavs", old["acquired_wavs"]),
                    ("acquired_eligible_segments", old["acquired_eligible_segments"]),
                    ("penalized_distance", old["heldout_penalized_distance"]),
                    ("cell_coverage", old["heldout_cell_coverage"]),
                    ("spent_fraction", old["spent_fraction"]),
                ):
                    if not math.isclose(anchor[key], expected, rel_tol=0, abs_tol=0.000002):
                        raise ValueError(f"v0.7 replay mismatch: {policy}/{key}")
            if ref_budget["random_repetitions"][0]["heldout_penalized_distance"] != (
                anchors["stratified_random"]["penalized_distance"]
            ):
                raise ValueError("v0.7 first random baseline is inconsistent")
            control_rows = [
                _evaluated_sequence(
                    order, lookup, wav_seconds, held, ceiling,
                    available_seconds, available_segments,
                )
                for order in trials
            ]
            fold_budgets.append({
                "budget_fraction": fraction,
                "duration_ceiling_seconds": round(ceiling, 6),
                "reference_policies": anchors,
                "coverage_first_random": control_rows,
            })
        result_folds.append({
            "fold": fold_index,
            "heldout_wavs": len(held),
            "training_wavs": len(lookup),
            "eligible_training_cells": sorted({
                cell for unit in lookup.values() for cell in unit.cells
            }),
            "budgets": fold_budgets,
        })

    summary = []
    for fraction in BUDGETS:
        items = [next(row for row in fold["budgets"]
                      if row["budget_fraction"] == fraction) for fold in result_folds]
        adapt = [row["reference_policies"]["adaptive_observed"] for row in items]
        fixed = [row["reference_policies"]["fixed_balanced"] for row in items]
        raw_random = [row["reference_policies"]["stratified_random"] for row in items]
        ctrl_by_repeat = [
            [row["coverage_first_random"][idx] for row in items]
            for idx in range(repeats)
        ]
        control_macro = np.asarray([
            mean(x["penalized_distance"] for x in trial)
            for trial in ctrl_by_repeat
        ], dtype=float)
        adaptive_macro = float(mean(x["penalized_distance"] for x in adapt))
        full_adaptive = all(x["full_context_coverage"] for x in adapt)
        fully_matched = [
            idx for idx, trial in enumerate(ctrl_by_repeat)
            if all(x["full_context_coverage"] for x in trial)
        ]
        matched_macro = [
            mean(x["matched_only_distance"] for x in ctrl_by_repeat[idx])
            for idx in fully_matched
        ]
        if full_adaptive and matched_macro:
            full_adaptive_matched = mean(x["matched_only_distance"] for x in adapt)
            full_control_matched = mean(matched_macro)
        else:
            full_adaptive_matched = full_control_matched = None
        better = int(np.sum(control_macro < adaptive_macro - EPS))
        ties = int(np.sum(np.abs(control_macro - adaptive_macro) <= EPS))
        summary.append({
            "budget_fraction": fraction,
            "adaptive_distance": round(adaptive_macro, 6),
            "fixed_distance": round(mean(x["penalized_distance"] for x in fixed), 6),
            "old_random_seed0_distance": round(
                mean(x["penalized_distance"] for x in raw_random), 6
            ),
            "context_random_mean_distance": round(float(np.mean(control_macro)), 6),
            "context_random_p05_distance": round(float(np.percentile(control_macro, 5)), 6),
            "context_random_p95_distance": round(float(np.percentile(control_macro, 95)), 6),
            "adaptive_minus_context_random_mean": round(
                adaptive_macro - float(np.mean(control_macro)), 6
            ),
            "context_random_draws_better_than_adaptive": better,
            "context_random_draws_equal_to_adaptive": ties,
            "context_random_draws_worse_than_adaptive": repeats - better - ties,
            "context_random_full_coverage_fold_trials": sum(
                r["full_context_coverage"] for trial in ctrl_by_repeat for r in trial
            ),
            "context_random_total_fold_trials": len(items) * repeats,
            "fully_covered_repeats_all_folds": len(fully_matched),
            "adaptive_fully_covered_all_folds": full_adaptive,
            "adaptive_mean_missing_penalty_component": round(
                mean(x["missing_penalty_contribution"] for x in adapt), 6
            ),
            "context_random_mean_missing_penalty_component": round(
                mean(x["missing_penalty_contribution"]
                     for trial in ctrl_by_repeat for x in trial), 6
            ),
            "adaptive_mean_spent_fraction": round(
                mean(x["spent_fraction"] for x in adapt), 6
            ),
            "context_random_mean_spent_fraction": round(
                mean(x["spent_fraction"] for trial in ctrl_by_repeat for x in trial), 6
            ),
            "adaptive_mean_acquired_segments": round(
                mean(x["acquired_eligible_segments"] for x in adapt), 3
            ),
            "context_random_mean_acquired_segments": round(
                mean(x["acquired_eligible_segments"] for trial in ctrl_by_repeat for x in trial), 3
            ),
            "fully_covered_matched_only_adaptive_mean": (
                round(full_adaptive_matched, 6)
                if full_adaptive_matched is not None else None
            ),
            "fully_covered_matched_only_context_random_mean": (
                round(full_control_matched, 6)
                if full_control_matched is not None else None
            ),
            "random_macro_distances": [
                round(float(score), 6) for score in control_macro
            ],
        })
    return {
        "experiment": "context_coverage_constrained_random_control",
        "version": VERSION,
        "source_protocol": REQUIRED_V07,
        "unique_eligible_wavs": len(units),
        "random_repeats_per_fold": repeats,
        "folds": result_folds,
        "summary": summary,
        "limitations": [
            "Context-first control uses phonetic metadata only, never unacquired waveform or duration.",
            "Metadata coverage is prioritized in the acquisition sequence, not guaranteed under short time caps.",
            "All algorithms stop at first non-fitting whole WAV; actual spent seconds may differ.",
            "Penalty and matched contributions use the same held-out WAV/cell denominator.",
            "Matched-only means are conditional and incomparable if context coverage differs.",
            "Full-coverage-only comparisons select random draws after observing coverage outcomes.",
            "Reported repeated draws represent acquisition randomness, not cross-singer confidence intervals.",
            "The original 2.0 missing-cell penalty and acoustic features are unchanged.",
            "Study is exploratory after inspection of the v0.7 results.",
            "OTO fricative acoustic coverage is not a singing-synthesis-quality score.",
        ],
    }


def evaluate_manifest(
    manifest_path: Path, reference_path: Path, output_path: Path,
    *, repeats: int = 100,
) -> dict:
    manifest = json.loads(manifest_path.expanduser().resolve().read_text("utf-8"))
    original = json.loads(reference_path.expanduser().resolve().read_text("utf-8"))
    banks = validate_manifest(manifest)
    if original.get("version") != REQUIRED_V07:
        raise ValueError("reference must be v0.7 duration experiment")
    reference_banks = {
        row["bank_id"]: row for row in original.get("results", [])
    }
    if set(reference_banks) != {row["bank_id"] for row in banks}:
        raise ValueError("reference banks do not match manifest")
    results = []
    for bank in banks:
        root = bank["root"]
        units, _ = load_recording_units(root)
        duration = eligible_wav_durations(root, units)
        row = evaluate_context_control(
            units, duration, reference_banks[bank["bank_id"]], repeats=repeats
        )
        row["bank_id"] = bank["bank_id"]
        row["speaker_group_id"] = bank["speaker_group_id"]
        results.append(row)
    groups = []
    for fraction in BUDGETS:
        items = [
            next(x for x in bank["summary"] if x["budget_fraction"] == fraction)
            for bank in results
        ]
        groups.append({
            "budget_fraction": fraction,
            "speaker_groups": len(items),
            "adaptive_better_than_control_mean": sum(
                x["adaptive_minus_context_random_mean"] < -EPS for x in items
            ),
            "adaptive_worse_than_control_mean": sum(
                x["adaptive_minus_context_random_mean"] > EPS for x in items
            ),
            "adaptive_tied_to_control_mean": sum(
                abs(x["adaptive_minus_context_random_mean"]) <= EPS for x in items
            ),
            "equal_speaker_mean_adaptive_minus_control": round(
                mean(x["adaptive_minus_context_random_mean"] for x in items), 6
            ),
            "mean_control_full_coverage_trial_fraction": round(
                mean(x["context_random_full_coverage_fold_trials"] /
                     x["context_random_total_fold_trials"] for x in items), 6
            ),
            "mean_adaptive_missing_penalty_component": round(
                mean(x["adaptive_mean_missing_penalty_component"] for x in items), 6
            ),
            "mean_control_missing_penalty_component": round(
                mean(x["context_random_mean_missing_penalty_component"] for x in items), 6
            ),
        })
    report = {
        "experiment": "cross_voicebank_context_coverage_random_control",
        "version": VERSION,
        "source_protocol": REQUIRED_V07,
        "requested_banks": len(banks),
        "evaluated_banks": len(results),
        "random_repeats_per_fold": repeats,
        "original_selection_rules_modified": False,
        "cross_bank_summary": groups,
        "results": results,
        "local_source_paths_in_output": False,
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
