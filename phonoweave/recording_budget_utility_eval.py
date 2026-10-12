"""Exploratory retrospective cross-bank comparison of frozen and utility selectors."""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean

from .recording_budget_benchmark import BUDGETS, load_recording_units, metadata, select_sequence
from .recording_budget_acquisition_utility import select_utility_sequence
from .recording_budget_context_control import coverage_first_random_sequence
from .recording_budget_context_fast_eval import PreparedHeldoutDistances
from .recording_budget_cost_audit import eligible_wav_durations
from .recording_budget_cross_voicebank import validate_manifest
from .recording_budget_disjoint_cv import PARTITION_SEED, disjoint_folds

VERSION = "1.0-utility-exploratory"


def evaluate_bank(units, seconds):
    folds = disjoint_folds(units, seed=PARTITION_SEED)
    rows = []
    for fold, test in enumerate(folds):
        held = {u.digest for u in test}
        train = tuple(u for u in units if u.digest not in held)
        lookup = {u.digest:u for u in train}
        meta = metadata(train)
        getter = lambda key: lookup[key].examples
        seeds = PARTITION_SEED + 8101 + fold
        sequences = {
            "utility_acoustic": select_utility_sequence(meta, getter),
            "utility_metadata_only": select_utility_sequence(meta, use_acoustic_feedback=False),
            "legacy_adaptive": select_sequence(meta,getter,policy="adaptive_observed",seed=seeds),
            "fixed_balanced": select_sequence(meta,getter,policy="fixed_balanced",seed=seeds),
            "coverage_random": coverage_first_random_sequence(meta,seed=seeds),
        }
        scorer = PreparedHeldoutDistances(lookup,test)
        time = sum(seconds[k] for k in lookup)
        segment_count = sum(len(u.examples) for u in train)
        scores = {
            policy: scorer.score_duration_budgets(
                sequence, seconds, BUDGETS, time, segment_count
            )
            for policy,sequence in sequences.items()
        }
        rows.append({
            "fold":fold,"train_wavs":len(train),"test_wavs":len(test),
            "budget_results":[{
                "budget_fraction":b,
                "methods":{policy:scores[policy][b] for policy in sequences},
            } for b in BUDGETS]
        })
    summary=[]
    for b in BUDGETS:
        by_fold=[next(row for row in fold["budget_results"]
                      if row["budget_fraction"] == b) for fold in rows]
        distances={
            method:round(mean(row["methods"][method]["penalized_distance"]
                              for row in by_fold),6)
            for method in by_fold[0]["methods"]
        }
        summary.append({
            "budget_fraction":b,
            "mean_distance_by_method":distances,
            "utility_acoustic_minus_metadata":round(
                distances["utility_acoustic"]-distances["utility_metadata_only"],6
            ),
            "utility_acoustic_minus_legacy":round(
                distances["utility_acoustic"]-distances["legacy_adaptive"],6
            ),
            "utility_acoustic_minus_coverage_random":round(
                distances["utility_acoustic"]-distances["coverage_random"],6
            ),
            "acoustic_mean_spent_fraction":round(mean(
                row["methods"]["utility_acoustic"]["spent_fraction"]
                for row in by_fold),6),
            "metadata_mean_spent_fraction":round(mean(
                row["methods"]["utility_metadata_only"]["spent_fraction"]
                for row in by_fold),6),
        })
    return {"version":VERSION,"unique_wavs":len(units),"folds":rows,"summary":summary}


def run_manifest(manifest:Path, output:Path) -> dict:
    data=json.loads(manifest.expanduser().resolve().read_text(encoding="utf-8"))
    banks=validate_manifest(data)
    results=[]
    for bank in banks:
        units,_=load_recording_units(bank["root"])
        durations=eligible_wav_durations(bank["root"],units)
        result=evaluate_bank(units,durations)
        result.update(bank_id=bank["bank_id"],speaker_group_id=bank["speaker_group_id"])
        results.append(result)
    macro=[]
    for b in BUDGETS:
        rows=[next(item for item in bank["summary"] if item["budget_fraction"]==b)
              for bank in results]
        macro.append({
            "budget_fraction":b,
            "speaker_count":len(rows),
            "acoustic_utility_better_than_metadata":sum(
                r["utility_acoustic_minus_metadata"] < -1e-8 for r in rows),
            "acoustic_utility_better_than_legacy":sum(
                r["utility_acoustic_minus_legacy"] < -1e-8 for r in rows),
            "acoustic_utility_better_than_coverage_random":sum(
                r["utility_acoustic_minus_coverage_random"] < -1e-8 for r in rows),
            "mean_acoustic_minus_metadata":round(
                mean(r["utility_acoustic_minus_metadata"] for r in rows),6),
            "mean_acoustic_minus_coverage_random":round(
                mean(r["utility_acoustic_minus_coverage_random"] for r in rows),6),
        })
    report={
        "version":VERSION,
        "experiment":"exploratory_acquisition_utility_comparison",
        "development_bank_status":"previously_inspected_not_independent_validation",
        "scoring":"frozen_oto_fricative_acoustic_distance",
        "cost":"strict_full_WAV_duration_ceiling_no_backfill",
        "results":results,"summary":macro,
    }
    target=output.expanduser().resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report
