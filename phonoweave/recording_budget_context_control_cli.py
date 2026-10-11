"""Coverage-first random baseline versus frozen v0.7 recording policies."""
from __future__ import annotations
import argparse
from pathlib import Path
from .recording_budget_context_control import evaluate_manifest

def main(argv=None):
    parser=argparse.ArgumentParser(description="Compare acoustic-adaptive selection with coverage-constrained metadata random baseline")
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=100)
    args=parser.parse_args(argv)
    report=evaluate_manifest(args.manifest,args.reference,args.output,repeats=args.repeats)
    print("Evaluated:",report["evaluated_banks"])
    for row in report["cross_bank_summary"]:
        print("{:.0%}: adaptive_better={}/{} mean_delta={:+.6f} control_full_coverage={:.1%} adaptive_penalty={:.5f} control_penalty={:.5f}".format(
            row["budget_fraction"],row["adaptive_better_than_control_mean"],
            row["speaker_groups"],row["equal_speaker_mean_adaptive_minus_control"],
            row["mean_control_full_coverage_trial_fraction"],
            row["mean_adaptive_missing_penalty_component"],
            row["mean_control_missing_penalty_component"]))
    for bank in report["results"]:
        print(bank["bank_id"]," ".join(
            "{:.0%}:{:+.5f}".format(s["budget_fraction"],s["adaptive_minus_context_random_mean"])
            for s in bank["summary"]))
    print("Saved:",args.output)

if __name__=="__main__":
    main()
