"""Evaluate authentic against within-cell-permuted acquired feedback."""
from __future__ import annotations
import argparse
from pathlib import Path
from .recording_budget_feedback_permutation import run_manifest

def main(argv=None):
    parser=argparse.ArgumentParser(description="Sham acoustic feedback control on frozen eight-bank duration experiment")
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=100)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    report=run_manifest(args.manifest,args.reference,args.output,repeats=args.repeats)
    print("Evaluated speaker groups:",report["speaker_groups"])
    for row in report["equal_speaker_summary"]:
        print("{:.0%}: authentic_better={}/{} authentic_worse={} ties={} mean_delta={:+.6f}".format(
            row["budget_fraction"],row["real_better_than_permuted_mean"],
            row["evaluated_speakers"],row["real_worse_than_permuted_mean"],
            row["ties"],row["equal_speaker_mean_true_minus_permuted"]))
    for bank in report["banks"]:
        print(bank["bank_id"]," ".join(
            "{:.0%}:{:+.5f},sham_wins={}/{}".format(
                r["budget_fraction"],r["true_minus_permuted_mean"],
                r["permuted_draws_better_than_true"],bank["permutations_per_fold"])
            for r in bank["summary"]))
    print("Saved:",args.output)

if __name__=="__main__":
    main()
