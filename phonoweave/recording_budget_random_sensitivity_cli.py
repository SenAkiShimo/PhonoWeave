"""Exploratory random-baseline sensitivity on the frozen v0.2 folds."""
from __future__ import annotations
import argparse
from pathlib import Path
from .recording_budget_random_sensitivity import run_voicebank_random_sensitivity

def main(argv=None):
    parser=argparse.ArgumentParser(description="Evaluate adaptive allocation against repeated stratified-random draws")
    parser.add_argument("voicebank",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=100)
    args=parser.parse_args(argv)
    report=run_voicebank_random_sensitivity(args.voicebank,args.output,n_random=args.repeats)
    print("Eligible WAVs:",report["extraction"]["unique_hashes_with_features"])
    print("Held-out WAVs tested once:",report["unique_heldout_wavs"])
    print("Random repetitions per fold:",report["random_repetitions_per_fold"])
    for row in report["summary"]:
        print("{:.0%}: fixed={:.5f} adaptive={:.5f} random_mean={:.5f} random_p05={:.5f} random_p95={:.5f} random_better={} random_equal={} random_worse={}".format(
            row["budget_fraction"],row["fixed_macro_mean"],row["adaptive_macro_mean"],
            row["random_macro_mean"],row["random_macro_p05"],row["random_macro_p95"],
            row["random_repeats_better_than_adaptive"],
            row["random_repeats_equal_to_adaptive"],
            row["random_repeats_worse_than_adaptive"]))
    print("Saved:",args.output)

if __name__=="__main__":
    main()
