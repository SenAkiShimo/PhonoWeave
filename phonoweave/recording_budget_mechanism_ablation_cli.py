"""CLI for frozen recording-budget mechanism ablations."""
from __future__ import annotations
import argparse
from pathlib import Path
from .recording_budget_mechanism_ablation import run_voicebank_mechanism_ablation

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate frozen adaptive allocation mechanism ablations")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_voicebank_mechanism_ablation(args.voicebank, args.output)
    print("Eligible WAVs:", report["extraction"]["unique_hashes_with_features"])
    for row in report["summary"]:
        print("{} {:.0%}: distance={:.5f} minus_full={:+.6f} better={} worse={} ties={}".format(
            row["variant"],row["budget_fraction"],row["mean_distance"],
            row["minus_full_adaptive"],row["ablation_better_folds"],
            row["ablation_worse_folds"],row["ties"]))
    print("Saved:",args.output)

if __name__ == "__main__":
    main()
