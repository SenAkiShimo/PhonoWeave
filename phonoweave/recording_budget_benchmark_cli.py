"""Retrospective WAV-budget benchmark CLI."""
from __future__ import annotations
import argparse
from pathlib import Path
from .recording_budget_benchmark import run_voicebank_benchmark

def main(argv=None):
    p=argparse.ArgumentParser(description="Compare fixed, random and observed-feature adaptive recording selection")
    p.add_argument("voicebank",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    report=run_voicebank_benchmark(args.voicebank,args.output)
    print("Unique eligible WAVs:",report["extraction"]["unique_hashes_with_features"])
    print("Splits:",len(report["splits"]))
    for row in report["aggregate"]:
        print("{} {:.0%}: coverage={} heldout_distance={}".format(
            row["policy"],row["budget_fraction"],row["mean_cell_coverage"],
            row["mean_penalized_acoustic_distance"]))
    print("Saved:",args.output)

if __name__=="__main__":
    main()
