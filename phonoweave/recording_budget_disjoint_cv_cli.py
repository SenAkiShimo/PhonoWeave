"""Disjoint-WAV recording budget benchmark."""
from __future__ import annotations
import argparse
from pathlib import Path
from .recording_budget_disjoint_cv import run_voicebank_disjoint_benchmark

def main(argv=None):
    parser=argparse.ArgumentParser(description="Evaluate fixed, random and adaptive acquisition on disjoint WAV folds")
    parser.add_argument("voicebank",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    report=run_voicebank_disjoint_benchmark(args.voicebank,args.output)
    print("Eligible unique WAVs:",report["extraction"]["unique_hashes_with_features"])
    print("Partition:",report["partition"])
    for item in report["aggregate"]:
        print("{} {:.0%}: coverage={} heldout_distance={}".format(
            item["policy"],item["budget_fraction"],item["mean_cell_coverage"],
            item["mean_penalized_acoustic_distance"]))
    for item in report["paired_comparisons"]:
        print("adaptive vs {} {:.0%}: wins={} losses={} ties={} mean_delta={:+.6f}".format(
            item["baseline"],item["budget_fraction"],item["adaptive_wins"],
            item["adaptive_losses"],item["ties"],item["mean_paired_delta"]))
    print("Saved:",args.output)

if __name__=="__main__":
    main()
