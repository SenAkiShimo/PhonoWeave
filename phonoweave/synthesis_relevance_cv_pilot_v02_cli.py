"""Balanced and OTO-duration-corrected CV substitution proxy pilot."""
from __future__ import annotations
import argparse
from pathlib import Path
from .synthesis_relevance_cv_pilot_v02 import run_voicebank

def main(argv=None):
    p=argparse.ArgumentParser(description="Run balanced CV boundary substitution screening (v0.2)")
    p.add_argument("voicebank",type=Path)
    p.add_argument("--base",choices=("sh","s","x"),default="sh")
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args(argv)
    r=run_voicebank(a.voicebank,a.output,base=a.base)
    print("Base:",r["base_unit"],"candidate events:",r["extraction"]["candidate_events"])
    for row in r["context_summary"]:
        print("{}: targets={} mean_cross_minus_same={} cross_worse={}".format(
            row["context"],row["targets"],row["mean_cross_minus_same"],row["cross_worse_targets"]))
    print("Subbank breakdown:")
    for row in r["subbank_summary"]:
        if row["targets"]:
            print("  {} {}: n={} mean={}".format(
                row["subbank"],row["context"],row["targets"],row["mean_cross_minus_same"]))
    print("Exclusions:",r["excluded_targets"])
    print("Saved:",a.output)

if __name__=="__main__":
    main()
