"""Read-only OTO-aligned CV substitution screening."""
from __future__ import annotations
import argparse
from pathlib import Path
from .synthesis_relevance_cv_pilot import run_voicebank

def main(argv=None):
    parser=argparse.ArgumentParser(description="Screen same-context versus cross-context OTO CV boundary penalties")
    parser.add_argument("voicebank",type=Path)
    parser.add_argument("--base",choices=("sh","s","x"),default="sh")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    result=run_voicebank(args.voicebank,args.output,base=args.base)
    print("Base:",result["base_unit"],"eligible OTO events:",result["extraction"]["candidate_events"])
    for row in result["context_summary"]:
        print("{}: targets={} mean_cross_minus_matched={} cross_worse_targets={}".format(
            row["context"],row["targets"],row["mean_cross_minus_matched"],
            row.get("cross_worse_target_count",0)))
    print("Excluded targets:",result["excluded_targets"])
    print("Saved:",args.output)

if __name__=="__main__":
    main()
