"""Run frozen recording-budget comparisons on independent voicebank groups."""
from __future__ import annotations
import argparse
from pathlib import Path
from .recording_budget_cross_voicebank import run_replication_manifest

def main(argv=None):
    p=argparse.ArgumentParser(description="Cross-voicebank replication of frozen recording-budget selection")
    p.add_argument("manifest",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--repeats",type=int,default=100)
    args=p.parse_args(argv)
    result=run_replication_manifest(args.manifest,args.output,repeats=args.repeats)
    print("Evaluated:",result["evaluated_banks"],"Excluded:",result["excluded_banks"])
    for bank in result["banks"]:
        print(bank["bank_id"],bank["status"],bank.get("eligible_unique_wavs"),
              bank.get("exclusion_reason",""))
    for row in result["summary_by_budget"]:
        print("{:.0%}: adaptive better than random mean {}/{}; average delta {}".format(
            row["budget_fraction"],row["banks_adaptive_better_than_random_mean"],
            row["n_independent_speaker_groups"],
            row["unweighted_mean_adaptive_minus_random_mean"]))
    print("Saved:",args.output)

if __name__=="__main__":
    main()
