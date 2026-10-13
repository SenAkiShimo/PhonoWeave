"""Read-only physical-WAV inventory eligibility audit."""
from __future__ import annotations
import argparse
from pathlib import Path
from .synthesis_relevance_inventory_feasibility import run_manifest

def main(argv=None):
    parser=argparse.ArgumentParser(description="Audit independent-WAV final coverage and strict matching eligibility")
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--base",choices=("sh",),default="sh")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    report=run_manifest(args.manifest,args.output,base=args.base)
    for bank in report["banks"]:
        a=bank["aggregate_counts"]
        c=bank["mutually_exclusive_target_statuses"]
        print(bank["bank_id"],"WAVs",bank["physical_WAVs"],
              "strict_targets",bank["strict_trial_replay"],
              "same_final_missing",c.get("missing_independent_same_final",0),
              "opposite_final_missing",c.get("missing_independent_opposite_final",0),
              "duration_blocked",sum(v for k,v in c.items() if "caliper_failure" in k or k=="arm_duration_imbalance"),
              "additional_final_instances_two_each",a["final_recording_deficit_two_each"])
    print("Saved:",args.output)

if __name__=="__main__":
    main()
