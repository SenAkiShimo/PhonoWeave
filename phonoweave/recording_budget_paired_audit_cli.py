"""CLI for paired-fold and holdout-overlap diagnostics."""
from __future__ import annotations
import argparse
from pathlib import Path
from .recording_budget_paired_audit import audit_budget_json

def main(argv=None):
    parser=argparse.ArgumentParser(description="Analyze frozen recording-budget paired fold results")
    parser.add_argument("report",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    result=audit_budget_json(args.report,args.output)
    for r in result["comparisons"]:
        print("{} {:.0%}: wins={} losses={} ties={} mean_delta={:+.6f} relative={:+.3f}%".format(
            r["baseline"],r["budget_fraction"],r["adaptive_wins"],
            r["adaptive_losses"],r["ties"],r["mean_paired_delta"],
            r["relative_mean_difference_pct"] or 0.0))
    print("Held-out overlap:",result["test_overlap"])
    print("Coarse coverage saturated:",result["coarse_cell_coverage_saturated"])
    if args.output:
        print("Saved:",args.output)

if __name__=="__main__":
    main()
