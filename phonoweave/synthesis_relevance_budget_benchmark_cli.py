"""Run read-only fixed-versus-adaptive, budget-matched benchmark."""
from __future__ import annotations
import argparse
from pathlib import Path
from .synthesis_relevance_budget_benchmark import run_manifest

def main(argv=None):
    p=argparse.ArgumentParser(description="Evaluate training-only split policies on held-out WAVs")
    p.add_argument("manifest",type=Path)
    p.add_argument("--base",choices=("sh",),default="sh")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    report=run_manifest(a.manifest,a.output,base=a.base)
    for bank in report["banks"]:
        print(bank["bank_id"],"status",bank["status"],
              "complete_subbanks",bank["complete_four_pair_subbanks"],
              "eligible_pair_cells",bank["evaluable_pair_cells"],
              "blocked_cells",bank["unevaluable_pair_cells"])
        for point in bank["budget_curves"]:
            if point["budget"]==2:
                print("  ",point["subbank"],point["method"],
                      point["mean_heldout_log_spectral_distance"])
    print("Saved:",a.output)
if __name__=="__main__":
    main()
