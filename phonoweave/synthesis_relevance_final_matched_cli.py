"""Evaluate prespecified same-final versus paired-rounded-final donor matching."""
from __future__ import annotations
import argparse
from pathlib import Path
from .synthesis_relevance_final_matched import run_manifest

def main(argv=None):
    p=argparse.ArgumentParser(description="Strict same-final donor vs paired-rounded-final counterfactual")
    p.add_argument("manifest",type=Path)
    p.add_argument("--base",choices=("sh",),default="sh")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    report=run_manifest(a.manifest,a.output,base=a.base)
    for b in report["banks"]:
        x=b["strict_overall"]["reconstruction"]
        diag=b["same_WAV_diagnostic"]
        print(b["bank_id"],"strict_targets",b["strict_target_events"],
              "strict_WAVs",b["strict_independent_target_WAVs"],
              "strict_delta",x.get("mean_cross_minus_same"),
              "strict_p",x.get("two_sided_WAV_sign_flip_p"),
              "same_WAV_diagnostic_pairs",diag["paired_units"])
    print("Saved:",a.output)

if __name__=="__main__":
    main()
