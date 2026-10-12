"""Run read-only objective substitution-cost analysis."""
from __future__ import annotations
import argparse
from pathlib import Path
from .synthesis_relevance_unit_merging_cost import run_manifest

def main(argv=None):
    p=argparse.ArgumentParser(description="Objective same-context vs cross-context unit substitution cost")
    p.add_argument("manifest",type=Path)
    p.add_argument("--base",choices=("sh","s","x"),default="sh")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    report=run_manifest(a.manifest,a.output,base=a.base)
    for b in report["banks"]:
        r=b["metrics"]["delta_reconstruction"]
        s=b["metrics"]["delta_seam"]
        print(b["bank_id"],"eligible_targets",b["eligible_targets"],
              "target_WAVs",b["eligible_target_wavs"],
              "reconstruction_delta",r.get("mean_cross_minus_same"),
              "reconstruction_p",r.get("two_sided_WAV_sign_flip_p"),
              "seam_delta",s.get("mean_cross_minus_same"),
              "seam_p",s.get("two_sided_WAV_sign_flip_p"))
    print("Saved:",a.output)

if __name__=="__main__":
    main()
