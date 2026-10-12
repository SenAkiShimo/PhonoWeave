"""Run read-only OTO and acoustic-separability validity audit."""
from __future__ import annotations
import argparse
from pathlib import Path
from .synthesis_relevance_validity_audit import run_manifest

def main(argv=None):
    p=argparse.ArgumentParser(description="Audit OTO acoustic extraction, grouped label null, and pitch-window sensitivity")
    p.add_argument("manifest",type=Path)
    p.add_argument("--base",choices=("sh","s","x"),default="sh")
    p.add_argument("--permutations",type=int,default=399)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    report=run_manifest(a.manifest,a.output,base=a.base,permutations=a.permutations)
    print("Banks:",report["evaluated_banks"])
    for bank in report["banks"]:
        reasons=bank["extraction_audit"]["extraction_reasons"]
        null=bank["grouped_label_permutation"]
        transfer=bank["leave_one_subbank_out"]
        print(bank["bank_id"],"events",bank["baseline_events"],
              "rejections",{k:v for k,v in reasons.items() if k!="accepted"},
              "BA",bank["baseline_grouped_cv"].get("balanced_accuracy"),
              "null_p",null.get("permutation_p_upper_tail"),
              "null_status",null["status"],
              "subbank_transfer_BA",transfer.get("mean_heldout_subbank_accuracy"))
    print("Saved:",a.output)

if __name__=="__main__":
    main()
