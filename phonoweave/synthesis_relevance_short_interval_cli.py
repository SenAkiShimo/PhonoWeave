"""Run objective short-OTO-interval evaluation without playing or writing audio."""
from __future__ import annotations
import argparse
from pathlib import Path
from .synthesis_relevance_short_interval import run_manifest

def main(argv=None):
    p=argparse.ArgumentParser(description="Short OTO-interval acoustic separability sensitivity")
    p.add_argument("manifest",type=Path)
    p.add_argument("--base",choices=("sh","s","x"),default="sh")
    p.add_argument("--permutations",type=int,default=199)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    report=run_manifest(a.manifest,a.output,base=a.base,permutations=a.permutations)
    print("Banks:",report["number_of_banks"])
    for b in report["banks"]:
        old=b["protocols"]["legacy"]
        new=b["protocols"]["short_interior"]
        matched=b["matched_legacy_vs_short"]
        print(
            b["bank_id"],
            "legacy_events",old["events"],"short_events",new["events"],
            "short_BA",new["grouped_cv"].get("balanced_accuracy"),
            "short_null_p",b["new_short_label_null"].get("permutation_p_upper_tail"),
            "matched_WAV_contexts",matched["WAV_context_cells"],
            "matched_legacy_BA",matched["legacy_grouped_cv"].get("balanced_accuracy"),
            "matched_short_BA",matched["short_grouped_cv"].get("balanced_accuracy"),
        )
    print("Saved:",a.output)

if __name__=="__main__":
    main()
