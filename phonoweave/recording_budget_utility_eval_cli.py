"""Run exploratory acquisition-utility comparison on declared voicebanks."""
from __future__ import annotations
import argparse
from pathlib import Path
from .recording_budget_utility_eval import run_manifest

def main(argv=None):
    p=argparse.ArgumentParser(description="Evaluate acquired-evidence utility against frozen recording-budget controls")
    p.add_argument("manifest",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    result=run_manifest(a.manifest,a.output)
    print("Voicebanks evaluated:",len(result["results"]))
    for row in result["summary"]:
        print("{:.0%}: acoustic_better_than_metadata={}/{} acoustic_better_than_legacy={} acoustic_better_than_coverage_random={} mean_delta_metadata={:+.6f} mean_delta_random={:+.6f}".format(
            row["budget_fraction"],row["acoustic_utility_better_than_metadata"],
            row["speaker_count"],row["acoustic_utility_better_than_legacy"],
            row["acoustic_utility_better_than_coverage_random"],
            row["mean_acoustic_minus_metadata"],
            row["mean_acoustic_minus_coverage_random"]))
    print("Saved:",a.output)

if __name__=="__main__":
    main()
