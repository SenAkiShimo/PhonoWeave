"""Cross-voicebank retrospective equal-duration recording-budget evaluation."""
from __future__ import annotations
import argparse
from pathlib import Path
from .recording_budget_duration_comparison import run_duration_manifest

def main(argv=None):
    parser=argparse.ArgumentParser(description="Compare acquisition policies at equal full-WAV duration ceilings")
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=100)
    args=parser.parse_args(argv)
    report=run_duration_manifest(args.manifest,args.output,repeats=args.repeats)
    print("Evaluated:",report["evaluated_banks"],"Excluded:",report["excluded_banks"])
    for bank in report["results"]:
        if bank["status"] != "evaluated":
            print(bank["bank_id"],"EXCLUDED",bank["exclusion_reason"])
            continue
        print(bank["bank_id"],"WAVs",bank["unique_eligible_wavs"],
              "seconds",bank["total_full_wav_seconds"])
        for row in bank["summary"]:
            print("  {:.0%}: adaptive={:.5f} fixed={:.5f} random_mean={:.5f} "
                  "random_better={}/{} spent_adaptive={:.1%} spent_fixed={:.1%}".format(
                  row["budget_fraction"],row["adaptive_distance"],
                  row["fixed_distance"],row["random_mean_distance"],
                  row["random_draws_better_than_adaptive"],
                  report["results"][0]["random_repeats_per_fold"],
                  row["adaptive_mean_spent_fraction"],
                  row["fixed_mean_spent_fraction"]))
    for row in report["equal_speaker_summary_by_budget"]:
        print("Cross-bank {:.0%}: adaptive_better_random={}/{} "
              "mean_delta={:+.6f}, spent_adaptive={:.1%}, spent_fixed={:.1%}".format(
              row["budget_fraction"],row["adaptive_better_than_random_mean"],
              row["speaker_groups"],row["mean_equal_speaker_adaptive_minus_random"] or 0,
              row["mean_adaptive_spent_fraction"] or 0,
              row["mean_fixed_spent_fraction"] or 0))
    print("Saved:",args.output)

if __name__=="__main__":
    main()
