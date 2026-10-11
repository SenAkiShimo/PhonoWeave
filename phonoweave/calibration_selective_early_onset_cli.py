"""CLI for frozen v2.8 selective-onset replay."""
from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_selective_early_onset import evaluate_selective_replay

def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare baseline, always-backtrace and selective onset replay")
    parser.add_argument("session", type=Path)
    args = parser.parse_args(argv)
    report = evaluate_selective_replay(args.session)
    clean = report["clean"]
    if clean:
        print("original: baseline={}ms always={}ms selective={}ms selected={}/{}".format(
            clean["baseline"]["conditional_mae_ms"],
            clean["always_backtrace"]["conditional_mae_ms"],
            clean["selective"]["conditional_mae_ms"],
            clean["selection_count"], clean["total_targets"]))
    for name, row in report["aggregate"].items():
        print("{}: baseline={}ms always={}ms selective={}ms better={}/{} selected={} worsened={}".format(
            name, row["baseline_median_mae_ms"], row["always_backtrace_median_mae_ms"],
            row["selective_median_mae_ms"], row["selective_better_trials_vs_baseline"],
            row["trials"], row["selective_applied_total"], row["selective_worsened_occurrences"]))
    print("Source SHA256:", report["source_sha256"])
    print("Saved: analysis/calibration_selective_early_onset_v2.9.json")

if __name__ == "__main__":
    main()
