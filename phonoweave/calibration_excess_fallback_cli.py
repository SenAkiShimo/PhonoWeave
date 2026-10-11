from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_excess_fallback_eval import evaluate_excess_fallback

def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare strict and shadow noise-excess frication candidates")
    parser.add_argument("session", type=Path)
    result = evaluate_excess_fallback(parser.parse_args(argv).session)
    print("Session:", result["session_id"])
    for scenario, row in result["summary"].items():
        a, b = row["strict"], row["fallback"]
        f, w = row["fixed_1250ms"], row["overlapping_200ms"]
        print(f"{scenario}: strict={a['positive_paired']}/{row['positive_occurrences']} "
              f"gold={a['gold_matched']}/{row['eligible_gold']} MAE={a['conditional_mae_ms']}ms; "
              f"fallback={b['positive_paired']}/{row['positive_occurrences']} "
              f"gold={b['gold_matched']}/{row['eligible_gold']} MAE={b['conditional_mae_ms']}ms "
              f"new_positives={row['new_positive_paired']}; "
              f"negative_fixed={f['strict_recording_triggers']}->{f['fallback_recording_triggers']}/{f['recordings']}, "
              f"negative_overlapping={w['strict_recording_triggers']}->{w['fallback_recording_triggers']}/{w['recordings']}")
    print("Failures:", result["failures"])
    print("Saved: analysis/calibration_excess_fallback_comparison_v2.5.json")

if __name__ == "__main__":
    main()
