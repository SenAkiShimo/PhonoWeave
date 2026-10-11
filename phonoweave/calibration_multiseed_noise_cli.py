from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_multiseed_noise import evaluate_multiseed

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate white/pink multi-seed onset stability")
    parser.add_argument("session", type=Path)
    result = evaluate_multiseed(parser.parse_args(argv).session)
    print("Session:", result["session_id"])
    for condition, data in result["aggregate"].items():
        strict, fallback = data["strict"], data["fallback"]
        print(
            f"{condition}: strict_paired={strict['paired_min']}-{strict['paired_max']}/8 "
            f"fallback_paired={fallback['paired_min']}-{fallback['paired_max']}/8; "
            f"annotated_strict={strict['annotated_matched_min']}-{strict['annotated_matched_max']}/4 "
            f"annotated_fallback={fallback['annotated_matched_min']}-{fallback['annotated_matched_max']}/4; "
            f"negative_trials_strict={strict['trials_with_any_negative_trigger']}/{data['replicates']} "
            f"negative_trials_fallback={fallback['trials_with_any_negative_trigger']}/{data['replicates']}"
        )
    print("Failures:", result["failures"])
    print("Saved: analysis/calibration_multiseed_noise_stability_v2.6.json")

if __name__ == "__main__":
    main()
