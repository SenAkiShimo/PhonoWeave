from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_paired_noise import evaluate_paired_noise

def main(argv=None):
    p=argparse.ArgumentParser(description="Paired CV onset robustness evaluation")
    p.add_argument("session",type=Path)
    r=evaluate_paired_noise(p.parse_args(argv).session)
    print("Session:",r["session_id"])
    for condition, s in r["summary"].items():
        print(f"{condition}: positive={s['positive_paired']}/{s['positive_total']}, "
              f"gold_matched={s['positive_matched']}/{s['positive_eligible']}, "
              f"MAE={s['positive_mae_ms']}ms; negative_paired={s['negative_paired']}/{s['negative_total']} "
              f"negative_onset_only={s['negative_onset_only']}")
    print("Failures:",r["failures"])
    print("Saved: analysis/calibration_paired_noise_v2.2.json")
if __name__=="__main__":main()
