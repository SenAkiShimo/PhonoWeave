from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_early_frication_multiseed import audit_early_frication_multiseed

def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit shadow early-frication onset timing")
    parser.add_argument("session", type=Path)
    report = audit_early_frication_multiseed(parser.parse_args(argv).session)
    c = report["clean"]
    print(f"clean: baseline_MAE={c['baseline']['conditional_mae_ms']}ms "
          f"shadow_MAE={c['shadow']['conditional_mae_ms']}ms "
          f"matched={c['baseline']['matched']}/{c['baseline']['eligible']}")
    for name, item in report["aggregate"].items():
        print(f"{name}: baseline_MAE={item['baseline_median_mae_ms']}ms "
              f"shadow_MAE={item['shadow_median_mae_ms']}ms "
              f"shadow_better={item['shadow_better_trial_count']}/{item['trials']} "
              f"baseline_gt40={item['baseline_errors_over_40ms_total']} "
              f"shadow_gt40={item['shadow_errors_over_40ms_total']} "
              f"negative_trials={item['negative_trials_with_paired_recording']}/{item['trials']}")
    print("Saved: analysis/calibration_multiseed_early_frication_v2.8.json")

if __name__ == "__main__":
    main()
