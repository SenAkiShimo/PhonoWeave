from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_noise_gate_audit import audit_noise_gates

def main(argv=None):
    p = argparse.ArgumentParser(description="Audit noisy frication gate failures")
    p.add_argument("session", type=Path)
    r = audit_noise_gates(p.parse_args(argv).session)
    print("Session:", r["session_id"])
    for condition, data in r["summary"].items():
        print(f"{condition}: paired={data['paired_candidates']}/{data['target_occurrences']}, "
              f"manual_paired={data['eligible_paired']}/{data['eligible_manual']}, "
              f"strict_failures={data['strict_gate_failures']}, "
              f"cv_or_sequence_review={data['cv_or_sequence_review_after_core']}, "
              f"single_gate_ablations={data['single_gate_ablation_recovery_diagnostic_only']}, "
              f"reasons={data['prediction_failure_reasons']}")
    print("Failures:", r["failures"])
    print("Saved: analysis/calibration_noise_gate_attribution_v2.3.json")

if __name__ == "__main__":
    main()
