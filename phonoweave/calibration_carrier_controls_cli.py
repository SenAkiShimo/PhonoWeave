from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_carrier_controls import evaluate_carriers

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("session",type=Path)
    args=parser.parse_args(argv)
    result=evaluate_carriers(args.session)
    print("Session:",result["session_id"])
    for name,s in result["summary"].items():
        print(f"{name}: carrier_windows={s['carrier_tokens']}, "
              f"core_triggers={s['core_candidate_count']} ({s['core_candidate_fraction']}), "
              f"shoulder_triggers={s['shoulder_candidate_count']} ({s['shoulder_candidate_fraction']})")
    print("CAUTION: carrier windows can contain the preceding consonant; NOT clean vowel-only false-positive rates.")
    print("Saved: analysis/calibration_carrier_negative_controls_v1.7.json")
    return 0
if __name__=="__main__":
    raise SystemExit(main())
