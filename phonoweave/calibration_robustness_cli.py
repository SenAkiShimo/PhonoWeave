from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_robustness import robustness_session

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("session",type=Path)
    args=parser.parse_args(argv)
    result=robustness_session(args.session)
    print("Session:",result["session_id"])
    for scenario,v in result["summary"].items():
        print(f"{scenario}: eligible={v['eligible']}, "
              f"core {v['core']['matched']}/{v['eligible']} MAE={v['core']['mae_ms']}ms "
              f"median_shift={v['core']['median_abs_shift_from_original_ms']}ms; "
              f"shoulder {v['shoulder']['matched']}/{v['eligible']} "
              f"MAE={v['shoulder']['mae_ms']}ms "
              f"median_shift={v['shoulder']['median_abs_shift_from_original_ms']}ms")
    print("Saved: analysis/calibration_robustness_v1.6.json")
    return 0
if __name__=="__main__":
    raise SystemExit(main())
