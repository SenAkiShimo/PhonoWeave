from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_frication_shoulder_eval import compare_shoulder_session

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("session",type=Path)
    args=parser.parse_args(argv)
    result=compare_shoulder_session(args.session)
    print("Session:",result["session_id"])
    print("Comparison:",result["summary"])
    for row in result["rows"]:
        print(f"{row['base_unit']} {row['context']} #{row['occurrence']}: "
              f"core={row['legacy_core_absolute_ms']} shoulder={row['experimental_shoulder_absolute_ms']} "
              f"shift={row['backtrace_ms']} gold={row['manual_absolute_ms']} "
              f"core_error={row['core_signed_error_ms']} shoulder_error={row['shoulder_signed_error_ms']}")
    print("Saved: analysis/calibration_frication_shoulder_v1.5.json")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
