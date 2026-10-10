from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_cv_consistency import evaluate_cv_consistency

def main(argv=None):
    p=argparse.ArgumentParser(description="Evaluate onset/CV acoustic consistency")
    p.add_argument("session",type=Path)
    args=p.parse_args(argv)
    r=evaluate_cv_consistency(args.session)
    print("Session:",r["session_id"])
    print("Summary:",r["summary"])
    for row in r["negatives"]:
        print(f"negative {row['category']}: {row['status']} ({row['reason']})")
    print("Saved: analysis/calibration_cv_consistency_v2.1.json")
if __name__=="__main__":main()
