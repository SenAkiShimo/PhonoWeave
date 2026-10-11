"""Export the versioned experimental calibration result manifest."""
from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_results import export_calibration_results

def main(argv=None):
    parser=argparse.ArgumentParser(description="Export research-only calibration results")
    parser.add_argument("session",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    data=export_calibration_results(args.session,output=args.output)
    print("Session:",data["session_id"])
    print("Schema:",data["schema"],data["schema_version"])
    print("Summary:",data["summary"])
    print("Status: research_only; human review required")
    print("Default voicebank decisions modified: False")

if __name__=="__main__":
    main()
