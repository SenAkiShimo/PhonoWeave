"""Budget-capped source recording sequence proposal."""
import argparse
import json
from pathlib import Path
from .recording_scheme_budget import compile_recording_budget, write_recording_budget, MODES

def main(argv=None):
    parser=argparse.ArgumentParser(description="Generate a coverage-proxy pilot reclist")
    parser.add_argument("--blueprint",type=Path,required=True)
    parser.add_argument("--budget",type=int,required=True)
    parser.add_argument("--mode",choices=MODES,default="syllables_and_starts")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    with args.blueprint.expanduser().open(encoding="utf-8") as f:
        blueprint=json.load(f)
    report=compile_recording_budget(blueprint,args.budget,args.mode)
    destination=write_recording_budget(report,args.output)
    print("Pilot recording budget plan:",destination)
    print("Selected source sequences:",len(report["selected_reclist"]))
    print("Production readiness: false; original bank unchanged")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
