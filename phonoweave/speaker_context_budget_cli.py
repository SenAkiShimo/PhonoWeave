"""Speaker-context candidate budget planner CLI."""
import argparse
import csv
import json
from pathlib import Path
from .speaker_context_budget import compile_speaker_context_budget, write_speaker_context_budget

def main(argv=None):
    parser=argparse.ArgumentParser(description="Speaker context evidence-aware recording budget")
    parser.add_argument("--blueprint",required=True,type=Path)
    parser.add_argument("--units",required=True,type=Path)
    parser.add_argument("--internal-links",required=True,type=Path)
    parser.add_argument("--budget",type=int,required=True)
    parser.add_argument("--repeat-target",type=int,default=3)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    with args.blueprint.expanduser().open(encoding="utf-8") as f:
        blueprint=json.load(f)
    with args.units.expanduser().open(newline="",encoding="utf-8") as f:
        units=list(csv.DictReader(f))
    with args.internal_links.expanduser().open(newline="",encoding="utf-8") as f:
        links=list(csv.DictReader(f))
    result=compile_speaker_context_budget(blueprint,units,links,args.budget,
                                          repeat_target=args.repeat_target)
    path=write_speaker_context_budget(result,args.output)
    print("Speaker context pilot:",path)
    print("Selected lines:",result["selection"]["selected_line_count"])
    print("Candidate context cells:",result["protected_observable_unit_pitch_cells"])
    print("Production export: disabled")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
