"""Explicit manual review for one frozen action plan; no source changes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .recording_scheme_decisions import (
    compile_decisions, save_review_template, write_reviewed_scheme,
)


def main(argv=None):
    parser=argparse.ArgumentParser(description="Review a frozen recording action plan")
    sub=parser.add_subparsers(dest="command",required=True)
    t=sub.add_parser("template")
    t.add_argument("--plan",type=Path,required=True)
    t.add_argument("--output",type=Path,required=True)
    c=sub.add_parser("compile")
    c.add_argument("--plan",type=Path,required=True)
    c.add_argument("--decisions",type=Path,required=True)
    c.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        plan=json.loads(args.plan.expanduser().read_text(encoding="utf-8"))
        if args.command=="template":
            save_review_template(plan,args.output)
            print("Created non-overwriting review template:",args.output)
        else:
            decisions=json.loads(args.decisions.expanduser().read_text(encoding="utf-8"))
            result=compile_decisions(plan,decisions)
            output=write_reviewed_scheme(result,args.output)
            print("Reviewed recording scheme:",output)
            print("Reviewed pitch cells:",result["reviewed_pitch_cells"])
            print("Approved diagnostic lines:",result["approved_diagnostic_lines"])
            print("Production alias/OTO export: disabled")
    except (ValueError,FileNotFoundError,FileExistsError,KeyError,
            TypeError,json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
