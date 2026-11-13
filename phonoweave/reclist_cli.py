from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .reclist import (
    UnitMap,
    build_reclist,
    missing_requirements,
    observed_syllables,
    reclist_report,
)
from .synthesis_inventory import build_synthesis_inventory, load_synthesis_inventory


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave build-reclist")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("reclist.txt"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-syllables", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    syllables = observed_syllables(args.voicebank)
    if not syllables:
        print("No Mandarin syllables recognized in the voicebank.", file=sys.stderr)
        return 1

    if args.inventory is not None:
        inventory = load_synthesis_inventory(args.inventory)
    else:
        inventory = build_synthesis_inventory(args.voicebank)

    units = UnitMap(inventory)
    reclist = build_reclist(
        syllables, units, args.max_syllables, args.restarts, args.seed
    )
    missing = missing_requirements(reclist.lines, reclist.requirements, units)
    if any(missing.values()):
        print(f"Coverage check failed: {missing}", file=sys.stderr)
        return 1

    baseline_units = UnitMap(None)
    baseline = build_reclist(
        syllables, baseline_units, args.max_syllables, args.restarts, args.seed
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(reclist.text(), encoding="utf-8")

    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else output.with_suffix(".report.json")
    )
    report = reclist_report(reclist, syllables, units, inventory, baseline)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    requirements = report["requirements"]
    print(f"Syllables: {len(syllables)}")
    print(
        f"Requirements: cv={requirements['cv']} vc={requirements['vc']} "
        f"start={requirements['start']} end={requirements['end']}"
    )
    print(f"Lines: {len(reclist.lines)} (lower bound {reclist.lower_bound})")
    print(f"Baseline, one unit per onset: {len(baseline.lines)} lines")
    if units.unmapped:
        print(f"Unmapped onset/final pairs: {len(units.unmapped)}")
    print(f"Reclist: {output}")
    print(f"Report: {report_path}")
    return 0
