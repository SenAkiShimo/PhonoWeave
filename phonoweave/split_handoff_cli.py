"""Emit annotator and tuner handoff from a frozen speaker budget."""
import argparse
import csv
import json
from pathlib import Path

from .split_handoff import compile_split_handoff, write_split_handoff


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compile review-only split handoff")
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--action-plan", type=Path, required=True)
    parser.add_argument("--alias-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.budget.expanduser().open(encoding="utf-8") as file:
        budget = json.load(file)
    with args.units.expanduser().open(encoding="utf-8", newline="") as file:
        units = list(csv.DictReader(file))
    with args.action_plan.expanduser().open(encoding="utf-8") as file:
        actions = json.load(file)
    with args.alias_records.expanduser().open(encoding="utf-8") as file:
        sources = [json.loads(line) for line in file if line.strip()]
    handoff = compile_split_handoff(budget, units, actions, sources)
    location = write_split_handoff(handoff, args.output)
    print("Annotator/tuner handoff:", location)
    print("Units:", handoff["unit_count"])
    print("Provisional split units:", handoff["provisional_split_units"])
    print("Production alias/OTO export: disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
