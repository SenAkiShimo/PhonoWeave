"""Compile a read-only source-derived reclist blueprint."""
import argparse
import json
from pathlib import Path
from .reference_reclist_blueprint import compile_reference_reclist, write_reference_reclist

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.alias_records.expanduser().open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    report = compile_reference_reclist(records)
    path = write_reference_reclist(report, args.output)
    print("Reference reclist:", path)
    print("Fully pitch-covered sequences:", report["all_pitch_reference_sequences"])
    print("Incomplete/ambiguous sequences:", report["interpretable_sequences"] - report["all_pitch_reference_sequences"])
    print("Source recordings and OTO files unchanged")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
