"""Command-line source alias precheck."""
import argparse
from pathlib import Path
from .recording_scheme_precheck import audit_files

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--alias-records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    a = parser.parse_args(argv)
    result = audit_files(a.plan.expanduser(), a.alias_records.expanduser(), a.output.expanduser())
    print("Checked review tasks:", result["review_tasks_checked"])
    print("Human decisions: unchanged")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
