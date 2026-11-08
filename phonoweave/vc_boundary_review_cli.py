"""CLI for source-verified manual VC boundary review and validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .vc_boundary_adjudication import (
    prepare_manual_boundary_review, write_validation_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave vc-review")
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("inspect", help="generate unverified boundary cue JSON")
    inspect.add_argument("--pack", required=True, type=Path)
    inspect.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("validate", help="validate six manually marked excerpts")
    verify.add_argument("--pack", required=True, type=Path)
    verify.add_argument("--annotations", required=True, type=Path)
    verify.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "inspect":
            model = prepare_manual_boundary_review(args.pack)
            source = args.pack.expanduser().resolve(strict=True)
            dest = args.output.expanduser().resolve()
            if dest == source or source in dest.parents:
                raise ValueError("output cannot be inside source review package")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("x", encoding="utf-8") as stream:
                json.dump(model, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            for row in model["clips"]:
                cue = row["screening"]
                print(f"{row['original_alias']}: {cue['status']} {cue['bracket_ms']}")
            print("Candidate brackets are NOT verified phonetic boundaries.")
        else:
            report = write_validation_report(args.pack, args.annotations, args.output)
            print(f"Completed reviews: {report['reviewed_rows']}/{report['target_rows']}")
            print(f"Status: {report['status']}")
            print("No split/share decision or synthesis benefit inferred.")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
