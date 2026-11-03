"""CLI for non-overwriting, literal recording-list coverage plans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reclist_budget import plan_reclist_file


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="phonoweave plan-literal-reclist")
    p.add_argument("reclist", type=Path)
    p.add_argument("--budget", required=True, type=int, help="maximum lines to select")
    p.add_argument("--separator", choices=("whitespace", "underscore"),
                   default="whitespace")
    p.add_argument("--encoding", help="default: UTF-8, then CP932")
    p.add_argument("--no-pairs", action="store_true",
                   help="cover only individual literal tokens")
    p.add_argument("--token-weight", type=int, default=1)
    p.add_argument("--pair-weight", type=int, default=2)
    p.add_argument("--output", "-o", type=Path, required=True,
                   help="new JSON path; refuses overwrite")
    args = p.parse_args(argv)
    try:
        report = plan_reclist_file(
            args.reclist, args.budget,
            separator=args.separator, encoding=args.encoding,
            include_pairs=not args.no_pairs, token_weight=args.token_weight,
            pair_weight=args.pair_weight,
        )
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
    except (OSError, UnicodeError, LookupError, ValueError) as exc:
        p.error(str(exc))
    c = report["coverage"]
    print(f"Selected {report['selected_count']} recording lines")
    print(f"Literal tokens: {c['token_types']}/{c['token_types_total']}")
    print(f"Adjacent surface pairs: {c['adjacent_pair_types']}/{c['adjacent_pair_types_total']}")
    print(f"Output: {output}")
    print("No phonetic/alias/OTO decisions were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
