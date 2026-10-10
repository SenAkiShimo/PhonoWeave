"""Text-only recording list import diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reclist import load_reclist


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave inspect-reclist")
    parser.add_argument("reclist", type=Path)
    parser.add_argument("--separator", choices=("whitespace", "underscore"), default="whitespace")
    parser.add_argument("--encoding", default=None, help="default: UTF-8, then CP932")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        document = load_reclist(args.reclist, separator=args.separator, encoding=args.encoding)
    except (OSError, UnicodeError, LookupError, ValueError) as exc:
        parser.error(str(exc))

    summary = document.summary()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print(f"Reclist: {summary['source']}")
    print(f"Encoding: {summary['encoding']}")
    print(f"Separator: {summary['separator']}")
    print(f"Lines: {summary['total_lines']} (recording={summary['recording_lines']}, "
          f"comment={summary['comment_lines']}, blank={summary['blank_lines']})")
    print(f"Tokens: {summary['token_occurrences']} ({summary['unique_tokens']} unique)")
    print("Tokens are literal text, not inferred phonemes or synthesis units.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
