"""Audit explicitly annotated phonetic transitions in a text reclist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phonetic_transition_graph import compile_transition_budget
from .reclist import load_reclist


def _no_duplicate_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON mapping key: {key!r}")
        obj[key] = value
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave plan-phonetic-transitions")
    parser.add_argument("reclist", type=Path)
    parser.add_argument("--lexicon", type=Path, required=True,
                        help="explicit user-reviewed token-to-phone mapping JSON")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--separator", choices=("whitespace", "underscore"),
                        default="whitespace")
    parser.add_argument("--encoding")
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        doc = load_reclist(args.reclist, separator=args.separator, encoding=args.encoding)
        lexicon = json.loads(args.lexicon.read_text(encoding="utf-8"),
                             object_pairs_hook=_no_duplicate_keys)
        result = compile_transition_budget(doc, lexicon, args.budget)
        output = args.output.expanduser().resolve()
        input_paths = (args.reclist.expanduser().resolve(),
                       args.lexicon.expanduser().resolve())
        if output in input_paths:
            raise ValueError("refusing to write report over an input file")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, LookupError, UnicodeError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print("Phonetic policy:", result["phonetic_policy"]["selected_count"], "lines")
    print("Literal policy:", result["literal_policy"]["selected_count"], "lines")
    print("Matched prefixes:", len(result["matched_prefix_curve"]))
    print("No automatic phonological split/merge or production alias output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
