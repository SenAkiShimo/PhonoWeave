"""Read-only comparison of speaker candidate and literal recording budgets."""
from __future__ import annotations

import argparse
from pathlib import Path

from .reclist_budget_comparison import (
    compare_literal_budget, load_verified_review_pack, write_comparison,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phonoweave compare-literal-budget",
        description="Compare literal and speaker candidate policies at exactly matched line counts",
    )
    parser.add_argument("--pack", required=True, type=Path,
                        help="existing output from compile-review-pack")
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="new output directory outside the source pack")
    parser.add_argument("--token-weight", type=int, default=1)
    parser.add_argument("--pair-weight", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        source = args.pack.expanduser().resolve(strict=True)
        output = args.output.expanduser().resolve()
        if source == output or source in output.parents:
            raise ValueError("comparison output must be outside the source review pack")
        blueprint, plan, units, links, provenance = load_verified_review_pack(source)
        result = compare_literal_budget(
            blueprint, plan, units, links,
            token_weight=args.token_weight, pair_weight=args.pair_weight,
        )
        result["verified_source"] = provenance
        written = write_comparison(result, output)
    except (OSError, ValueError, UnicodeError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print("Matched line-count audit:", written)
    print("Common-prefix comparisons:", len(result["matched_curve"]))
    print("Equal-cost paired prefixes end at:", result["matched_prefix_count"])
    print("Research-only; no alias, WAV or OTO changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
