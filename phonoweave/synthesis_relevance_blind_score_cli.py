"""Score independently collected blind listening votes."""
from __future__ import annotations

import argparse
from pathlib import Path
from .synthesis_relevance_blind_score import score_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Descriptive blinded listening score")
    parser.add_argument("--blind", required=True, type=Path)
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = score_directory(args.blind, args.private_key,
                                 args.response, args.output)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print("Listening responses:", report["listeners"])
    print("Pooled vs split votes:", report["votes"])
    print("These are descriptive counts, NOT perceptual significance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
