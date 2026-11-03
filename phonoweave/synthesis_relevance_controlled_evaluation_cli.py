"""CLI for preregistered S01-only /sh/ candidate split evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
from .synthesis_relevance_controlled_evaluation import run_controlled_proxy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phonoweave.synthesis_relevance_controlled_evaluation_cli",
        description="S01 training / S02 independent-session spectral-proxy benchmark",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, help="default: manifest parent")
    parser.add_argument("--output", type=Path, required=True,
                        help="new, non-overwriting JSON file")
    args = parser.parse_args(argv)
    try:
        report = run_controlled_proxy(args.manifest, args.output, root=args.root)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print("Status:", report["status"])
    print("Matched 0..4 split pair acoustic curves:", len(report["budget_curves"]))
    print("Render/listening validation: not performed")
    return 0 if report["status"] == "evaluable_acoustic_proxy" else 2


if __name__ == "__main__":
    raise SystemExit(main())
