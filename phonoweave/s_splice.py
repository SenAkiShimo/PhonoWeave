from __future__ import annotations

import argparse
from pathlib import Path

from .splice import splice_relevance_test


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.s_splice")
    parser.add_argument("voicebank", type=Path)
    args = parser.parse_args()

    result = splice_relevance_test(args.voicebank, "s")

    print("Base unit: s")
    print(f"Rounded targets: {result.targets}")
    print(f"Skipped: {result.skipped}")
    if result.mean_delta is not None:
        print(f"Mean substitution delta: {result.mean_delta:+.4f}")
        print(f"Mean relative delta: {result.mean_relative_delta:+.1%}")
        print(f"Permutation p: {result.permutation_p:.4f}")
    print()

    for item in result.subbanks:
        print(f"{item.subbank}: targets={item.targets}")
        print(f"  natural boundary: {item.mean_natural_penalty:.4f}")
        print(f"  rounded control: {item.mean_rounded_control_penalty:.4f}")
        print(f"  plain substitution: {item.mean_plain_substitution_penalty:.4f}")
        print(f"  delta: {item.mean_delta:+.4f} ({item.mean_relative_delta:+.1%})")
        print(f"  permutation_p: {item.permutation_p:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
