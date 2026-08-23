from __future__ import annotations

import argparse
from pathlib import Path

from .rhotic_relevance import rhotic_relevance_test


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.rhotic_relevance_cli")
    parser.add_argument("voicebank", type=Path)
    args = parser.parse_args()

    result = rhotic_relevance_test(args.voicebank)
    print("Rhotic synthesis relevance proxy:")
    print(f"  samples: {result.samples}")
    print(f"  skipped: {result.skipped}")
    print()

    for comparison in result.comparisons:
        label = f"{comparison.target_context} <- {comparison.substitution_context}"
        print(label)
        print(f"  targets: {comparison.targets}")
        print(f"  mean delta: {_fmt(comparison.mean_delta, signed=True)}")
        if comparison.mean_relative_delta is None:
            print("  mean relative delta: n/a")
        else:
            print(f"  mean relative delta: {comparison.mean_relative_delta:+.1%}")
        print(f"  permutation p: {_fmt(comparison.permutation_p)}")
        for item in comparison.subbanks:
            print(
                f"  {item.subbank}: n={item.targets}, "
                f"control={item.mean_control_penalty:.4f}, "
                f"substitution={item.mean_substitution_penalty:.4f}, "
                f"delta={item.mean_delta:+.4f} ({item.mean_relative_delta:+.1%}), "
                f"p={item.permutation_p:.4f}"
            )
        print()

    print("Interpretation:")
    print("  positive delta = cross-context substitution is worse than same-context control")
    print("  front comparisons test split necessity")
    print("  plain/rounded comparisons test merge safety in both directions")
    print("  this remains a signal-level proxy, not a perceptual result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
