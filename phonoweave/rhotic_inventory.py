from __future__ import annotations

import argparse
from pathlib import Path

from .rhotic import analyze_rhotic_contrast


def _pair_key(left: str, right: str) -> frozenset[str]:
    return frozenset((left, right))


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.rhotic_inventory")
    parser.add_argument("voicebank", type=Path)
    args = parser.parse_args()

    result = analyze_rhotic_contrast(args.voicebank)
    pairwise = {
        _pair_key(pair.left, pair.right): pair
        for pair in result.pairwise
    }

    print("Rhotic inventory candidates:")
    print(f"  three-way cross-subbank accuracy: {result.cross_subbank_balanced_accuracy:.3f}")
    print()

    ranked: list[tuple[float, str]] = []
    for partition in result.partitions:
        left = "+".join(partition.left)
        right = "+".join(partition.right)
        external = partition.cross_subbank_balanced_accuracy

        merged = partition.left if len(partition.left) > 1 else partition.right
        if len(merged) != 2:
            continue
        internal_pair = pairwise[_pair_key(merged[0], merged[1])]
        internal = internal_pair.cross_subbank_balanced_accuracy
        if external is None or internal is None:
            continue

        margin = external - internal
        significant_layers = sum(
            item.permutation_p < 0.05
            for item in internal_pair.subbanks
        )
        ranked.append((margin, f"{left} | {right}"))

        print(f"  {left} | {right}")
        print(f"    external separation: {external:.3f}")
        print(
            f"    internal separability ({'+'.join(merged)}): "
            f"{internal:.3f}"
        )
        print(f"    merge margin: {margin:+.3f}")
        print(
            f"    internal significant pitch layers: "
            f"{significant_layers}/{len(internal_pair.subbanks)}"
        )
        if partition.cross_by_subbank:
            details = ", ".join(
                f"{name}={value:.3f}"
                for name, value in partition.cross_by_subbank.items()
            )
            print(f"    partition held out: {details}")
        if internal_pair.cross_by_subbank:
            details = ", ".join(
                f"{name}={value:.3f}"
                for name, value in internal_pair.cross_by_subbank.items()
            )
            print(f"    merged-pair held out: {details}")
        print()

    if ranked:
        ranked.sort(reverse=True)
        print(f"Best structural merge candidate: {ranked[0][1]}")
        print("This is an exploratory ranking, not a final synthesis decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
