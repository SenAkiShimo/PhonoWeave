from __future__ import annotations

import argparse
from pathlib import Path

from .rhotic import analyze_rhotic_contrast


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.rhotic_inventory")
    parser.add_argument("voicebank", type=Path)
    args = parser.parse_args()

    result = analyze_rhotic_contrast(args.voicebank)
    print("Rhotic inventory candidates:")
    print(f"  three-way cross-subbank accuracy: {result.cross_subbank_balanced_accuracy:.3f}")
    print()

    for partition in result.partitions:
        left = "+".join(partition.left)
        right = "+".join(partition.right)
        score = partition.cross_subbank_balanced_accuracy
        print(f"  {left} | {right}: cross_subbank_balanced_accuracy={score:.3f}")
        if partition.cross_by_subbank:
            details = ", ".join(
                f"{name}={value:.3f}"
                for name, value in partition.cross_by_subbank.items()
            )
            print(f"    held out: {details}")
        for item in partition.subbanks:
            print(
                f"    {item.subbank}: {item.left_count}/{item.right_count}, "
                f"loo_balanced_accuracy={item.loo_balanced_accuracy:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
