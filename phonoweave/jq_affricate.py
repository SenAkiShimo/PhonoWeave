from __future__ import annotations

import argparse
from pathlib import Path

from .affricate import analyze_affricate_contrast


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.jq_affricate")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--base", required=True, choices=("j", "q"))
    args = parser.parse_args()

    result = analyze_affricate_contrast(args.voicebank, args.base)
    print(f"Base unit: {result.base_unit}")
    print(f"Samples: {result.samples}")
    print(f"Skipped: {result.skipped}")
    if result.mean_distance is not None:
        print(f"Mean distance: {result.mean_distance:.3f}")
        print(f"Distance CV: {result.distance_cv:.3f}")
    if result.cross_subbank_balanced_accuracy is not None:
        print(f"Cross-subbank balanced accuracy: {result.cross_subbank_balanced_accuracy:.3f}")
        details = ", ".join(f"{name}={score:.3f}" for name, score in result.cross_by_subbank.items())
        print(f"  held out: {details}")
    print()

    for item in result.subbanks:
        print(
            f"{item.subbank}: plain={item.plain_count}, rounded={item.rounded_count}, "
            f"distance={item.distance:.3f}, "
            f"loo_balanced_accuracy={item.loo_balanced_accuracy:.3f}, "
            f"permutation_p={item.permutation_p:.4f}"
        )
        print(
            f"  centroid_hz: {item.mean_plain['centroid_hz']:.1f} -> "
            f"{item.mean_rounded['centroid_hz']:.1f} "
            f"(effect={item.effects['centroid_hz']:+.3f})"
        )
        print(
            f"  high_band_ratio: {item.mean_plain['high_band_ratio']:.3f} -> "
            f"{item.mean_rounded['high_band_ratio']:.3f} "
            f"(effect={item.effects['high_band_ratio']:+.3f})"
        )
        print(
            f"  frication_duration_ms: {item.mean_plain['frication_duration_ms']:.1f} -> "
            f"{item.mean_rounded['frication_duration_ms']:.1f} "
            f"(effect={item.effects['frication_duration_ms']:+.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
