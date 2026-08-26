from __future__ import annotations

import argparse
from pathlib import Path

from .fh_fricative import analyze_fh_fricative


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave analyze-fh-fricative")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--base", choices=("f", "h"), required=True)
    args = parser.parse_args(argv)

    result = analyze_fh_fricative(args.voicebank, args.base)
    counts = ", ".join(f"{name}={count}" for name, count in result.counts.items())

    print(f"Base unit: {result.base_unit}")
    print("Role scope: internal")
    print(f"Samples: {result.samples}")
    print(f"Skipped: {result.skipped}")
    print(f"Duplicate observations removed: {result.duplicate_observations_removed}")
    print(f"Ambiguous segments removed: {result.ambiguous_segments_removed}")
    print(f"Ambiguous observations removed: {result.ambiguous_observations_removed}")
    print(f"Families: {counts}")
    print()

    if result.cross_oto_set_balanced_accuracy is None:
        print("Cross-OTO-set balanced accuracy: n/a")
    else:
        print(
            "Cross-OTO-set balanced accuracy: "
            f"{result.cross_oto_set_balanced_accuracy:.3f}"
        )
        held = ", ".join(
            f"{name}={score:.3f}"
            for name, score in sorted(result.cross_by_oto_set.items())
        )
        print(f"Held out: {held}")

    if result.stratified_distance is None:
        print("Stratified inference: n/a")
        return 0

    print(
        "Stratified: "
        f"distance={result.stratified_distance:.3f}, "
        f"p={result.stratified_permutation_p:.4f}, "
        f"OTO sets={len(result.oto_sets)}"
    )
    ranked = sorted(
        result.stratified_effects.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    if ranked:
        print("Effects (rounded - other):")
        for name, value in ranked:
            print(
                f"  {name}: {value:+.3f} "
                f"({result.effect_sign_agreement.get(name, 0)}/{len(result.oto_sets)})"
            )

    print("OTO sets:")
    for item in result.oto_sets:
        print(
            f"  {item.oto_set}: rounded={item.rounded_count}, "
            f"other={item.other_count}, distance={item.distance:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
