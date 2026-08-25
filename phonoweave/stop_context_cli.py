from __future__ import annotations

import argparse
from pathlib import Path

from .stop_context import analyze_stop_context


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave analyze-stop-context")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--base", required=True, choices=("b", "p", "d", "t", "g", "k"))
    args = parser.parse_args(argv)

    result = analyze_stop_context(args.voicebank, args.base)
    print(f"Base unit: {result.base_unit}")
    print("Role: internal")
    print(f"Samples: {result.samples}")
    print(f"Skipped: {result.skipped}")
    print(f"Duplicate observations removed: {result.duplicate_observations_removed}")
    print(f"Ambiguous segments removed: {result.ambiguous_segments_removed}")
    print(f"Ambiguous observations removed: {result.ambiguous_observations_removed}")
    counts = ", ".join(f"{name}={count}" for name, count in result.counts.items())
    print(f"Counts: {counts}")
    print("Experimental acoustic evidence only; no synthesis-inventory decision is produced.")
    print()

    for pair in result.pairwise:
        cross = (
            f"{pair.cross_oto_set_balanced_accuracy:.3f}"
            if pair.cross_oto_set_balanced_accuracy is not None
            else "n/a"
        )
        distance = (
            f"{pair.stratified_distance:.3f}"
            if pair.stratified_distance is not None
            else "n/a"
        )
        raw_p = (
            f"{pair.stratified_permutation_p:.4f}"
            if pair.stratified_permutation_p is not None
            else "n/a"
        )
        holm_p = (
            f"{pair.stratified_p_holm:.4f}"
            if pair.stratified_p_holm is not None
            else "n/a"
        )
        print(
            f"{pair.left} vs {pair.right}: cross_ba={cross}, "
            f"distance={distance}, p={raw_p}, holm_p={holm_p}, "
            f"oto_sets={pair.oto_sets}"
        )
        if pair.cross_by_oto_set:
            held = ", ".join(
                f"{name}={score:.3f}"
                for name, score in pair.cross_by_oto_set.items()
            )
            print(f"  held out: {held}")
        ranked = sorted(
            pair.stratified_effects.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5]
        if ranked:
            effects = ", ".join(
                f"{name}={value:+.3f} "
                f"({pair.effect_sign_agreement.get(name, 0)}/{pair.oto_sets})"
                for name, value in ranked
            )
            print(f"  effects: {effects}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
