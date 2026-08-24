from __future__ import annotations

import argparse
from pathlib import Path

from .nasal import analyze_nasal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.nasal_cli")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--base", choices=("m", "n"), default="m")
    args = parser.parse_args(argv)

    result = analyze_nasal(args.voicebank, args.base)
    print(f"Base unit: {result.base_unit}")
    print(f"Samples: {result.samples}")
    print(f"Skipped: {result.skipped}")
    print(
        "Duplicate observations removed: "
        f"{result.duplicate_observations_removed}"
    )
    print(f"Ambiguous segments removed: {result.ambiguous_segments_removed}")
    print("Experimental acoustic evidence only; no inventory decision is produced.")
    print()

    for role in result.roles:
        counts = ", ".join(
            f"{name}={count}"
            for name, count in role.counts.items()
        )
        print(f"{role.role}: {counts}")
        for window in role.windows:
            print(
                f"  {window.name} "
                f"({window.start_fraction:.2f}-{window.end_fraction:.2f})"
            )
            for pair in window.pairwise:
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
                p_value = (
                    f"{pair.stratified_permutation_p:.4f}"
                    if pair.stratified_permutation_p is not None
                    else "n/a"
                )
                holm = (
                    f"{pair.stratified_p_holm:.4f}"
                    if pair.stratified_p_holm is not None
                    else "n/a"
                )
                print(
                    f"    {pair.left} vs {pair.right}: "
                    f"cross_ba={cross}, distance={distance}, "
                    f"p={p_value}, holm_p={holm}, "
                    f"oto_sets={pair.oto_sets}"
                )
                if pair.cross_by_oto_set:
                    held = ", ".join(
                        f"{name}={score:.3f}"
                        for name, score in pair.cross_by_oto_set.items()
                    )
                    print(f"      held out: {held}")
                ranked = sorted(
                    pair.stratified_effects.items(),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )[:4]
                if ranked:
                    effects = ", ".join(
                        f"{name}={value:+.3f} "
                        f"({pair.effect_sign_agreement.get(name, 0)}/{pair.oto_sets})"
                        for name, value in ranked
                    )
                    print(f"      effects: {effects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
