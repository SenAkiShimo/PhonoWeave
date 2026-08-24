from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lateral_timecourse import analyze_lateral_timecourse


def _payload(result) -> dict[str, object]:
    return {
        "samples": result.samples,
        "skipped": result.skipped,
        "duplicate_segments": result.duplicate_segments,
        "roles": [
            {
                "role": role.role,
                "counts": role.counts,
                "windows": [
                    {
                        "name": window.name,
                        "start_fraction": window.start_fraction,
                        "end_fraction": window.end_fraction,
                        "pairwise": [
                            {
                                "left": pair.left,
                                "right": pair.right,
                                "cross_oto_set_balanced_accuracy": pair.cross_oto_set_balanced_accuracy,
                                "cross_by_oto_set": pair.cross_by_oto_set,
                                "stratified_distance": pair.stratified_distance,
                                "stratified_permutation_p": pair.stratified_permutation_p,
                                "stratified_p_holm": pair.stratified_p_holm,
                                "stratified_effects": pair.stratified_effects,
                                "effect_sign_agreement": pair.effect_sign_agreement,
                                "oto_sets": pair.oto_sets,
                            }
                            for pair in window.pairwise
                        ],
                    }
                    for window in role.windows
                ],
            }
            for role in result.roles
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.lateral_timecourse_cli")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = analyze_lateral_timecourse(args.voicebank)
    payload = _payload(result)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("Base unit: l")
    print(f"Samples: {result.samples}")
    print(f"Skipped: {result.skipped}")
    print(f"Duplicate segments removed: {result.duplicate_segments}")
    print("Temporal localization only; no synthesis or inventory decision is produced.")
    print()

    for role in result.roles:
        counts = ", ".join(f"{name}={count}" for name, count in role.counts.items())
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
                    f"    {pair.left} vs {pair.right}: cross_ba={cross}, "
                    f"distance={distance}, p={raw_p}, holm_p={holm_p}"
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
