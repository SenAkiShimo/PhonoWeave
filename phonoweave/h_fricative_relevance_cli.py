from __future__ import annotations

import argparse
from pathlib import Path

from .h_fricative_relevance import h_fricative_relevance_test


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave analyze-h-fricative-relevance")
    parser.add_argument("voicebank", type=Path)
    args = parser.parse_args(argv)

    result = h_fricative_relevance_test(args.voicebank)
    print("Base unit: h")
    print("Role scope: internal")
    print(f"Acoustic gate passed: {result.acoustic_gate_passed}")
    if result.acoustic_p is not None:
        print(f"Acoustic p: {result.acoustic_p:.4f}")
    if not result.acoustic_gate_passed:
        return 0

    print(f"Samples: {result.samples}")
    print(f"Skipped: {result.skipped}")
    print(f"Duplicate observations removed: {result.duplicate_observations_removed}")
    print(f"Ambiguous segments removed: {result.ambiguous_segments_removed}")
    print(f"Ambiguous observations removed: {result.ambiguous_observations_removed}")
    print()

    for item in result.comparisons:
        boundary = (
            f"{item.mean_boundary_spectral_delta:+.4f}"
            if item.mean_boundary_spectral_delta is not None
            else "n/a"
        )
        raw_p = (
            f"{item.boundary_spectral_p:.4f}"
            if item.boundary_spectral_p is not None
            else "n/a"
        )
        holm_p = (
            f"{item.boundary_spectral_p_holm:.4f}"
            if item.boundary_spectral_p_holm is not None
            else "n/a"
        )
        body = (
            f"{item.mean_body_spectral_delta:+.4f}"
            if item.mean_body_spectral_delta is not None
            else "n/a"
        )
        body_p = (
            f"{item.body_spectral_p:.4f}"
            if item.body_spectral_p is not None
            else "n/a"
        )
        print(f"{item.target_family} <- {item.substitution_family}")
        print(f"  targets: {item.targets}")
        print(f"  boundary delta: {boundary}, p={raw_p}, holm_p={holm_p}")
        print(f"  body delta: {body}, p={body_p}")
        for oto in item.oto_sets:
            print(
                f"  {oto.oto_set}: targets={oto.targets}, "
                f"boundary_delta={oto.mean_boundary_spectral_delta:+.4f}, "
                f"body_delta={oto.mean_body_spectral_delta:+.4f}"
            )
        print()

    if result.pair is not None:
        print("Pair summary")
        print(f"  both boundary positive: {result.pair.both_boundary_positive}")
        print(
            "  both boundary Holm significant: "
            f"{result.pair.both_boundary_holm_significant}"
        )
        print(
            "  all OTO sets boundary positive: "
            f"{result.pair.all_oto_sets_boundary_positive}"
        )
        print(
            "  split supported under proxy: "
            f"{result.pair.split_supported_under_proxy}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
