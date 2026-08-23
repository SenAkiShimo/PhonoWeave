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
        print(
            f"  body spectral delta: {_fmt(comparison.mean_body_spectral_delta, signed=True)}, "
            f"p={_fmt(comparison.body_spectral_p)}"
        )
        print(
            f"  body periodicity delta: {_fmt(comparison.mean_body_periodicity_delta, signed=True)}, "
            f"p={_fmt(comparison.body_periodicity_p)}"
        )
        print(
            f"  boundary delta: {_fmt(comparison.mean_boundary_delta, signed=True)}, "
            f"p={_fmt(comparison.boundary_p)}"
        )
        for item in comparison.subbanks:
            print(f"  {item.subbank}: n={item.targets}")
            print(
                f"    body spectral: control={item.mean_control_body_spectral:.4f}, "
                f"substitution={item.mean_substitution_body_spectral:.4f}, "
                f"delta={item.mean_body_spectral_delta:+.4f}, p={item.body_spectral_p:.4f}"
            )
            print(
                f"    body periodicity: control={item.mean_control_body_periodicity:.4f}, "
                f"substitution={item.mean_substitution_body_periodicity:.4f}, "
                f"delta={item.mean_body_periodicity_delta:+.4f}, p={item.body_periodicity_p:.4f}"
            )
            print(
                f"    boundary: control={item.mean_control_boundary:.4f}, "
                f"substitution={item.mean_substitution_boundary:.4f}, "
                f"delta={item.mean_boundary_delta:+.4f}, p={item.boundary_p:.4f}"
            )
        print()

    print("Interpretation:")
    print("  positive body delta = substituted realization is less target-like than same-context control")
    print("  positive boundary delta = substituted realization creates a larger local splice mismatch")
    print("  body evidence is primary for rhotics; boundary evidence is secondary")
    print("  plain/rounded merge safety requires both substitution directions to remain small")
    print("  this remains a signal-level proxy, not a perceptual result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
