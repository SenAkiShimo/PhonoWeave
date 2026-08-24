from __future__ import annotations

import argparse
from pathlib import Path

from .lateral_relevance import lateral_relevance_test


def _value(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.lateral_relevance_cli")
    parser.add_argument("voicebank", type=Path)
    args = parser.parse_args(argv)

    result = lateral_relevance_test(args.voicebank)
    print("Base unit: l")
    print(f"Samples: {result.samples}")
    print(f"Skipped: {result.skipped}")
    print(f"Duplicate segments removed: {result.duplicate_segments}")
    print("Signal-based synthesis relevance proxy only; no inventory decision is produced.")
    print()

    for role in result.roles:
        print(role.role)
        for comparison in role.comparisons:
            print(
                f"  {comparison.target_family} <- {comparison.substitution_family}: "
                f"targets={comparison.targets}"
            )
            print(
                "    body: "
                f"spectral_delta={_value(comparison.mean_body_spectral_delta)}, "
                f"p={_value(comparison.body_spectral_p)}, "
                f"periodicity_delta={_value(comparison.mean_body_periodicity_delta)}, "
                f"p={_value(comparison.body_periodicity_p)}"
            )
            print(
                "    boundary: "
                f"spectral_delta={_value(comparison.mean_boundary_spectral_delta)}, "
                f"p={_value(comparison.boundary_spectral_p)}, "
                f"holm_p={_value(comparison.boundary_spectral_p_holm)}, "
                f"periodicity_delta={_value(comparison.mean_boundary_periodicity_delta)}, "
                f"p={_value(comparison.boundary_periodicity_p)}"
            )
            if comparison.oto_sets:
                print("    oto_sets:")
                for item in comparison.oto_sets:
                    print(
                        f"      {item.oto_set}: targets={item.targets}, "
                        f"body_spectral_delta={item.mean_body_spectral_delta:+.4f}, "
                        f"body_periodicity_delta={item.mean_body_periodicity_delta:+.4f}, "
                        f"boundary_spectral_delta={item.mean_boundary_spectral_delta:+.4f}, "
                        f"boundary_periodicity_delta={item.mean_boundary_periodicity_delta:+.4f}"
                    )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
