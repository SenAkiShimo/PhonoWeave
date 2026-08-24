from __future__ import annotations

import argparse
from pathlib import Path

from .nasal_relevance import nasal_relevance_test


def _value(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.nasal_relevance_cli")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--base", choices=("m", "n"), default="n")
    args = parser.parse_args(argv)

    result = nasal_relevance_test(args.voicebank, args.base)
    print(f"Base unit: {result.base_unit}")
    print(f"Samples: {result.samples}")
    print(f"Skipped: {result.skipped}")
    print(f"Duplicate observations removed: {result.duplicate_observations_removed}")
    print(f"Ambiguous segments removed: {result.ambiguous_segments_removed}")
    print("Signal-based synthesis relevance proxy only; no inventory decision is produced.")
    print()

    for comparison in result.comparisons:
        print(
            f"{comparison.target_family} <- {comparison.substitution_family}: "
            f"targets={comparison.targets}"
        )
        print(
            "  body: "
            f"spectral_delta={_value(comparison.mean_body_spectral_delta)}, "
            f"p={_value(comparison.body_spectral_p)}, "
            f"holm_p={_value(comparison.body_spectral_p_holm)}, "
            f"periodicity_delta={_value(comparison.mean_body_periodicity_delta)}, "
            f"p={_value(comparison.body_periodicity_p)}, "
            f"balance_delta={_value(comparison.mean_body_balance_delta)}, "
            f"p={_value(comparison.body_balance_p)}"
        )
        print(
            "  transition: "
            f"spectral_delta={_value(comparison.mean_transition_spectral_delta)}, "
            f"p={_value(comparison.transition_spectral_p)}, "
            f"holm_p={_value(comparison.transition_spectral_p_holm)}, "
            f"periodicity_delta={_value(comparison.mean_transition_periodicity_delta)}, "
            f"p={_value(comparison.transition_periodicity_p)}, "
            f"balance_delta={_value(comparison.mean_transition_balance_delta)}, "
            f"p={_value(comparison.transition_balance_p)}"
        )
        if comparison.oto_sets:
            print("  oto_sets:")
            for item in comparison.oto_sets:
                print(
                    f"    {item.oto_set}: targets={item.targets}, "
                    f"body_spectral_delta={item.mean_body_spectral_delta:+.4f}, "
                    f"transition_spectral_delta={item.mean_transition_spectral_delta:+.4f}, "
                    f"transition_periodicity_delta={item.mean_transition_periodicity_delta:+.4f}, "
                    f"transition_balance_delta={item.mean_transition_balance_delta:+.4f}"
                )
        print()

    print("Bidirectional pair summary")
    for pair in result.pairs:
        print(
            f"  {pair.left} vs {pair.right}: "
            f"transition_positive={pair.both_transition_positive}, "
            f"transition_holm_significant={pair.both_transition_holm_significant}, "
            f"body_positive={pair.both_body_positive}, "
            f"body_holm_significant={pair.both_body_holm_significant}, "
            f"all_oto_sets_transition_positive={pair.all_oto_sets_transition_positive}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
