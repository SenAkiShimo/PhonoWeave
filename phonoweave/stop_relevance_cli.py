from __future__ import annotations

import argparse
from pathlib import Path

from .stop_relevance import stop_relevance_test


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave analyze-stop-relevance")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--base", required=True, choices=("b", "p", "d", "t", "g", "k"))
    args = parser.parse_args(argv)

    result = stop_relevance_test(args.voicebank, args.base)
    print(f"Base unit: {result.base_unit}")
    print("Role: internal")
    print(f"Samples: {result.samples}")
    print(f"Skipped: {result.skipped}")
    print(f"Duplicate observations removed: {result.duplicate_observations_removed}")
    print(f"Ambiguous segments removed: {result.ambiguous_segments_removed}")
    print(f"Ambiguous observations removed: {result.ambiguous_observations_removed}")
    if result.acoustic_supported_pairs:
        pairs = ", ".join(f"{left}<->{right}" for left, right in result.acoustic_supported_pairs)
    else:
        pairs = "none"
    print(f"Acoustic-supported pairs entering relevance: {pairs}")
    print("Signal proxy only; no synthesis-inventory decision is produced.")
    print()

    for item in result.comparisons:
        print(f"{item.target_family} <- {item.substitution_family}: targets={item.targets}")
        print(
            "  onset spectral: "
            f"delta={_fmt(item.mean_onset_spectral_delta)}, "
            f"p={_fmt(item.onset_spectral_p)}, "
            f"holm_p={_fmt(item.onset_spectral_p_holm)}"
        )
        print(
            "  burst spectral: "
            f"delta={_fmt(item.mean_burst_spectral_delta)}, "
            f"p={_fmt(item.burst_spectral_p)}"
        )
        print(
            "  release timing: "
            f"delta_ms={_fmt(item.mean_timing_delta_ms, 2)}, "
            f"p={_fmt(item.timing_p)}"
        )
        if item.oto_sets:
            details = ", ".join(
                f"{row.oto_set}=onset{row.mean_onset_spectral_delta:+.4f}/"
                f"burst{row.mean_burst_spectral_delta:+.4f}/"
                f"timing{row.mean_timing_delta_ms:+.2f}ms"
                for row in item.oto_sets
            )
            print(f"  oto_sets: {details}")

    if result.pairs:
        print()
        print("Bidirectional pair summaries")
        for pair in result.pairs:
            print(
                f"  {pair.left}<->{pair.right}: "
                f"both_onset_positive={pair.both_onset_positive}, "
                f"both_onset_holm_significant={pair.both_onset_holm_significant}, "
                f"all_oto_sets_onset_positive={pair.all_oto_sets_onset_positive}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
