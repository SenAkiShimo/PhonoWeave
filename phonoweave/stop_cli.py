from __future__ import annotations

import argparse
from pathlib import Path

from .stop import analyze_stop_diagnostic


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave analyze-stop")
    parser.add_argument("voicebank", type=Path)
    args = parser.parse_args(argv)

    result = analyze_stop_diagnostic(args.voicebank)
    print(f"Voicebank: {result.voicebank}")
    print(f"Candidates: {result.candidates}")
    print(f"Samples: {result.samples}")
    print(f"Skipped: {result.skipped}")
    print(f"Duplicate observations removed: {result.duplicate_observations_removed}")
    print(f"Ambiguous segments removed: {result.ambiguous_segments_removed}")
    print(f"Ambiguous observations removed: {result.ambiguous_observations_removed}")
    print("Diagnostic acoustic evidence only; no synthesis-inventory decision is produced.")
    print()

    print("Base summaries")
    for item in result.bases:
        print(
            f"  {item.base_unit}: n={item.samples}, "
            f"release_to_vowel_ms={_fmt(item.median_release_to_vowel_ms)}, "
            f"release_strength={_fmt(item.median_release_strength, 3)}, "
            f"vowel_periodicity={_fmt(item.median_vowel_periodicity, 3)}, "
            f"burst_centroid_hz={_fmt(item.median_burst_centroid_hz, 1)}, "
            f"burst_high_ratio={_fmt(item.median_burst_high_ratio, 3)}"
        )

    print()
    print("Matched aspiration contrasts")
    print("  delta = aspirated minus unaspirated release-to-vowel duration")
    for item in result.contrasts:
        held = ", ".join(
            f"{name}={value:+.2f}"
            for name, value in item.oto_set_median_deltas_ms.items()
        ) or "n/a"
        print(
            f"  {item.place} {item.role}: matched_cells={item.matched_cells}, "
            f"median_delta_ms={_fmt(item.median_aspiration_delta_ms)}, "
            f"positive_cells={item.positive_cells}/{item.matched_cells}, "
            f"oto_sets={held}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
