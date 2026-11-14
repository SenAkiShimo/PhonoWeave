from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .transition_consistency import (
    analyze_samples,
    collect_samples,
    paired_difference_reports,
)


def _value(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _print_pairs(report) -> None:
    for pair in report.pairs:
        print(
            f"  {pair.family_a} vs {pair.family_b} @ progress {pair.level:.2f}: "
            f"layers+={pair.positive_layers}/{pair.layers}, "
            f"n={pair.targets_a}/{pair.targets_b}, "
            f"stat={_value(pair.statistic, 3)}, p={_value(pair.p)}, "
            f"holm_p={_value(pair.p_holm)}"
        )
    print(f"  Result: {report.verdict}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave transition-report")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--base", nargs="+", default=["m", "n", "l", "r"])
    parser.add_argument("--permutations", type=int, default=4000)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args(argv)

    reports = []
    collected = {}
    for base in args.base:
        samples, skipped = collect_samples(args.voicebank, base)
        collected[base] = samples
        report = analyze_samples(base, samples, skipped, args.permutations)
        reports.append(report)
        print(f"Onset: {report.base_unit}")
        print(f"  Samples: {report.samples}, skipped: {report.skipped} {report.skip_reasons}")
        _print_pairs(report)

    if len(args.base) > 1:
        print("Same-final differences between onsets (shared vowel effects cancel):")
        print()
        for report in paired_difference_reports(collected, args.permutations):
            print(f"Onsets: {report.base_unit}")
            _print_pairs(report)
            reports.append(report)

    print("Exploratory spectral-envelope evidence only; no inventory decision is changed.")

    if args.json_path:
        payload = [asdict(report) for report in reports]
        args.json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
