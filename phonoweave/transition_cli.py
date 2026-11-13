from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .transition_consistency import transition_consistency


def _value(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave transition-report")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--base", nargs="+", default=["m", "n", "l", "r"])
    parser.add_argument("--permutations", type=int, default=4000)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args(argv)

    reports = []
    for base in args.base:
        report = transition_consistency(args.voicebank, base, args.permutations)
        reports.append(report)
        print(f"Onset: {report.base_unit}")
        print(f"  Samples: {report.samples}, skipped: {report.skipped} {report.skip_reasons}")
        print(
            "  Within-family distance, same final vs different final: "
            f"{_value(report.same_final_distance, 2)} dB vs "
            f"{_value(report.different_final_distance, 2)} dB"
        )
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
    print("Exploratory spectral-envelope evidence only; no inventory decision is changed.")

    if args.json_path:
        payload = [asdict(report) for report in reports]
        args.json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
