from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration_anchor_failure_analysis import analyze_anchor_failures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave-calibration-anchor-failure-analysis")
    parser.add_argument("session", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = analyze_anchor_failures(args.session)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    absolute = payload["headline"]["absolute_anchor"]
    relative = payload["headline"]["relative_repeat_alignment"]
    print(f"session: {payload['session_id']}")
    print(
        "absolute anchor: "
        f"{absolute['decision']} · MAE {absolute['mae_ms']} ms · p90 {absolute['p90_absolute_error_ms']} ms"
    )
    print(
        "relative repeat alignment: "
        f"{relative['decision']} · MAE {relative['mae_ms']} ms · p90 {relative['p90_absolute_error_ms']} ms"
    )
    print("class ranking by absolute MAE:")
    for row in payload["class_ranking_by_absolute_mae"]:
        print(
            f"  {row['class_name']}: MAE {row['mae_ms']} ms, "
            f"median signed {row['median_signed_error_ms']} ms, n={row['n']}"
        )
    print("next detector recommendations:")
    for row in payload["class_recommendations"]:
        print(f"  {row['class_name']}: {row['recommended_detector']} · {row['dominant_failure']}")
    print("output: analysis/calibration_anchor_failure_analysis_v0.5.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
