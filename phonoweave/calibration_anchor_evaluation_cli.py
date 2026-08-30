from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration_anchor_evaluation import evaluate_manual_anchors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave-calibration-anchor-eval")
    parser.add_argument("session", type=Path)
    parser.add_argument("--json", action="store_true", help="print full JSON payload")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = evaluate_manual_anchors(args.session)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    manual = payload["manual_labels"]
    absolute = payload["absolute_anchor"]["primary_metrics"]
    relative = payload["relative_repeat_alignment"]["primary_metrics"]

    print(f"session: {payload['session_id']}")
    print(
        "manual labels: "
        f"{manual['total']} total, {manual['ok']} primary, {manual['uncertain']} uncertain"
    )
    print("absolute paired anchor vs manual:")
    print(f"  n: {absolute['n']}")
    print(f"  MAE: {absolute['mae_ms']} ms")
    print(f"  median abs error: {absolute['median_absolute_error_ms']} ms")
    print(f"  p90 abs error: {absolute['p90_absolute_error_ms']} ms")
    print(f"  median signed error: {absolute['median_signed_error_ms']} ms")
    print("relative repeat alignment vs manual repeat lag:")
    print(f"  n: {relative['n']}")
    print(f"  MAE: {relative['mae_ms']} ms")
    print(f"  median abs error: {relative['median_absolute_error_ms']} ms")
    print(f"  p90 abs error: {relative['p90_absolute_error_ms']} ms")
    print(f"  median signed error: {relative['median_signed_error_ms']} ms")
    print("output: analysis/calibration_anchor_evaluation_v0.4.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
