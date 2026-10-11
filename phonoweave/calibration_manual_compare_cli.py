from __future__ import annotations

import argparse
from pathlib import Path

from .calibration_manual_compare import evaluate_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave-manual-compare")
    parser.add_argument("session", type=Path)
    args = parser.parse_args(argv)
    result = evaluate_session(args.session)
    print(f"Session: {result['session_id']}")
    print(f"Manual labels: {result['manual_annotations']}")
    for name, block in result["models"].items():
        metric = block["summary"]
        print(f"{name}: matched={metric['matched']}/{metric['eligible']} "
              f"MAE={metric['mae_ms']} ms, p90={metric['p90_absolute_error_ms']} ms")
        print(f"  statuses={block['status_counts']}")
    print(f"Sibilant unresolved reasons: {result['sibilant_unresolved_reasons']}")
    print("Saved: analysis/calibration_manual_comparison_v0.8.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
