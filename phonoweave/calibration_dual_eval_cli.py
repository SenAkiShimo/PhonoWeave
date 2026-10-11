from __future__ import annotations

import argparse
from pathlib import Path
from .calibration_dual_eval import evaluate_dual_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate separate onset and CV anchors.")
    parser.add_argument("session", type=Path)
    args = parser.parse_args(argv)
    result = evaluate_dual_session(args.session)
    print(f"Session: {result['session_id']}")
    for name, metrics in result["summary"]["comparison"].items():
        print(f"{name}: matched={metrics['matched']}/{metrics['eligible']} "
              f"MAE={metrics['mae_ms']}ms p90={metrics['p90_absolute_error_ms']}ms "
              f"median_signed={metrics['median_signed_error_ms']}ms")
    print("Statuses:", result["summary"]["prediction_statuses"])
    print("Failures:", result["summary"]["failures"])
    for row in result["rows"]:
        print(
            f"{row['base_unit']} {row['context']} #{row['occurrence']} "
            f"{row['anchor_type']}: predicted={row['predicted_ms']} "
            f"manual={row['manual_ms']:.3f} source={row['manual_source']} "
            f"error={row['error_ms']} status={row['prediction_status']}"
        )
    print("Saved: analysis/calibration_dual_anchor_evaluation_v1.4.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
