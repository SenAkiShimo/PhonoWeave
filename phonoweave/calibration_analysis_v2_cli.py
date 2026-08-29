from __future__ import annotations

import argparse
from pathlib import Path

from .calibration_analysis_v2 import analyze_calibration_session_v2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave-calibration-analysis-v2")
    parser.add_argument("session", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = analyze_calibration_session_v2(args.session)
    summary = result["summary"]
    print(f"session: {result['session_id']}")
    print(f"prompts: {summary['recorded_prompts']}/{summary['protocol_prompts']}")
    print(f"observations: {summary['observations']}")
    print(f"skipped: {summary['skipped_observations']}")
    print(f"tested onsets: {summary['tested_onsets']}")
    print("untested onsets: " + (", ".join(summary["untested_onsets"]) or "none"))
    print(f"median detected offset: {summary['median_detected_offset_ms']} ms")
    print(f"p90 timing repeat spread: {summary['p90_timing_repeat_spread_ms']} ms")
    print(f"max timing repeat spread: {summary['maximum_timing_repeat_spread_ms']} ms")
    print("\npairwise screening:")
    for base in result["bases"]:
        pairs = base["pairwise"]
        if not pairs:
            print(f"  {base['base_unit']}: no context contrast")
            continue
        for pair in pairs:
            print(
                f"  {base['base_unit']} "
                f"{pair['context_a']} vs {pair['context_b']}: "
                f"distance={pair['standardized_distance']:.4f} "
                f"repeat={pair['within_context_repeat_distance']} "
                f"ratio={pair['separation_to_repeat_ratio']} "
                f"timing_spread={pair['maximum_timing_repeat_spread_ms']} ms "
                f"[{pair['screening_label']}]"
            )
    print("\noutputs:")
    root = args.session.expanduser().resolve() / "analysis"
    print(root / "calibration_screening_v0.2.json")
    print(root / "calibration_observations_v0.2.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
