from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration_analysis import analyze_calibration_session


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave-calibration-analysis")
    parser.add_argument("session", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = analyze_calibration_session(args.session)
    summary = result["summary"]
    print(f"session: {result['session_id']}")
    print(
        "prompts: "
        f"{summary['recorded_prompts']}/{summary['protocol_prompts']}"
    )
    print(f"observations: {summary['observations']}")
    print(f"skipped: {summary['skipped_observations']}")
    print(f"tested onsets: {summary['tested_onsets']}")
    print(
        "untested onsets: "
        + (", ".join(summary["untested_onsets"]) or "none")
    )
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
                f"alignment={pair['minimum_alignment_db']:.3f} dB "
                f"[{pair['screening_label']}]"
            )
    print("\noutputs:")
    print(args.session.expanduser().resolve() / "analysis" / "calibration_screening_v0.1.json")
    print(args.session.expanduser().resolve() / "analysis" / "calibration_observations_v0.1.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
