from __future__ import annotations

import argparse
from pathlib import Path

from .calibration_beep_alignment import analyze_beep_alignment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave-calibration-beep-alignment")
    parser.add_argument("session", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = analyze_beep_alignment(args.session)
    summary = result["summary"]
    print(f"session: {result['session_id']}")
    print(f"recordings: {summary['recordings']}")
    print(f"aligned: {summary['aligned']}")
    print(f"failed: {summary['failed']}")
    print(f"median actual interval: {summary['median_actual_interval_ms']} ms")
    print(f"p90 max interval error: {summary['p90_max_interval_error_ms']} ms")
    print(f"minimum frequency margin: {summary['minimum_frequency_margin']}")
    print(f"median sequence score: {summary['median_sequence_score']}")
    print("\nper recording:")
    for item in result["alignments"]:
        target_times = ", ".join(
            f"#{cue['occurrence']}={cue['cue_ms']:.1f}ms"
            for cue in item["target_cues"]
        )
        print(
            f"  {item['prompt_index'] + 1:03d} {item['base_unit']} {item['context_family']}: "
            f"beeps={item['detected_beeps']}/{item['expected_beeps']} "
            f"interval={item['median_interval_ms']}ms "
            f"maxerr={item['max_interval_error_ms']}ms "
            f"score={item['sequence_score']:.3f} "
            f"targets=[{target_times}]"
        )
    if result["failed"]:
        print("\nfailed:")
        for item in result["failed"]:
            print(f"  {item['prompt_index'] + 1:03d}: {item['reason']}")
    print("\noutput:")
    print(args.session.expanduser().resolve() / "analysis" / "calibration_beep_alignment_v0.3.json")
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
