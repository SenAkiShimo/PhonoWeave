from __future__ import annotations

import argparse
from pathlib import Path

from .calibration_target_alignment import analyze_target_alignment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave-calibration-target-alignment")
    parser.add_argument("session", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = analyze_target_alignment(args.session)
    summary = payload["summary"]
    print(f"session: {payload['session_id']}")
    print(f"recordings: {summary['recordings']}")
    print(f"aligned: {summary['aligned']}")
    print(f"failed: {summary['failed']}")
    print(f"observations: {summary['observations']}")
    print(f"median cue to anchor: {summary['median_cue_to_anchor_ms']} ms")
    print(f"p90 repeat spread: {summary['p90_repeat_spread_ms']} ms")
    print(f"max repeat spread: {summary['max_repeat_spread_ms']} ms")
    print(f"qc: {summary['qc_counts']}")
    print("\nper recording:")
    for row in payload["alignments"]:
        anchors = ", ".join(
            f"#{item['occurrence']}={item['cue_to_anchor_ms']}ms/{item['anchor_type']}"
            for item in row["anchors"]
        )
        print(
            f"  {row['prompt_index'] + 1:03d} {row['base_unit']} {row['context_family']}: "
            f"spread={row['repeat_spread_ms']}ms qc={row['qc']} anchors=[{anchors}]"
        )
    if payload["failed"]:
        print("\nfailed:")
        for item in payload["failed"]:
            print(f"  {item['prompt_index'] + 1:03d}: {item['reason']}")
    output = Path(payload["session_dir"]) / "analysis" / "calibration_target_alignment_v0.3.json"
    print(f"\noutput:\n{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
