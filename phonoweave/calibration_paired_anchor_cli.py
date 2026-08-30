from __future__ import annotations

import argparse
from pathlib import Path

from .calibration_paired_anchor import analyze_paired_anchors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()

    payload = analyze_paired_anchors(args.session_dir)
    summary = payload["summary"]
    print(f"session: {payload['session_id']}")
    print(f"recordings: {summary['recordings']}")
    print(f"anchored: {summary['anchored']}")
    print(f"failed: {summary['failed']}")
    print(f"median repeat anchor spread: {summary['median_repeat_anchor_spread_ms']} ms")
    print(f"p90 repeat anchor spread: {summary['p90_repeat_anchor_spread_ms']} ms")
    print(f"median shared anchor: {summary['median_shared_anchor_ms_after_cue']} ms after cue")
    print(f"median usable anchor: {summary['median_usable_anchor_ms_after_cue']} ms after cue")
    print(f"qc: {summary['qc_counts']}")
    print("\nper recording:")
    for row in payload["anchors"]:
        print(
            f"  {row['prompt_index'] + 1:03d} {row['base_unit']} {row['context_family']}: "
            f"lag={row['consensus_lag_ms']}ms align={row['alignment_qc']} "
            f"shared={row['shared_anchor_ms_after_cue']}ms "
            f"repeats=[{row['occurrence_1_anchor_ms_after_cue']}ms, "
            f"{row['occurrence_2_anchor_ms_after_cue']}ms] "
            f"spread={row['repeat_anchor_spread_ms']}ms "
            f"support={row['event_support']} qc={row['qc']}"
        )
    print("\noutput:")
    print(Path(payload["session_dir"]) / "analysis" / "calibration_paired_anchor_v0.3.3.json")


if __name__ == "__main__":
    main()
