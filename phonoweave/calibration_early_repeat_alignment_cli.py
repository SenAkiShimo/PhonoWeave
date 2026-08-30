from __future__ import annotations

import argparse
from pathlib import Path

from .calibration_early_repeat_alignment import analyze_early_repeat_alignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze early repeat alignment for PhonoWeave calibration recordings.")
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()

    result = analyze_early_repeat_alignment(args.session_dir)
    summary = result["summary"]
    print(f"session: {result['session_id']}")
    print(f"recordings: {summary['recordings']}")
    print(f"aligned: {summary['aligned']}")
    print(f"failed: {summary['failed']}")
    print(f"median shape score: {summary['median_shape_score']}")
    print(f"median event score: {summary['median_event_score']}")
    print(f"p90 lag disagreement: {summary['p90_lag_disagreement_ms']} ms")
    print(f"median consensus lag: {summary['median_consensus_lag_ms']} ms")
    print(f"qc: {summary['qc_counts']}")
    print("\nper recording:")
    for row in result["alignments"]:
        print(
            f"  {row['prompt_index'] + 1:03d} {row['base_unit']} {row['context_family']}: "
            f"shape={row['shape_lag_ms']}ms/{row['shape_score']:.3f} "
            f"event={row['event_lag_ms']}ms/{row['event_score']:.3f} "
            f"disagree={row['lag_disagreement_ms']}ms "
            f"consensus={row['consensus_lag_ms']}ms qc={row['qc']}"
        )
    if result["failed"]:
        print("\nfailed:")
        for row in result["failed"]:
            print(f"  {row['prompt_index'] + 1:03d}: {row['reason']}")

    output = args.session_dir.expanduser().resolve() / "analysis" / "calibration_early_repeat_alignment_v0.3.2.json"
    print(f"\noutput:\n{output}")


if __name__ == "__main__":
    main()
