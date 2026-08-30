from __future__ import annotations

import sys
from pathlib import Path

from .calibration_repeat_alignment import analyze_repeat_alignment


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m phonoweave.calibration_repeat_alignment_cli SESSION_DIR")

    payload = analyze_repeat_alignment(Path(sys.argv[1]))
    summary = payload["summary"]
    print(f"session: {payload['session_id']}")
    print(f"recordings: {summary['recordings']}")
    print(f"aligned: {summary['aligned']}")
    print(f"failed: {summary['failed']}")
    print(f"median spectral score: {summary['median_spectral_score']}")
    print(f"median activity score: {summary['median_activity_score']}")
    print(f"p90 lag disagreement: {summary['p90_lag_disagreement_ms']} ms")
    print(f"median pair lag: {summary['median_pair_lag_ms']} ms")
    print(f"qc: {summary['qc_counts']}")
    print("\nper recording:")
    for row in payload["alignments"]:
        print(
            f"  {row['prompt_index'] + 1:03d} {row['base_unit']} {row['context_family']}: "
            f"spectral={row['spectral_lag_ms']}ms/{row['spectral_score']:.3f} "
            f"activity={row['activity_lag_ms']}ms/{row['activity_score']:.3f} "
            f"disagree={row['lag_disagreement_ms']}ms qc={row['qc']}"
        )

    output = Path(payload["session_dir"]) / "analysis" / "calibration_repeat_alignment_v0.3.1.json"
    print("\noutput:")
    print(output)


if __name__ == "__main__":
    main()
