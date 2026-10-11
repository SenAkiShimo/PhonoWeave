from __future__ import annotations

import argparse
from pathlib import Path

from .calibration_periodicity_repair import evaluate_periodicity_repair


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave-periodicity-repair")
    parser.add_argument("session", type=Path)
    args = parser.parse_args(argv)
    r = evaluate_periodicity_repair(args.session)
    print(f"Session: {r['session_id']}")
    print(f"Summary: {r['summary']}")
    for row in r["rows"]:
        d = row["repaired"]
        print(
            f"{row['base_unit']} {row['context']} #{row['occurrence']}: "
            f"periodicity median {row['periodicity_p50_old']} -> {row['periodicity_p50_repaired']}; "
            f"strict={d['strict_longest_run']} frames; "
            f"without-RMS={d['no_rms_longest_run_diagnostic_only']} frames; "
            f"estimate={d['onset_ms_after_cue']}; error={row['signed_error_ms']}"
        )
    print("Saved: analysis/calibration_periodicity_repair_v1.0.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
