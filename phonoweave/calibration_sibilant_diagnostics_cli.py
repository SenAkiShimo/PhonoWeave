from __future__ import annotations

import argparse
from pathlib import Path

from .calibration_sibilant_diagnostics import GATE_NAMES, diagnose_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave-sibilant-diagnostics")
    parser.add_argument("session", type=Path)
    args = parser.parse_args(argv)
    result = diagnose_session(args.session)
    print(f"Session: {result['session_id']}, {result['summary']}")
    for row in result["rows"]:
        d = row["gate_diagnostics"]
        if "error" in d:
            print(f"{row['base_unit']} {row['context']} #{row['occurrence']}: {d['error']}")
            continue
        counts = "/".join(str(d["gate_pass_counts"][name]) for name in GATE_NAMES)
        print(
            f"{row['base_unit']} {row['context']} #{row['occurrence']}: "
            f"gates={counts}, joint={d['all_five_pass_frames']}, "
            f"longest_run={d['longest_all_five_run_frames']}, "
            f"best={d['best_gate_count']}/5 "
            f"missing={','.join(d['missing_gates_at_best_frame']) or '-'}"
        )
    print("Saved: analysis/calibration_sibilant_gate_diagnostics_v0.6.1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
