from __future__ import annotations

import argparse
from pathlib import Path
from .calibration_manual_window_audit import audit_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave-manual-window-audit")
    parser.add_argument("session", type=Path)
    args = parser.parse_args(argv)
    r = audit_session(args.session)
    print(f"Session: {r['session_id']}")
    print(f"Summary: {r['summary']}")
    for row in r["rows"]:
        a = row["audit"]
        gold = a["gold"]
        print(
            f"{row['base_unit']} {row['context']} #{row['occurrence']}: "
            f"label={row['label_status']} "
            f"in_search={gold.get('inside_automatic_search', 'n/a')} "
            f"gate_counts_near_gold={gold.get('local_gate_counts', 'n/a')} "
            f"periodicity_near_gold={gold.get('periodicity_near_gold_median')} "
            f"no_rms_candidate={a['no_rms_candidate_ms_diagnostic_only']} "
            f"no_rms_error={gold.get('no_rms_candidate_error_ms_diagnostic_only')} "
            f"signal={row['at_manual_anchor']}"
        )
    for row in r["failures"]:
        print(f"FAILURE {row}")
    print("Saved: analysis/calibration_manual_window_audit_v1.1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
