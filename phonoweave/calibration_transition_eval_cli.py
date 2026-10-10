from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_transition_eval import analyze_transition_session

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    args = parser.parse_args(argv)
    result = analyze_transition_session(args.session)
    print("Session:", result["session_id"])
    print("Summary:", result["summary"])
    for r in result["rows"]:
        print(f"{r['base_unit']} {r['context']} #{r['occurrence']}: "
              f"nominal={r['nominal_cue_ms']}ms onset={r['detected_onset_absolute_ms']}ms "
              f"status={r['status']} duration={r['frication_duration_ms']}ms "
              f"manual_delta={r['manual_minus_candidate_ms']}ms "
              f"review_manual={r['manual_requires_review']}")
    print("Saved: analysis/calibration_transition_onset_v1.2.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
