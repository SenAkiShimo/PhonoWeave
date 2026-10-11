from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration_sibilant_onset import analyze_sibilant_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave-calibration-sibilant-onset")
    parser.add_argument("session", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = analyze_sibilant_session(args.session)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"session: {result['session_id']}")
        for name, value in result["summary"].items():
            print(f"{name}: {value}")
        print("output: analysis/calibration_sibilant_onset_v0.6.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
