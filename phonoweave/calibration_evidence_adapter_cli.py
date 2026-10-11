"""Read-only conversion from calibration results to timing evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

from .calibration_evidence_adapter import adapt_calibration_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate and inspect calibration evidence")
    parser.add_argument("session", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = adapt_calibration_evidence(
        args.session, prediction_path=args.predictions, output=args.output
    )
    print("Session:", report["session_id"])
    print("Schema:", report["schema"], report["schema_version"])
    print("Summary:", report["summary"])
    print("Identity binding:", report["identity_binding"])
    print("Speaker profile decisions modified:", report["profile_decisions_modified"])
    print("Synthesis inventory modified:", report["synthesis_inventory_modified"])


if __name__ == "__main__":
    main()
