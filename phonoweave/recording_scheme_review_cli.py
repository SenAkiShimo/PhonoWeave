"""Compile a read-only recording-scheme review pack from an existing voicebank."""
from __future__ import annotations

import argparse
from pathlib import Path

from .recording_scheme_review import build_from_voicebank, write_review_package


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile synthesis inventory, alias/OTO review and supplemental reclist"
    )
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="new output directory; never overwritten")
    args = parser.parse_args(argv)
    if not args.voicebank.expanduser().is_dir():
        parser.error("voicebank directory does not exist")
    if args.output.expanduser().exists():
        parser.error("output directory already exists; choose a new path")
    try:
        package = build_from_voicebank(args.voicebank)
        location = write_review_package(package, args.output)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print("Recording-scheme review pack:", location)
    print("Speaker:", package.summary["speaker_id"])
    print("Neutral units:", package.summary["units"])
    print("Observed OTO aliases:", package.summary["oto_entries"])
    print("Suggested diagnostic lines:", package.summary["suggested_diagnostic_lines"])
    print("Status: manual review required; voicebank is unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
