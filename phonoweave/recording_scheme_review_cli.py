"""Compile a read-only recording-scheme review pack from an existing voicebank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .recording_scheme_review import build_from_voicebank, write_review_package


def voicebank_from_manifest(manifest: Path, bank_id: str) -> Path:
    data = json.loads(manifest.expanduser().resolve(strict=True).read_text("utf-8"))
    if data.get("schema") != "phonoweave.voicebank_replication_manifest.v1":
        raise ValueError("unsupported voicebank manifest schema")
    entries = data.get("voicebanks")
    if not isinstance(entries, list):
        raise ValueError("manifest voicebanks must be an array")
    matches = [bank for bank in entries
               if isinstance(bank, dict) and bank.get("bank_id") == bank_id]
    if len(matches) != 1:
        raise ValueError("bank ID missing or duplicated in manifest")
    value = matches[0].get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("selected bank has no path")
    root = Path(value).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("selected bank path is not a directory")
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile synthesis inventory, alias/OTO review and supplemental reclist"
    )
    parser.add_argument("voicebank", nargs="?", type=Path,
                        help="voicebank directory, or use --manifest with --bank-id")
    parser.add_argument("--manifest", type=Path,
                        help="existing local voicebank replication manifest")
    parser.add_argument("--bank-id",
                        help="one bank ID from the existing local manifest")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="new output directory; never overwritten")
    args = parser.parse_args(argv)
    if args.voicebank is not None and (args.manifest or args.bank_id):
        parser.error("choose positional voicebank OR --manifest with --bank-id")
    if args.voicebank is None and not (args.manifest and args.bank_id):
        parser.error("provide a voicebank path OR both --manifest and --bank-id")
    if args.output.expanduser().exists():
        parser.error("output directory already exists; choose a new path")
    try:
        root = (args.voicebank if args.voicebank is not None
                else voicebank_from_manifest(args.manifest, args.bank_id))
        if not root.expanduser().is_dir():
            raise ValueError("voicebank directory does not exist")
        package = build_from_voicebank(root)
        location = write_review_package(package, args.output)
    except (FileExistsError, FileNotFoundError, ValueError,
            json.JSONDecodeError) as exc:
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
