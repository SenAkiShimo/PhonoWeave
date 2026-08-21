from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inspect import inspect_voicebank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("voicebank", type=Path)
    inspect_parser.add_argument("--json", action="store_true")
    return parser


def _subbank_payload(summary) -> dict[str, object]:
    return {
        "name": summary.name,
        "path": str(summary.path),
        "oto_files": summary.oto_files,
        "entries": summary.entries,
        "missing_wavs": summary.missing_wavs,
        "observations": summary.observations,
        "groups": summary.groups,
    }


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "inspect":
        result = inspect_voicebank(args.voicebank)
        payload = {
            "voicebank": str(result.root),
            "oto_files": result.oto_files,
            "entries": result.entries,
            "missing_wavs": result.missing_wavs,
            "parse_warnings": result.parse_warnings,
            "observations": result.observations,
            "groups": result.groups,
            "subbanks": [_subbank_payload(summary) for summary in result.subbanks],
        }

        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"Voicebank: {payload['voicebank']}")
            print(f"OTO files: {payload['oto_files']}")
            print(f"Entries: {payload['entries']}")
            print(f"Missing WAVs: {payload['missing_wavs']}")
            print(f"Parse warnings: {payload['parse_warnings']}")
            print(f"Mandarin observations: {payload['observations']}")
            for base, contexts in payload["groups"].items():
                summary = ", ".join(f"{name}={count}" for name, count in contexts.items())
                print(f"{base}: {summary}")

            if payload["subbanks"]:
                print()
                print("Subbanks:")
                for subbank in payload["subbanks"]:
                    print(
                        f"  {subbank['name']}: entries={subbank['entries']}, "
                        f"missing_wavs={subbank['missing_wavs']}, observations={subbank['observations']}"
                    )
                    for base, contexts in subbank["groups"].items():
                        summary = ", ".join(f"{name}={count}" for name, count in contexts.items())
                        print(f"    {base}: {summary}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
