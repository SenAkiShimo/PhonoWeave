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
        "valid_entries": summary.valid_entries,
        "missing_wavs": summary.missing_wavs,
        "observations": summary.observations,
        "groups": summary.groups,
    }


def _prefix_rule_payload(rule) -> dict[str, object]:
    return {
        "color": rule.color,
        "prefix": rule.prefix,
        "suffix": rule.suffix,
        "tones": list(rule.tones),
        "tone_ranges": list(rule.tone_ranges),
    }


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "inspect":
        result = inspect_voicebank(args.voicebank)
        payload = {
            "voicebank": str(result.root),
            "oto_files": result.oto_files,
            "entries": result.entries,
            "valid_entries": result.valid_entries,
            "missing_wavs": result.missing_wavs,
            "parse_warnings": result.parse_warnings,
            "observations": result.observations,
            "groups": result.groups,
            "subbanks": [_subbank_payload(summary) for summary in result.subbanks],
            "prefix_rules": [_prefix_rule_payload(rule) for rule in result.prefix_rules],
        }

        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"Voicebank: {payload['voicebank']}")
            print(f"OTO files: {payload['oto_files']}")
            print(f"Entries: {payload['entries']}")
            print(f"Valid entries: {payload['valid_entries']}")
            print(f"Missing WAVs: {payload['missing_wavs']}")
            print(f"Parse warnings: {payload['parse_warnings']}")
            print(f"Mandarin observations: {payload['observations']}")
            for base, contexts in payload["groups"].items():
                summary = ", ".join(f"{name}={count}" for name, count in contexts.items())
                print(f"{base}: {summary}")

            if payload["prefix_rules"]:
                print()
                print("Prefix map:")
                for rule in payload["prefix_rules"]:
                    ranges = ", ".join(rule["tone_ranges"]) or "(unresolved)"
                    prefix = rule["prefix"] or '""'
                    suffix = rule["suffix"] or '""'
                    color = rule["color"] or "default"
                    print(
                        f"  {color}: prefix={prefix!r}, suffix={suffix!r}, tones={ranges}"
                    )

            if payload["subbanks"]:
                print()
                print("OTO sets:")
                for subbank in payload["subbanks"]:
                    print(
                        f"  {subbank['name']}: entries={subbank['entries']}, "
                        f"valid={subbank['valid_entries']}, missing_wavs={subbank['missing_wavs']}, "
                        f"observations={subbank['observations']}"
                    )
                    for base, contexts in subbank["groups"].items():
                        summary = ", ".join(f"{name}={count}" for name, count in contexts.items())
                        print(f"    {base}: {summary}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
