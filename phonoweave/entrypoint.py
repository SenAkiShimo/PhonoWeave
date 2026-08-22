from __future__ import annotations

import argparse
import json
import sys

from .inventory import analyze_voicebank_inventory


def _inventory_payload(result) -> dict[str, object]:
    return {
        "voicebank": str(result.voicebank),
        "decisions": [
            {
                "base_unit": item.base_unit,
                "class_name": item.class_name,
                "acoustic_evidence": item.acoustic_evidence,
                "synthesis_evidence": item.synthesis_evidence,
                "decision": item.decision,
                "confidence": item.confidence,
                "notes": list(item.notes),
            }
            for item in result.decisions
        ],
    }


def _analyze_voicebank(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave analyze-voicebank")
    parser.add_argument("voicebank")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = analyze_voicebank_inventory(args.voicebank)
    payload = _inventory_payload(result)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Voicebank: {payload['voicebank']}")
    print("Speaker realization inventory")
    print()
    for item in result.decisions:
        print(f"{item.base_unit} [{item.class_name}]")
        print(f"  acoustic: {item.acoustic_evidence}")
        print(f"  synthesis: {item.synthesis_evidence}")
        print(f"  decision: {item.decision}")
        print(f"  confidence: {item.confidence}")
        for note in item.notes:
            print(f"  {note}")
        print()
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "analyze-voicebank":
        return _analyze_voicebank(sys.argv[2:])

    from .cli import main as legacy_main

    return legacy_main()


if __name__ == "__main__":
    raise SystemExit(main())
