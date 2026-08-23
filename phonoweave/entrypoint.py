from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .context_audit import audit_contexts
from .coverage import analyze_coverage
from .inventory import analyze_voicebank_inventory
from .profile import write_speaker_profile
from .source_audit import audit_sources
from .synthesis_inventory import write_synthesis_inventory


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


def _coverage_payload(result) -> dict[str, object]:
    return {
        "voicebank": str(result.voicebank),
        "observations": result.observations,
        "onset_observations": result.onset_observations,
        "zero_onset_observations": result.zero_onset_observations,
        "items": [
            {
                "base_unit": item.base_unit,
                "observations": item.observations,
                "status": item.status,
                "analyzer": item.analyzer,
            }
            for item in result.items
        ],
    }


def _context_audit_payload(result) -> dict[str, object]:
    return {
        "voicebank": str(result.voicebank),
        "zero_onset_observations": result.zero_onset_observations,
        "items": [
            {
                "base_unit": item.base_unit,
                "status": item.status,
                "observations": item.observations,
                "finals": [
                    {
                        "final": final.final,
                        "observations": final.observations,
                        "subbanks": list(final.subbanks),
                    }
                    for final in item.finals
                ],
            }
            for item in result.items
        ],
    }


def _analyze_voicebank(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave analyze-voicebank")
    parser.add_argument("voicebank", type=Path)
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

    coverage = analyze_coverage(args.voicebank)
    unsupported = [item for item in coverage.items if item.status == "unsupported"]
    analyzed = [item for item in coverage.items if item.status == "analyzed"]
    print("Coverage")
    print(f"  Mandarin observations recognized: {coverage.observations}")
    print(f"  onset observations: {coverage.onset_observations}")
    print(f"  zero-onset orthographic observations: {coverage.zero_onset_observations}")
    print(f"  onset types analyzed: {len(analyzed)}")
    print(f"  onset types not yet analyzed: {len(unsupported)}")
    if unsupported:
        details = ", ".join(
            f"{item.base_unit}({item.observations})"
            for item in unsupported
        )
        print(f"  unsupported: {details}")
    return 0


def _coverage(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave coverage")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = analyze_coverage(args.voicebank)
    payload = _coverage_payload(result)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Voicebank: {payload['voicebank']}")
    print(f"Mandarin observations recognized: {result.observations}")
    print(f"Onset observations: {result.onset_observations}")
    print(f"Zero-onset orthographic observations: {result.zero_onset_observations}")
    print()
    for item in result.items:
        analyzer = item.analyzer or "not implemented"
        print(
            f"{item.base_unit}: observations={item.observations}, "
            f"status={item.status}, analyzer={analyzer}"
        )
    return 0


def _context_audit(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave context-audit")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--base")
    parser.add_argument("--unsupported-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = audit_contexts(
        args.voicebank,
        base_unit=args.base,
        unsupported_only=args.unsupported_only,
    )
    payload = _context_audit_payload(result)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Voicebank: {payload['voicebank']}")
    print(f"Zero-onset orthographic observations: {result.zero_onset_observations}")
    print()
    for item in result.items:
        print(
            f"{item.base_unit}: observations={item.observations}, "
            f"status={item.status}"
        )
        for final in item.finals:
            layers = ",".join(final.subbanks)
            print(
                f"  {final.final}: observations={final.observations}, "
                f"subbanks={layers}"
            )
    return 0


def _source_audit(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave source-audit")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--base", action="append", dest="bases")
    args = parser.parse_args(argv)

    bases = tuple(args.bases or ("m", "n", "l"))
    result = audit_sources(args.voicebank, bases)
    print(f"Voicebank: {result.voicebank}")
    print()
    for item in result.bases:
        roles = ", ".join(f"{key}={value}" for key, value in item.roles.items()) or "-"
        print(
            f"{item.base_unit}: observations={item.observations}, "
            f"unique_wavs={item.unique_wavs}, "
            f"unique_segments={item.unique_segments}, roles={roles}"
        )
        for final in item.finals:
            final_roles = ", ".join(
                f"{key}={value}" for key, value in final.roles.items()
            ) or "-"
            layers = ",".join(final.subbanks)
            print(
                f"  {final.final}: observations={final.observations}, "
                f"unique_segments={final.unique_segments}, "
                f"roles={final_roles}, subbanks={layers}"
            )
        print()
    return 0


def _build_profile(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave build-profile")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("speaker_profile.yaml"),
    )
    args = parser.parse_args(argv)

    profile = write_speaker_profile(args.voicebank, args.output)
    output = args.output.expanduser().resolve()
    print(f"Speaker: {profile.speaker_id}")
    print(f"Language: {profile.language}")
    print(f"Realizations: {len(profile.realizations)}")
    print(f"Output: {output}")
    return 0


def _build_synthesis_inventory(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave build-synthesis-inventory")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("synthesis_inventory.yaml"),
    )
    args = parser.parse_args(argv)

    inventory = write_synthesis_inventory(args.voicebank, args.output)
    output = args.output.expanduser().resolve()
    print(f"Speaker: {inventory.speaker_id}")
    print(f"Language: {inventory.language}")
    print(f"Synthesis units: {len(inventory.units)}")
    print(f"Output: {output}")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "analyze-voicebank":
        return _analyze_voicebank(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "coverage":
        return _coverage(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "context-audit":
        return _context_audit(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "source-audit":
        return _source_audit(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "build-profile":
        return _build_profile(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "build-synthesis-inventory":
        return _build_synthesis_inventory(sys.argv[2:])

    from .cli import main as legacy_main

    return legacy_main()


if __name__ == "__main__":
    raise SystemExit(main())
