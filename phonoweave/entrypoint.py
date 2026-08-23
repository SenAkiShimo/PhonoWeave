from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .context_audit import audit_contexts
from .coverage import analyze_coverage
from .inventory import analyze_voicebank_inventory
from .lateral import analyze_lateral
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
    for base in result.bases:
        roles = ", ".join(f"{name}={count}" for name, count in base.roles.items())
        print(
            f"{base.base_unit}: observations={base.observations}, "
            f"unique_wavs={base.unique_wavs}, unique_segments={base.unique_segments}, "
            f"roles={roles}"
        )
        for final in base.finals:
            final_roles = ", ".join(f"{name}={count}" for name, count in final.roles.items())
            layers = ",".join(final.subbanks)
            print(
                f"  {final.final}: observations={final.observations}, "
                f"unique_segments={final.unique_segments}, roles={final_roles}, "
                f"subbanks={layers}"
            )
    return 0


def _analyze_lateral(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave analyze-lateral")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = analyze_lateral(args.voicebank)
    payload = {
        "samples": result.samples,
        "skipped": result.skipped,
        "duplicate_segments": result.duplicate_segments,
        "roles": [
            {
                "role": role.role,
                "counts": role.counts,
                "cross_subbank_balanced_accuracy": role.cross_subbank_balanced_accuracy,
                "cross_by_subbank": role.cross_by_subbank,
                "pairwise": [
                    {
                        "left": pair.left,
                        "right": pair.right,
                        "cross_subbank_balanced_accuracy": pair.cross_subbank_balanced_accuracy,
                        "cross_by_subbank": pair.cross_by_subbank,
                        "mean_distance": pair.mean_distance,
                        "stratified_distance": pair.stratified_distance,
                        "stratified_permutation_p": pair.stratified_permutation_p,
                        "stratified_p_holm": pair.stratified_p_holm,
                        "stratified_effects": pair.stratified_effects,
                        "effect_sign_agreement": pair.effect_sign_agreement,
                        "stratified_subbanks": pair.stratified_subbanks,
                        "subbanks": [
                            {
                                "subbank": item.subbank,
                                "left_count": item.left_count,
                                "right_count": item.right_count,
                                "distance": item.distance,
                                "loo_balanced_accuracy": item.loo_balanced_accuracy,
                                "permutation_p": item.permutation_p,
                            }
                            for item in pair.subbanks
                        ],
                    }
                    for pair in role.pairwise
                ],
            }
            for role in result.roles
        ],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("Base unit: l")
    print(f"Samples: {result.samples}")
    print(f"Skipped: {result.skipped}")
    print(f"Duplicate segments removed: {result.duplicate_segments}")
    print("Experimental acoustic evidence only; no inventory decision is produced.")
    print()
    for role in result.roles:
        counts = ", ".join(f"{name}={count}" for name, count in role.counts.items())
        print(f"{role.role}: {counts}")
        if role.cross_subbank_balanced_accuracy is not None:
            print(f"  cross-subbank balanced accuracy: {role.cross_subbank_balanced_accuracy:.3f}")
            held = ", ".join(f"{name}={score:.3f}" for name, score in role.cross_by_subbank.items())
            print(f"  held out: {held}")
        else:
            print("  cross-subbank balanced accuracy: n/a")
        print("  pairwise:")
        for pair in role.pairwise:
            cross = (
                f"{pair.cross_subbank_balanced_accuracy:.3f}"
                if pair.cross_subbank_balanced_accuracy is not None
                else "n/a"
            )
            distance = f"{pair.mean_distance:.3f}" if pair.mean_distance is not None else "n/a"
            print(
                f"    {pair.left} vs {pair.right}: "
                f"cross_ba={cross}, mean_distance={distance}"
            )
            if pair.cross_by_subbank:
                held = ", ".join(
                    f"{name}={score:.3f}" for name, score in pair.cross_by_subbank.items()
                )
                print(f"      held out: {held}")
            if pair.stratified_distance is not None:
                raw_p = pair.stratified_permutation_p
                holm_p = pair.stratified_p_holm
                raw_text = f"{raw_p:.4f}" if raw_p is not None else "n/a"
                holm_text = f"{holm_p:.4f}" if holm_p is not None else "n/a"
                print(
                    f"      stratified: distance={pair.stratified_distance:.3f}, "
                    f"p={raw_text}, holm_p={holm_text}, "
                    f"layers={pair.stratified_subbanks}"
                )
                ranked = sorted(
                    pair.stratified_effects.items(),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )[:4]
                if ranked:
                    effects = ", ".join(
                        f"{name}={value:+.3f} "
                        f"({pair.effect_sign_agreement.get(name, 0)}/{pair.stratified_subbanks})"
                        for name, value in ranked
                    )
                    print(f"      effects: {effects}")
            for item in pair.subbanks:
                print(
                    f"      {item.subbank}: n={item.left_count}/{item.right_count}, "
                    f"distance={item.distance:.3f}, "
                    f"loo_ba={item.loo_balanced_accuracy:.3f}, "
                    f"p={item.permutation_p:.4f}"
                )
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
    if len(sys.argv) > 1 and sys.argv[1] == "analyze-lateral":
        return _analyze_lateral(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "build-profile":
        return _build_profile(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "build-synthesis-inventory":
        return _build_synthesis_inventory(sys.argv[2:])

    from .cli import main as legacy_main

    return legacy_main()


if __name__ == "__main__":
    raise SystemExit(main())
