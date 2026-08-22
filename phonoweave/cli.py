from __future__ import annotations

import argparse
import json
from pathlib import Path

from .affricate import analyze_affricate_contrast
from .analyze import analyze_fricative_contrast
from .inspect import inspect_voicebank
from .render_ab import render_ab_pairs
from .rhotic import analyze_rhotic_contrast
from .splice import splice_relevance_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("voicebank", type=Path)
    inspect_parser.add_argument("--json", action="store_true")

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("voicebank", type=Path)
    analyze_parser.add_argument("--base", default="sh", choices=("sh",))
    analyze_parser.add_argument("--json", action="store_true")

    affricate_parser = subparsers.add_parser("analyze-affricate")
    affricate_parser.add_argument("voicebank", type=Path)
    affricate_parser.add_argument("--base", required=True, choices=("zh", "ch"))
    affricate_parser.add_argument("--json", action="store_true")

    rhotic_parser = subparsers.add_parser("analyze-rhotic")
    rhotic_parser.add_argument("voicebank", type=Path)
    rhotic_parser.add_argument("--json", action="store_true")

    splice_parser = subparsers.add_parser("splice-test")
    splice_parser.add_argument("voicebank", type=Path)
    splice_parser.add_argument("--base", default="sh", choices=("sh",))
    splice_parser.add_argument("--json", action="store_true")

    render_parser = subparsers.add_parser("render-ab")
    render_parser.add_argument("voicebank", type=Path)
    render_parser.add_argument("--base", default="sh", choices=("sh",))
    render_parser.add_argument("--per-subbank", type=int, default=3)
    render_parser.add_argument("--output", type=Path, default=Path("phonoweave_ab"))
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


def _region_payload(region) -> dict[str, object]:
    return {
        "standardized_distance": region.standardized_distance,
        "loo_balanced_accuracy": region.loo_balanced_accuracy,
        "permutation_p": region.permutation_p,
        "mean_plain": region.mean_plain,
        "mean_rounded": region.mean_rounded,
        "standardized_effects": region.standardized_effects,
    }


def _analysis_payload(result) -> dict[str, object]:
    return {
        "base_unit": result.base_unit,
        "samples": result.samples,
        "skipped": result.skipped,
        "mean_core_distance": result.mean_core_distance,
        "core_distance_cv": result.core_distance_cv,
        "mean_late_distance": result.mean_late_distance,
        "late_distance_cv": result.late_distance_cv,
        "cross_core_balanced_accuracy": result.cross_core_balanced_accuracy,
        "cross_late_balanced_accuracy": result.cross_late_balanced_accuracy,
        "cross_core_by_subbank": result.cross_core_by_subbank,
        "cross_late_by_subbank": result.cross_late_by_subbank,
        "subbanks": [
            {
                "subbank": item.subbank,
                "plain_count": item.plain_count,
                "rounded_count": item.rounded_count,
                "core": _region_payload(item.core),
                "late": _region_payload(item.late),
            }
            for item in result.subbanks
        ],
    }


def _affricate_payload(result) -> dict[str, object]:
    return {
        "base_unit": result.base_unit,
        "samples": result.samples,
        "skipped": result.skipped,
        "mean_distance": result.mean_distance,
        "distance_cv": result.distance_cv,
        "cross_subbank_balanced_accuracy": result.cross_subbank_balanced_accuracy,
        "cross_by_subbank": result.cross_by_subbank,
        "subbanks": [
            {
                "subbank": item.subbank,
                "plain_count": item.plain_count,
                "rounded_count": item.rounded_count,
                "distance": item.distance,
                "loo_balanced_accuracy": item.loo_balanced_accuracy,
                "permutation_p": item.permutation_p,
                "mean_plain": item.mean_plain,
                "mean_rounded": item.mean_rounded,
                "effects": item.effects,
            }
            for item in result.subbanks
        ],
    }


def _rhotic_payload(result) -> dict[str, object]:
    return {
        "samples": result.samples,
        "skipped": result.skipped,
        "cross_subbank_balanced_accuracy": result.cross_subbank_balanced_accuracy,
        "cross_by_subbank": result.cross_by_subbank,
        "subbanks": [
            {
                "subbank": item.subbank,
                "counts": item.counts,
                "loo_balanced_accuracy": item.loo_balanced_accuracy,
                "means": item.means,
            }
            for item in result.subbanks
        ],
        "pairwise": [
            {
                "left": pair.left,
                "right": pair.right,
                "cross_subbank_balanced_accuracy": pair.cross_subbank_balanced_accuracy,
                "cross_by_subbank": pair.cross_by_subbank,
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
            for pair in result.pairwise
        ],
    }


def _splice_payload(result) -> dict[str, object]:
    return {
        "base_unit": result.base_unit,
        "targets": result.targets,
        "skipped": result.skipped,
        "mean_delta": result.mean_delta,
        "mean_relative_delta": result.mean_relative_delta,
        "permutation_p": result.permutation_p,
        "subbanks": [
            {
                "subbank": item.subbank,
                "targets": item.targets,
                "mean_natural_penalty": item.mean_natural_penalty,
                "mean_rounded_control_penalty": item.mean_rounded_control_penalty,
                "mean_plain_substitution_penalty": item.mean_plain_substitution_penalty,
                "mean_delta": item.mean_delta,
                "mean_relative_delta": item.mean_relative_delta,
                "permutation_p": item.permutation_p,
            }
            for item in result.subbanks
        ],
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
                    print(f"  {color}: prefix={prefix!r}, suffix={suffix!r}, tones={ranges}")

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

    if args.command == "analyze":
        result = analyze_fricative_contrast(args.voicebank, args.base)
        payload = _analysis_payload(result)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"Base unit: {result.base_unit}")
            print(f"Samples: {result.samples}")
            print(f"Skipped: {result.skipped}")
            if result.mean_core_distance is not None:
                print(f"Mean core distance: {result.mean_core_distance:.3f}")
                print(f"Core distance CV: {result.core_distance_cv:.3f}")
            if result.mean_late_distance is not None:
                print(f"Mean late distance: {result.mean_late_distance:.3f}")
                print(f"Late distance CV: {result.late_distance_cv:.3f}")
            if result.cross_core_balanced_accuracy is not None:
                print(f"Cross-subbank core balanced accuracy: {result.cross_core_balanced_accuracy:.3f}")
                details = ", ".join(f"{name}={score:.3f}" for name, score in result.cross_core_by_subbank.items())
                print(f"  held out: {details}")
            if result.cross_late_balanced_accuracy is not None:
                print(f"Cross-subbank late balanced accuracy: {result.cross_late_balanced_accuracy:.3f}")
                details = ", ".join(f"{name}={score:.3f}" for name, score in result.cross_late_by_subbank.items())
                print(f"  held out: {details}")
            print()
            for item in result.subbanks:
                print(f"{item.subbank}: plain={item.plain_count}, rounded={item.rounded_count}")
                for name, region in (("core", item.core), ("late", item.late)):
                    print(
                        f"  {name}: distance={region.standardized_distance:.3f}, "
                        f"loo_balanced_accuracy={region.loo_balanced_accuracy:.3f}, "
                        f"permutation_p={region.permutation_p:.4f}"
                    )
                    print(
                        f"    centroid_hz: {region.mean_plain['centroid_hz']:.1f} -> "
                        f"{region.mean_rounded['centroid_hz']:.1f} "
                        f"(effect={region.standardized_effects['centroid_hz']:+.3f})"
                    )
                    print(
                        f"    high_band_ratio: {region.mean_plain['high_band_ratio']:.3f} -> "
                        f"{region.mean_rounded['high_band_ratio']:.3f} "
                        f"(effect={region.standardized_effects['high_band_ratio']:+.3f})"
                    )
                    print(
                        f"    slope: {region.mean_plain['slope']:.3f} -> "
                        f"{region.mean_rounded['slope']:.3f} "
                        f"(effect={region.standardized_effects['slope']:+.3f})"
                    )
        return 0

    if args.command == "analyze-affricate":
        result = analyze_affricate_contrast(args.voicebank, args.base)
        payload = _affricate_payload(result)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"Base unit: {result.base_unit}")
            print(f"Samples: {result.samples}")
            print(f"Skipped: {result.skipped}")
            if result.mean_distance is not None:
                print(f"Mean distance: {result.mean_distance:.3f}")
                print(f"Distance CV: {result.distance_cv:.3f}")
            if result.cross_subbank_balanced_accuracy is not None:
                print(f"Cross-subbank balanced accuracy: {result.cross_subbank_balanced_accuracy:.3f}")
                details = ", ".join(f"{name}={score:.3f}" for name, score in result.cross_by_subbank.items())
                print(f"  held out: {details}")
            print()
            for item in result.subbanks:
                print(
                    f"{item.subbank}: plain={item.plain_count}, rounded={item.rounded_count}, "
                    f"distance={item.distance:.3f}, "
                    f"loo_balanced_accuracy={item.loo_balanced_accuracy:.3f}, "
                    f"permutation_p={item.permutation_p:.4f}"
                )
                print(
                    f"  centroid_hz: {item.mean_plain['centroid_hz']:.1f} -> "
                    f"{item.mean_rounded['centroid_hz']:.1f} "
                    f"(effect={item.effects['centroid_hz']:+.3f})"
                )
                print(
                    f"  high_band_ratio: {item.mean_plain['high_band_ratio']:.3f} -> "
                    f"{item.mean_rounded['high_band_ratio']:.3f} "
                    f"(effect={item.effects['high_band_ratio']:+.3f})"
                )
                print(
                    f"  frication_duration_ms: {item.mean_plain['frication_duration_ms']:.1f} -> "
                    f"{item.mean_rounded['frication_duration_ms']:.1f} "
                    f"(effect={item.effects['frication_duration_ms']:+.3f})"
                )
        return 0

    if args.command == "analyze-rhotic":
        result = analyze_rhotic_contrast(args.voicebank)
        payload = _rhotic_payload(result)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("Base unit: r")
            print(f"Samples: {result.samples}")
            print(f"Skipped: {result.skipped}")
            if result.cross_subbank_balanced_accuracy is not None:
                print(f"Cross-subbank balanced accuracy: {result.cross_subbank_balanced_accuracy:.3f}")
                details = ", ".join(f"{name}={score:.3f}" for name, score in result.cross_by_subbank.items())
                print(f"  held out: {details}")
            print()
            for item in result.subbanks:
                counts = ", ".join(f"{name}={count}" for name, count in item.counts.items())
                print(f"{item.subbank}: {counts}, loo_balanced_accuracy={item.loo_balanced_accuracy:.3f}")
                for context in ("plain", "front", "rounded"):
                    means = item.means.get(context)
                    if means is None:
                        continue
                    print(
                        f"  {context}: periodicity={means['periodicity']:.3f}, "
                        f"flatness={means['spectral_flatness']:.3f}, "
                        f"centroid_hz={means['centroid_hz']:.1f}, "
                        f"F2={means['f2_hz']:.1f}, F3={means['f3_hz']:.1f}, "
                        f"F3-F2={means['f3_minus_f2_hz']:.1f}"
                    )
            print()
            print("Pairwise:")
            for pair in result.pairwise:
                cross = (
                    f"{pair.cross_subbank_balanced_accuracy:.3f}"
                    if pair.cross_subbank_balanced_accuracy is not None
                    else "n/a"
                )
                print(f"  {pair.left} vs {pair.right}: cross_subbank_balanced_accuracy={cross}")
                if pair.cross_by_subbank:
                    details = ", ".join(f"{name}={score:.3f}" for name, score in pair.cross_by_subbank.items())
                    print(f"    held out: {details}")
                for item in pair.subbanks:
                    print(
                        f"    {item.subbank}: {item.left_count}/{item.right_count}, "
                        f"distance={item.distance:.3f}, "
                        f"loo_balanced_accuracy={item.loo_balanced_accuracy:.3f}, "
                        f"permutation_p={item.permutation_p:.4f}"
                    )
        return 0

    if args.command == "splice-test":
        result = splice_relevance_test(args.voicebank, args.base)
        payload = _splice_payload(result)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"Base unit: {result.base_unit}")
            print(f"Rounded targets: {result.targets}")
            print(f"Skipped: {result.skipped}")
            if result.mean_delta is not None:
                print(f"Mean substitution delta: {result.mean_delta:+.4f}")
                print(f"Mean relative delta: {result.mean_relative_delta:+.1%}")
                print(f"Permutation p: {result.permutation_p:.4f}")
            print()
            for item in result.subbanks:
                print(f"{item.subbank}: targets={item.targets}")
                print(f"  natural boundary: {item.mean_natural_penalty:.4f}")
                print(f"  rounded control: {item.mean_rounded_control_penalty:.4f}")
                print(f"  plain substitution: {item.mean_plain_substitution_penalty:.4f}")
                print(f"  delta: {item.mean_delta:+.4f} ({item.mean_relative_delta:+.1%})")
                print(f"  permutation_p: {item.permutation_p:.4f}")
        return 0

    if args.command == "render-ab":
        if args.per_subbank < 1:
            raise SystemExit("--per-subbank must be at least 1")
        result = render_ab_pairs(
            args.voicebank,
            args.output,
            base_unit=args.base,
            per_subbank=args.per_subbank,
        )
        print(f"Output: {result.output_dir}")
        print(f"Pairs: {len(result.pairs)}")
        for pair in result.pairs:
            print(f"{pair.subbank}: target={pair.target_alias}, delta={pair.delta:+.4f}")
            print(f"  N: {pair.natural_path.name}")
            print(f"  A: {pair.a_path.name}  donor={pair.rounded_donor_alias}")
            print(f"  B: {pair.b_path.name}  donor={pair.plain_donor_alias}")
        print(f"Manifest: {result.output_dir / 'manifest.csv'}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
