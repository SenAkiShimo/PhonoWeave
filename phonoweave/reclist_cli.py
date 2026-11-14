from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .final_classes import derive_final_classes
from .reclist import (
    STOP_ONSETS,
    Reclist,
    Syllable,
    UnitMap,
    build_reclist,
    missing_requirements,
    observed_syllables,
    reclist_report,
)
from .tolerance import apply_tolerance
from .synthesis_inventory import (
    SynthesisInventory,
    build_synthesis_inventory,
    load_synthesis_inventory,
)


@dataclass(frozen=True)
class CompiledReclist:
    reclist: Reclist
    baseline: Reclist
    alternative: Reclist
    alternative_mode: str
    units: UnitMap
    report: dict[str, object]
    output: Path
    report_path: Path


def compile_reclist(
    syllables: tuple[Syllable, ...],
    inventory: SynthesisInventory,
    output: Path,
    report_path: Path | None = None,
    max_syllables: int = 10,
    restarts: int = 40,
    seed: int = 0,
    unresolved: str = "collapse",
    stop_vc: str = "shared",
) -> CompiledReclist:
    collapse = unresolved == "collapse"
    shared = STOP_ONSETS if stop_vc == "shared" else frozenset()
    units = UnitMap(inventory, collapse_unresolved=collapse, shared_connection=shared)
    reclist = build_reclist(syllables, units, max_syllables, restarts, seed)
    missing = missing_requirements(reclist.lines, reclist.requirements, units)
    if any(missing.values()):
        raise ValueError(f"coverage check failed: {missing}")

    baseline = build_reclist(syllables, UnitMap(None), max_syllables, restarts, seed)
    alternative = build_reclist(
        syllables,
        UnitMap(inventory, collapse_unresolved=not collapse, shared_connection=shared),
        max_syllables,
        restarts,
        seed,
    )
    alternative_mode = "split" if collapse else "collapse"

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(reclist.text(), encoding="utf-8")

    report_path = (
        report_path.expanduser().resolve()
        if report_path is not None
        else output.with_suffix(".report.json")
    )
    report = reclist_report(
        reclist, syllables, units, inventory, baseline, alternative, alternative_mode
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return CompiledReclist(
        reclist=reclist,
        baseline=baseline,
        alternative=alternative,
        alternative_mode=alternative_mode,
        units=units,
        report=report,
        output=output,
        report_path=report_path,
    )


def print_summary(compiled: CompiledReclist, syllables: int, unresolved: str) -> None:
    requirements = compiled.report["requirements"]
    print(f"Syllables: {syllables}")
    print(
        f"Requirements: cv={requirements['cv']} vc={requirements['vc']} "
        f"start={requirements['start']} end={requirements['end']}"
    )
    print(f"Lines: {len(compiled.reclist.lines)} (lower bound {compiled.reclist.lower_bound})")
    print(f"Unresolved decisions: {unresolved}")
    print(f"Unresolved {compiled.alternative_mode}: {len(compiled.alternative.lines)} lines")
    print(f"Baseline, one unit per onset: {len(compiled.baseline.lines)} lines")
    if compiled.units.unmapped:
        print(f"Unmapped onset/final pairs: {len(compiled.units.unmapped)}")
    print(f"Reclist: {compiled.output}")
    print(f"Report: {compiled.report_path}")


def add_reclist_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tolerance",
        choices=("strict", "balanced", "loose"),
        default="strict",
        help="strict keeps every recommended split; balanced also merges splits that "
        "are not high-confidence; loose merges every split without strong acoustic evidence",
    )
    parser.add_argument(
        "--stop-vc",
        choices=("shared", "split"),
        default="shared",
        help="shared: a vowel-to-stop connection is recorded once per stop whatever "
        "vowel follows; split: one per context unit of the stop",
    )
    parser.add_argument(
        "--finals-from-bank",
        action="store_true",
        help="use the VC final classes the voicebank itself uses (WAV grouping) "
        "instead of one VC final per syllable final",
    )
    parser.add_argument("--max-syllables", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--unresolved",
        choices=("collapse", "split"),
        default="collapse",
        help="collapse: one unit per onset where the decision is unresolved; "
        "split: keep every unresolved group as its own unit",
    )


def load_syllables(args: argparse.Namespace) -> tuple[Syllable, ...]:
    classes = derive_final_classes(args.voicebank) if args.finals_from_bank else None
    syllables = observed_syllables(args.voicebank, classes)
    if classes is not None and syllables:
        tails = {item.tail for item in syllables}
        finals = {item.final for item in syllables}
        print(f"VC final classes from the voicebank: {len(finals)} finals -> {len(tails)} classes")
    return syllables


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave build-reclist")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("reclist.txt"))
    parser.add_argument("--report", type=Path)
    add_reclist_arguments(parser)
    args = parser.parse_args(argv)

    syllables = load_syllables(args)
    if not syllables:
        print("No Mandarin syllables recognized in the voicebank.", file=sys.stderr)
        return 1

    if args.inventory is not None:
        inventory = load_synthesis_inventory(args.inventory)
    else:
        inventory = build_synthesis_inventory(args.voicebank)
    inventory, merged = apply_tolerance(inventory, args.tolerance)
    if merged:
        print(f"Merged under tolerance {args.tolerance}: {' '.join(merged)}")

    try:
        compiled = compile_reclist(
            syllables,
            inventory,
            args.output,
            args.report,
            args.max_syllables,
            args.restarts,
            args.seed,
            args.unresolved,
            args.stop_vc,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    print_summary(compiled, len(syllables), args.unresolved)
    return 0
