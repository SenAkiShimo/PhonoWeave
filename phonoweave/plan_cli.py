from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bank_audit import audit_bank, format_audit
from .bank_concordance import compare_with_bank, format_concordance
from .bank_supplement import build_bank_supplement, select_rows, uncovered_connections
from .diagnostic_selector import select_diagnostic_items
from .evidence_gap import build_evidence_completion_plan
from .evidence_gap_cli import _payload
from .inventory import analyze_voicebank_inventory
from .profile import build_speaker_profile
from .reclist_cli import (
    add_reclist_arguments,
    compile_reclist,
    load_syllables,
    print_summary,
)
from .supplement_plan import build_supplement_plan
from .supplement_reclist import build_supplement_reclist
from .tolerance import apply_tolerance
from .transition_evidence import apply_transition_evidence, collect_reports
from .synthesis_inventory import build_synthesis_inventory, synthesis_inventory_yaml


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave plan")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("phonoweave_plan"))
    parser.add_argument(
        "--with-supplement",
        action="store_true",
        help="also write the optional supplementary recording list and evidence gaps",
    )
    parser.add_argument(
        "--transition-evidence",
        action="store_true",
        help="split m, n and l along the axes the transition analysis supports",
    )
    parser.add_argument("--transition-permutations", type=int, default=4000)
    parser.add_argument(
        "--supplement-onsets",
        nargs="+",
        metavar="ONSET",
        help="only include these onsets in bank_supplement.txt",
    )
    parser.add_argument(
        "--single-layer-evidence",
        action="store_true",
        help="for a voicebank with one usable layer, judge s, sh, x and the affricates "
        "from within-layer tests (low confidence)",
    )
    add_reclist_arguments(parser)
    args = parser.parse_args(argv)

    syllables = load_syllables(args)
    if not syllables:
        print("No Mandarin syllables recognized in the voicebank.", file=sys.stderr)
        return 1

    analysis = analyze_voicebank_inventory(args.voicebank, args.single_layer_evidence)
    profile = build_speaker_profile(args.voicebank, analysis)
    inventory = build_synthesis_inventory(args.voicebank, profile)
    transition_notes: list[str] = []
    if args.transition_evidence:
        inventory, transition_notes = apply_transition_evidence(
            inventory,
            collect_reports(args.voicebank, permutations=args.transition_permutations),
        )
    raw_inventory = inventory
    inventory, merged = apply_tolerance(inventory, args.tolerance)

    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    inventory_path = directory / "synthesis_inventory.yaml"
    inventory_path.write_text(synthesis_inventory_yaml(raw_inventory), encoding="utf-8")

    try:
        compiled = compile_reclist(
            syllables,
            inventory,
            directory / "reclist.txt",
            directory / "reclist.report.json",
            args.max_syllables,
            args.restarts,
            args.seed,
            args.unresolved,
            args.stop_vc,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    audit = audit_bank(args.voicebank)
    audit_path = directory / "bank_audit.txt"
    audit_path.write_text(format_audit(audit) + "\n", encoding="utf-8")
    (directory / "bank_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = compare_with_bank(audit, inventory, syllables, args.tolerance)
    supplement_rows = select_rows(rows, args.supplement_onsets)
    bank_supplement = build_bank_supplement(supplement_rows, syllables, args.max_syllables)
    missing_connections = uncovered_connections(bank_supplement, supplement_rows, syllables)
    if missing_connections:
        print(f"Supplement coverage check failed: {sorted(missing_connections)}", file=sys.stderr)
        return 1
    bank_supplement_path = directory / "bank_supplement.txt"
    if bank_supplement.lines:
        bank_supplement_path.write_text(bank_supplement.text(), encoding="utf-8")
    elif bank_supplement_path.exists():
        bank_supplement_path.unlink()

    concordance_path = directory / "bank_concordance.txt"
    concordance_path.write_text(
        format_concordance(rows) + "\n",
        encoding="utf-8",
    )

    gaps = build_evidence_completion_plan(analysis)
    supplement_path = directory / "supplement_reclist.txt"
    gaps_path = directory / "evidence_gaps.json"
    supplement_reclist = None
    if args.with_supplement:
        supplement = build_supplement_plan(gaps)
        selection = select_diagnostic_items(args.voicebank, supplement)
        supplement_reclist = build_supplement_reclist(selection)
        supplement_path.write_text(supplement_reclist.text(), encoding="utf-8")
        gaps_path.write_text(
            json.dumps(
                _payload(gaps, supplement, selection, supplement_reclist),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for note in transition_notes:
        print(f"Transition evidence: {note}")
    if merged:
        print(f"Merged under tolerance {args.tolerance}: {' '.join(merged)}")
    print_summary(compiled, len(syllables), args.unresolved)
    print(
        f"Unresolved decisions: {len(gaps.gaps)} "
        "(treated as one unit per onset; no recording needed)"
    )
    print(f"Inventory: {inventory_path}")
    print(f"Bank audit: {audit_path}")
    print(f"Comparison with the voicebank's own splits: {concordance_path}")
    if bank_supplement.lines:
        print(
            f"Supplement for this voicebank: {len(bank_supplement.lines)} lines for "
            f"{', '.join(bank_supplement.labels)} ({bank_supplement_path})"
        )
    else:
        print("Supplement for this voicebank: nothing to record at this tolerance")
    if supplement_reclist is not None:
        print(f"Supplement reclist lines: {len(supplement_reclist.lines)}")
        print(f"Supplement reclist: {supplement_path}")
        print(f"Evidence gaps: {gaps_path}")
    return 0
