from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bank_audit import audit_bank, format_audit
from .bank_concordance import compare_with_bank, format_concordance
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
    add_reclist_arguments(parser)
    args = parser.parse_args(argv)

    syllables = load_syllables(args)
    if not syllables:
        print("No Mandarin syllables recognized in the voicebank.", file=sys.stderr)
        return 1

    analysis = analyze_voicebank_inventory(args.voicebank)
    profile = build_speaker_profile(args.voicebank, analysis)
    inventory = build_synthesis_inventory(args.voicebank, profile)

    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    inventory_path = directory / "synthesis_inventory.yaml"
    inventory_path.write_text(synthesis_inventory_yaml(inventory), encoding="utf-8")

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

    concordance_path = directory / "bank_concordance.txt"
    concordance_path.write_text(
        format_concordance(compare_with_bank(audit, inventory, syllables)) + "\n",
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

    print_summary(compiled, len(syllables), args.unresolved)
    print(
        f"Unresolved decisions: {len(gaps.gaps)} "
        "(treated as one unit per onset; no recording needed)"
    )
    print(f"Inventory: {inventory_path}")
    print(f"Bank audit: {audit_path}")
    print(f"Comparison with the voicebank's own splits: {concordance_path}")
    if supplement_reclist is not None:
        print(f"Supplement reclist lines: {len(supplement_reclist.lines)}")
        print(f"Supplement reclist: {supplement_path}")
        print(f"Evidence gaps: {gaps_path}")
    return 0
