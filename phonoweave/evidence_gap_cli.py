from __future__ import annotations

import argparse
import json
from pathlib import Path

from .diagnostic_selector import select_diagnostic_items
from .evidence_gap import build_evidence_completion_plan
from .inventory import analyze_voicebank_inventory
from .supplement_plan import build_supplement_plan
from .supplement_reclist import build_supplement_reclist


def _payload(plan, supplement, selection, reclist) -> dict[str, object]:
    return {
        "gaps": [
            {
                "base_unit": gap.base_unit,
                "class_name": gap.class_name,
                "gap_type": gap.gap_type,
                "priority": gap.priority,
                "recommended_action": gap.recommended_action,
                "role_scope": gap.role_scope,
                "context_families": list(gap.context_families),
                "rationale": list(gap.rationale),
            }
            for gap in plan.gaps
        ],
        "supplement": {
            "diagnostic_items": supplement.diagnostic_items,
            "requests": [
                {
                    "base_unit": request.base_unit,
                    "class_name": request.class_name,
                    "gap_type": request.gap_type,
                    "priority": request.priority,
                    "role_scope": request.role_scope,
                    "targets": [
                        {
                            "context_family": target.context_family,
                            "diagnostic_items": target.diagnostic_items,
                        }
                        for target in request.targets
                    ],
                    "pitch_policy": request.pitch_policy,
                    "automatic_round_limit": request.automatic_round_limit,
                    "stop_rule": request.stop_rule,
                }
                for request in supplement.requests
            ],
            "selection": [
                {
                    "base_unit": item.base_unit,
                    "final": item.final,
                    "syllable": item.syllable,
                    "context_family": item.context_family,
                    "role_scope": item.role_scope,
                    "existing_observations": item.existing_observations,
                    "replicate": item.replicate,
                }
                for item in selection.items
            ],
            "unfilled": list(selection.unfilled),
            "reclist": [line.text for line in reclist.lines],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave evidence-gaps")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-reclist", type=Path)
    args = parser.parse_args(argv)

    analysis = analyze_voicebank_inventory(args.voicebank)
    plan = build_evidence_completion_plan(analysis)
    supplement = build_supplement_plan(plan)
    selection = select_diagnostic_items(args.voicebank, supplement)
    reclist = build_supplement_reclist(selection)

    if args.write_reclist is not None:
        output = args.write_reclist.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(reclist.text(), encoding="utf-8")

    if args.json:
        print(json.dumps(_payload(plan, supplement, selection, reclist), ensure_ascii=False, indent=2))
        return 0

    print(f"Voicebank: {analysis.voicebank}")
    print(f"Unresolved evidence gaps: {len(plan.gaps)}")
    print(f"Supplemental recording: {len(plan.supplemental_recording)}")
    print(f"Perceptual validation: {len(plan.perceptual_validation)}")
    print(f"Supplement diagnostic items: {supplement.diagnostic_items}")
    print(f"Selected diagnostic items: {len(selection.items)}")
    print(f"Unfilled diagnostic targets: {len(selection.unfilled)}")
    print(f"Supplement reclist lines: {len(reclist.lines)}")
    print()

    request_by_base = {request.base_unit: request for request in supplement.requests}
    selected_by_base: dict[str, list[object]] = {}
    for item in selection.items:
        selected_by_base.setdefault(item.base_unit, []).append(item)

    for gap in plan.gaps:
        print(f"{gap.base_unit} [{gap.class_name}]")
        print(f"  gap: {gap.gap_type}")
        print(f"  priority: {gap.priority}")
        print(f"  next: {gap.recommended_action}")
        if gap.role_scope:
            print(f"  role_scope: {gap.role_scope}")
        if gap.context_families:
            print(f"  context_families: {','.join(gap.context_families)}")
        request = request_by_base.get(gap.base_unit)
        if request is not None:
            print(f"  supplement_role: {request.role_scope or 'all'}")
            for target in request.targets:
                print(
                    f"  supplement_target: {target.context_family} "
                    f"x{target.diagnostic_items} diagnostic item(s)"
                )
            for item in selected_by_base.get(gap.base_unit, []):
                print(
                    f"  selected_item: {item.syllable} "
                    f"[{item.context_family}] replicate={item.replicate} "
                    f"existing_observations={item.existing_observations}"
                )
            print(f"  pitch_policy: {request.pitch_policy}")
            print(f"  automatic_round_limit: {request.automatic_round_limit}")
            print(f"  stop_rule: {request.stop_rule}")
        for item in gap.rationale:
            print(f"  {item}")
        print()

    print("Supplement reclist")
    for line in reclist.lines:
        print(f"  {line.text}")

    if selection.unfilled:
        print()
        print("Unfilled diagnostic targets")
        for item in selection.unfilled:
            print(f"  {item}")
    if args.write_reclist is not None:
        print()
        print(f"Reclist written: {args.write_reclist.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
