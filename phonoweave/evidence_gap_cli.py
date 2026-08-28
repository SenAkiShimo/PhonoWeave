from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evidence_gap import build_evidence_completion_plan
from .inventory import analyze_voicebank_inventory
from .supplement_plan import build_supplement_plan


def _payload(plan, supplement) -> dict[str, object]:
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
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave evidence-gaps")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    analysis = analyze_voicebank_inventory(args.voicebank)
    plan = build_evidence_completion_plan(analysis)
    supplement = build_supplement_plan(plan)

    if args.json:
        print(json.dumps(_payload(plan, supplement), ensure_ascii=False, indent=2))
        return 0

    print(f"Voicebank: {analysis.voicebank}")
    print(f"Unresolved evidence gaps: {len(plan.gaps)}")
    print(f"Supplemental recording: {len(plan.supplemental_recording)}")
    print(f"Perceptual validation: {len(plan.perceptual_validation)}")
    print(f"Supplement diagnostic items: {supplement.diagnostic_items}")
    print()
    request_by_base = {request.base_unit: request for request in supplement.requests}
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
            print(f"  pitch_policy: {request.pitch_policy}")
            print(f"  automatic_round_limit: {request.automatic_round_limit}")
            print(f"  stop_rule: {request.stop_rule}")
        for item in gap.rationale:
            print(f"  {item}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
