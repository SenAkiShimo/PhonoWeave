from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evidence_gap import build_evidence_completion_plan
from .inventory import analyze_voicebank_inventory


def _payload(plan) -> dict[str, object]:
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
        ]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave evidence-gaps")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    analysis = analyze_voicebank_inventory(args.voicebank)
    plan = build_evidence_completion_plan(analysis)

    if args.json:
        print(json.dumps(_payload(plan), ensure_ascii=False, indent=2))
        return 0

    print(f"Voicebank: {analysis.voicebank}")
    print(f"Unresolved evidence gaps: {len(plan.gaps)}")
    print(f"Supplemental recording: {len(plan.supplemental_recording)}")
    print(f"Perceptual validation: {len(plan.perceptual_validation)}")
    print()
    for gap in plan.gaps:
        print(f"{gap.base_unit} [{gap.class_name}]")
        print(f"  gap: {gap.gap_type}")
        print(f"  priority: {gap.priority}")
        print(f"  next: {gap.recommended_action}")
        if gap.role_scope:
            print(f"  role_scope: {gap.role_scope}")
        if gap.context_families:
            print(f"  context_families: {','.join(gap.context_families)}")
        for item in gap.rationale:
            print(f"  {item}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
