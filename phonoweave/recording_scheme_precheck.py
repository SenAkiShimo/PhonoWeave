"""Read-only automated alias-evidence checks for review tasks.

Automatic findings do not record human acceptance, authorize recordings,
or infer acoustic validity. They identify source presence versus absence
at specific pitch subbanks using raw JSONL strings, never CSV escaping.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path


def audit_plan_evidence(plan: dict, records: list[dict]) -> dict:
    if plan.get("schema") != "phonoweave.recording_scheme_action_plan.v0.1":
        raise ValueError("unexpected plan schema")
    indexed = defaultdict(list)
    for record in records:
        if record.get("schema") != "phonoweave.lossless_alias_record.v0.1":
            raise ValueError("unexpected alias record schema")
        src, role = record["source"], record["role"]
        if any(src.get(k) != role.get(k)
               for k in ("source_oto", "line_number", "original_alias")):
            raise ValueError("alias evidence provenance mismatch")
        pitch = role.get("pitch_suffix")
        if role.get("pitch_suffix_status") == "matched":
            indexed[pitch].append(record)
    findings = []
    for task in plan["tasks"]:
        if task["review_action"] != "review_cross_pitch_coverage":
            continue
        for pitch in task["pitch_groups"]:
            for syllable in task["suggested_syllables"]:
                variants = []
                for group, group_records in sorted(indexed.items()):
                    hits = [
                        r for r in group_records
                        if r["role"].get("role") in {"standalone_cv", "initial_cv"}
                        and r["role"].get("right_token") == syllable
                    ]
                    variants.append({
                        "pitch_group": group,
                        "matching_cv_aliases": len(hits),
                        "source_examples": [
                            {"source_oto": r["source"]["source_oto"],
                             "line_number": r["source"]["line_number"],
                             "source_wav": r["source"]["source_wav"],
                             "original_alias": r["source"]["original_alias"],
                             "role": r["role"]["role"]}
                            for r in hits[:3]
                        ],
                    })
                findings.append({
                    "task_id": task["task_id"], "pitch_group": pitch,
                    "syllable": syllable,
                    "status": "source_alias_evidence_only",
                    "observed_pitch_variants": variants,
                    "human_review_required": True,
                })
    return {
        "schema": "phonoweave.automatic_evidence_precheck.v0.1",
        "speaker_id": plan["speaker_id"],
        "review_tasks_checked": len({x["task_id"] for x in findings}),
        "cross_pitch_findings": findings,
        "human_resolutions_written": 0,
        "recordings_authorized": 0,
    }


def audit_files(plan_path: Path, records_path: Path, output: Path) -> dict:
    with plan_path.open(encoding="utf-8") as f:
        plan = json.load(f)
    with records_path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    result = audit_plan_evidence(plan, records)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return result
