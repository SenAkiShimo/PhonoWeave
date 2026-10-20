"""Read-only structural evidence preflight for every recording-scheme task.

A matched source alias is not a verified acoustic segment or recording need.
No task resolution is written and no production alias, oto.ini, or WAV changes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

PLAN_SCHEMA = "phonoweave.recording_scheme_action_plan.v0.1"
RECORD_SCHEMA = "phonoweave.lossless_alias_record.v0.1"
ACTIONS = frozenset({
    "review_cross_pitch_coverage", "review_internal_candidate_mapping",
    "review_context_mapping", "review_role_scoped_mapping",
    "review_provisional_unit_decision",
})
CV_ROLES = frozenset({"initial_cv", "standalone_cv"})


def _example(record: dict) -> dict:
    source, role = record["source"], record["role"]
    return {
        "source_oto": source["source_oto"],
        "line_number": source["line_number"],
        "source_wav": source["source_wav"],
        "original_alias": source["original_alias"],
        "role": role["role"],
    }


def _index_records(records: list[dict]):
    by_pitch = defaultdict(list)
    by_identity = {}
    for record in records:
        if record.get("schema") != RECORD_SCHEMA:
            raise ValueError("unexpected alias record schema")
        source, role = record["source"], record["role"]
        keys = ("source_oto", "line_number", "original_alias")
        if any(source.get(k) != role.get(k) for k in keys):
            raise ValueError("alias evidence provenance mismatch")
        for optional in ("compilation_preview", "original_oto_timing"):
            item = record.get(optional)
            if item is not None and any(source.get(k) != item.get(k) for k in keys):
                raise ValueError("alias optional record provenance mismatch")
        ident = tuple(source[k] for k in keys)
        if ident in by_identity:
            raise ValueError("duplicate source OTO entry identity")
        by_identity[ident] = record
        if role.get("pitch_suffix_status") == "matched":
            by_pitch[role.get("pitch_suffix")].append(record)
    return by_pitch, by_identity


def _cross_pitch(task: dict, by_pitch: dict) -> list[dict]:
    findings = []
    for pitch in task["pitch_groups"]:
        for syllable in task["suggested_syllables"]:
            variants = []
            for group, group_records in sorted(by_pitch.items()):
                hits = [r for r in group_records
                        if r["role"].get("role") in CV_ROLES and
                        r["role"].get("right_token") == syllable]
                variants.append({
                    "pitch_group": group,
                    "matching_cv_aliases": len(hits),
                    "role_counts": dict(sorted(Counter(
                        r["role"]["role"] for r in hits).items())),
                    "source_examples": [_example(r) for r in hits[:3]],
                })
            findings.append({
                "task_id": task["task_id"], "pitch_group": pitch,
                "syllable": syllable, "status": "source_alias_evidence_only",
                "observed_pitch_variants": variants,
                "human_review_required": True,
            })
    return findings


def _task_details(task: dict, by_pitch: dict, by_identity: dict):
    action = task["review_action"]
    pitch_rows = {p: by_pitch.get(p, ()) for p in task["pitch_groups"]}
    base = task["base_unit"]
    if action == "review_internal_candidate_mapping":
        inspected = []
        for sample in task.get("source_examples", []):
            key = (sample["source_oto"], sample["line_number"],
                   sample["original_alias"])
            record = by_identity.get(key)
            reasons = []
            if record is None:
                reasons.append("source_entry_not_found")
            else:
                source, role = record["source"], record["role"]
                compiled = record.get("compilation_preview", {})
                if source.get("source_wav") != sample["source_wav"]:
                    reasons.append("source_wav_mismatch")
                if role.get("role") != "vc_transition" or role.get("right_token") != base:
                    reasons.append("vc_role_or_consonant_mismatch")
                if role.get("pitch_suffix") != sample["pitch_group"] or (
                        role.get("pitch_suffix_status") != "matched"):
                    reasons.append("pitch_mismatch")
                if compiled and compiled.get("compile_status") != "roundtrip_review_only":
                    reasons.append("alias_not_roundtrip_verified")
            inspected.append({
                "pitch_group": sample["pitch_group"],
                "source_oto": sample["source_oto"],
                "line_number": sample["line_number"],
                "source_wav": sample["source_wav"],
                "original_alias": sample["original_alias"],
                "structural_match": not reasons,
                "issues": reasons,
            })
        covered = {r["pitch_group"] for r in inspected if r["structural_match"]}
        complete = bool(inspected) and all(r["structural_match"] for r in inspected)
        return ("source_examples_consistent" if complete else
                "source_examples_need_review", {
                    "checked_source_examples": len(inspected),
                    "consistent_examples": sum(r["structural_match"] for r in inspected),
                    "pitch_groups_with_matching_examples": sorted(covered),
                    "source_examples": inspected,
                    "limit": "OTO roles and provenance only; acoustic boundary not checked",
                })
    if action == "review_context_mapping":
        evidence = []
        for pitch, records in pitch_rows.items():
            matches = [r for r in records
                       if r["source"].get("base_unit") == base and
                       r["role"].get("role") in CV_ROLES]
            evidence.append({
                "pitch_group": pitch,
                "observed_cv_aliases_for_base": len(matches),
                "role_counts": dict(sorted(Counter(
                    r["role"]["role"] for r in matches).items())),
                "source_examples": [_example(r) for r in matches[:2]],
            })
        return "observed_base_aliases_not_context_assigned", {
            "pitch_evidence": evidence,
            "limit": "Orthographic base match cannot prove split/merge context or quality",
        }
    if action == "review_role_scoped_mapping":
        evidence = []
        for pitch, records in pitch_rows.items():
            matches = [r for r in records
                       if r["role"].get("role") == "vc_transition" and
                       r["role"].get("right_token") == base]
            evidence.append({
                "pitch_group": pitch,
                "candidate_vc_aliases": len(matches),
                "previous_vowel_tokens": sorted({
                    r["role"].get("left_token", "") for r in matches}),
                "source_examples": [_example(r) for r in matches[:2]],
            })
        return "vc_aliases_not_following_vowel_assigned", {
            "pitch_evidence": evidence,
            "limit": "VC spelling alone cannot establish the following vowel family",
        }
    if action == "review_provisional_unit_decision":
        evidence = []
        for pitch, records in pitch_rows.items():
            matches = [r for r in records if
                       r["source"].get("candidate_unit_id") == task["unit_id"]
                       and r["role"].get("role") in CV_ROLES]
            evidence.append({
                "pitch_group": pitch,
                "direct_candidate_cv_aliases": len(matches),
                "source_examples": [_example(r) for r in matches[:2]],
            })
        return "alias_inventory_only_no_synthesis_outcome", {
            "pitch_evidence": evidence,
            "limit": "Source inventory cannot validate acoustic or perceptual split advantage",
        }
    raise ValueError("unknown review action")


def audit_plan_evidence(plan: dict, records: list[dict]) -> dict:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unexpected plan schema")
    if not isinstance(plan.get("tasks"), list):
        raise ValueError("action plan tasks must be a list")
    by_pitch, by_identity = _index_records(records)
    findings, task_findings = [], []
    seen = set()
    for task in plan["tasks"]:
        task_id, action = task["task_id"], task["review_action"]
        if task_id in seen or action not in ACTIONS:
            raise ValueError("duplicate task ID or unknown review action")
        seen.add(task_id)
        if action == "review_cross_pitch_coverage":
            cross = _cross_pitch(task, by_pitch)
            findings.extend(cross)
            status = "source_alias_evidence_only"
            detail = {"cross_pitch_findings": cross}
        else:
            status, detail = _task_details(task, by_pitch, by_identity)
        task_findings.append({
            "task_id": task_id, "review_action": action,
            "unit_id": task.get("unit_id", ""), "pitch_groups": task["pitch_groups"],
            "structural_status": status, "evidence": detail,
            "human_review_required": True,
            "review_resolution": "unchanged",
        })
    return {
        "schema": "phonoweave.automatic_evidence_precheck.v0.2",
        "speaker_id": plan["speaker_id"],
        "review_tasks_checked": len(task_findings),
        "review_action_counts": dict(sorted(Counter(
            x["review_action"] for x in task_findings).items())),
        "structural_status_counts": dict(sorted(Counter(
            x["structural_status"] for x in task_findings).items())),
        "cross_pitch_findings": findings,
        "task_findings": task_findings,
        "human_resolutions_written": 0,
        "recordings_authorized": 0,
        "production_alias_export": "disabled",
        "limitations": [
            "Structural source checks do not verify recorded acoustic content.",
            "Evidence prechecks do not validate split/merge benefits.",
            "All human review resolutions remain unchanged.",
        ],
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
