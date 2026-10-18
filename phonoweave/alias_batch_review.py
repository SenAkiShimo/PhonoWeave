"""Batch naming policy review for traceable alias/OTO annotation candidates.

Group approval is ONLY a naming-policy decision. Each source-specific boundary
and role requires its own explicit approval and independently accepted source
context review before an annotation task can be issued. Never writes oto.ini.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from string import Formatter

from .alias_oto_task_compiler import (
    _ALIAS, _canonical_sha, _read_json, _source_index,
    build_review_template, compile_oto_tasks, load_pack, write_compiled_tasks,
)

SCHEMA = "phonoweave.batch_alias_naming_review.v0.1"
PREVIEW_SCHEMA = "phonoweave.batch_alias_naming_preview.v0.1"
ALLOWED_FIELDS = frozenset({
    "unit_id", "context", "pitch_group", "task_id", "line_id",
    "syllable", "syllable_index",
})
GROUP_EDITABLE = frozenset({"decision", "alias_format", "reviewer_note"})
TARGET_EDITABLE = frozenset({
    "decision", "role_verified", "boundary_verified", "reviewer_note",
})


def _format_alias(pattern: str, target: dict) -> str:
    if not isinstance(pattern, str) or not pattern or len(pattern) > 180:
        raise ValueError("approved group requires a nonempty alias format")
    fields = set()
    try:
        for _literal, field, spec, conversion in Formatter().parse(pattern):
            if field is not None:
                if field not in ALLOWED_FIELDS or spec or conversion:
                    raise ValueError("unsafe alias format field")
                fields.add(field)
        if not fields:
            raise ValueError("alias format must reference a source target field")
        rendered = pattern.format_map({key: target[key] for key in fields})
    except (KeyError, IndexError, AttributeError, ValueError) as exc:
        raise ValueError(f"invalid alias format: {exc}") from exc
    if not _ALIAS.fullmatch(rendered) or rendered != rendered.strip():
        raise ValueError("rendered alias violates conservative alias grammar")
    return rendered


def make_batch_review(action: dict, directives: dict, records: list[dict],
                      fingerprint: str) -> dict:
    """Group source-backed targets by realization and context across pitches."""
    source = build_review_template(action, directives, records, fingerprint)
    groups_by_key = defaultdict(list)
    for target in source["targets"]:
        groups_by_key[(target["unit_id"], target["context"])].append(target)
    groups, target_groups = [], {}
    for number, key in enumerate(sorted(groups_by_key), 1):
        group_id = f"AN-{number:03d}"
        members = groups_by_key[key]
        for member in members:
            target_groups[member["task_id"]] = group_id
        groups.append({
            "group_id": group_id, "unit_id": key[0], "context": key[1],
            "intended_role": "internal_cv",
            "pitch_groups": sorted({x["pitch_group"] for x in members}),
            "target_task_ids": [x["task_id"] for x in members],
            "target_count": len(members),
            "decision": "pending", "alias_format": "", "reviewer_note": "",
        })
    return {
        "schema": SCHEMA, "speaker_id": source["speaker_id"],
        "source_pack_sha256": fingerprint,
        "source_action_plan_sha256": source["source_action_plan_sha256"],
        "source_targets_sha256": _canonical_sha({"targets": source["targets"]}),
        "reviewer_id": "", "group_count": len(groups),
        "target_count": len(source["targets"]), "groups": groups,
        "targets": [{
            "task_id": row["task_id"], "group_id": target_groups[row["task_id"]],
            "decision": "pending", "role_verified": False,
            "boundary_verified": False, "reviewer_note": "",
        } for row in source["targets"]],
        "scope": "review_only_not_production_mapping",
    }


def _check_immutable(actual: dict, original: dict, editable: frozenset,
                     label: str) -> None:
    if not isinstance(actual, dict) or set(actual) != set(original):
        raise ValueError(f"{label} schema/fields changed from frozen template")
    if any(actual[key] != val for key, val in original.items()
           if key not in editable):
        raise ValueError(f"{label} identity changed from frozen template")


def expand_batch_review(batch: dict, action: dict, directives: dict,
                        records: list[dict], fingerprint: str) -> tuple[dict, dict]:
    """Render approved naming *proposals*; never infer per-target approvals."""
    expected = make_batch_review(action, directives, records, fingerprint)
    if not isinstance(batch, dict):
        raise ValueError("batch review must be a JSON object")
    immutable_root = set(expected) - {"reviewer_id", "groups", "targets"}
    if set(batch) != set(expected) or any(
        batch.get(key) != expected[key] for key in immutable_root
    ):
        raise ValueError("batch review stale or source pack mismatch")
    reviewer = batch["reviewer_id"]
    if not isinstance(reviewer, str):
        raise ValueError("reviewer identity must be a string")
    if (not isinstance(batch["groups"], list) or not isinstance(batch["targets"], list)
            or len(batch["groups"]) != len(expected["groups"])
            or len(batch["targets"]) != len(expected["targets"])):
        raise ValueError("batch group/target count mismatch")

    group_by_id = {}
    decisions = Counter()
    approved_groups = []
    for group, frozen in zip(batch["groups"], expected["groups"]):
        _check_immutable(group, frozen, GROUP_EDITABLE, "group")
        choice = group["decision"]
        if choice not in {"pending", "approve_naming", "reject"}:
            raise ValueError("invalid group naming decision")
        if not isinstance(group["alias_format"], str) or not isinstance(group["reviewer_note"], str):
            raise ValueError("invalid group naming review fields")
        if choice == "approve_naming":
            if not (reviewer.strip() and group["reviewer_note"].strip()):
                raise ValueError("approved group requires reviewer and rationale")
            # Formatting validation is target-aware, below.
            approved_groups.append(group["group_id"])
        elif (group["alias_format"] or
              (choice == "reject" and not group["reviewer_note"].strip())):
            raise ValueError("unapproved group has alias format or rejection lacks rationale")
        decisions[choice] += 1
        group_by_id[group["group_id"]] = group

    source = build_review_template(action, directives, records, fingerprint)
    source["reviewer_id"] = reviewer
    reserved = _source_index(records)[1]
    proposed = set()
    preview, collisions = [], []
    for actual, frozen, individual in zip(
            batch["targets"], expected["targets"], source["targets"]):
        _check_immutable(actual, frozen, TARGET_EDITABLE, "target")
        group = group_by_id[frozen["group_id"]]
        choice = actual["decision"]
        if choice not in {"pending", "approve_annotation", "reject"}:
            raise ValueError("invalid individual annotation decision")
        if (type(actual["role_verified"]) is not bool or
                type(actual["boundary_verified"]) is not bool or
                not isinstance(actual["reviewer_note"], str)):
            raise ValueError("invalid individual review fields")
        if choice != "approve_annotation" and (
            actual["role_verified"] or actual["boundary_verified"]
        ):
            raise ValueError("unapproved target cannot be marked verified")
        if choice == "reject" and not actual["reviewer_note"].strip():
            raise ValueError("rejected target requires rationale")
        if choice == "approve_annotation" and (
            group["decision"] != "approve_naming" or
            not actual["role_verified"] or not actual["boundary_verified"] or
            not actual["reviewer_note"].strip() or not reviewer.strip()
        ):
            raise ValueError("target approval requires approved group and independent role/boundary review")
        alias = None
        conflict = False
        if group["decision"] == "approve_naming":
            alias = _format_alias(group["alias_format"], individual)
            conflict = alias in reserved or alias in proposed
            if conflict:
                collisions.append({"task_id": individual["task_id"],
                                   "group_id": group["group_id"], "alias": alias})
            proposed.add(alias)
        # Transfer only explicit per-target decisions to the existing strict
        # source/independent-context compiler. A group approval is not enough.
        individual.update({
            "decision": choice, "new_alias": alias if choice == "approve_annotation" else "",
            "intended_role": "internal_cv" if choice == "approve_annotation" else "",
            "role_verified": actual["role_verified"],
            "boundary_verified": actual["boundary_verified"],
            "reviewer_note": actual["reviewer_note"],
        })
        preview.append({
            "task_id": individual["task_id"], "group_id": group["group_id"],
            "source_oto": individual["source_oto"],
            "oto_line_number": individual["oto_line_number"],
            "pitch_group": individual["pitch_group"],
            "original_alias": individual["original_alias"],
            "proposed_alias": alias, "collision": conflict,
            "individual_decision": choice,
            "annotation_approval": "requires_independent_review",
        })
    if collisions:
        raise ValueError("proposed alias collision with existing or other suggested names: " +
                         ", ".join(x["task_id"] for x in collisions[:8]))
    return source, {
        "schema": PREVIEW_SCHEMA,
        "source_pack_sha256": fingerprint,
        "group_count": len(group_by_id),
        "target_count": len(preview),
        "group_decisions": dict(sorted(decisions.items())),
        "approved_naming_groups": approved_groups,
        "explicitly_approved_individual_targets": sum(
            x["individual_decision"] == "approve_annotation" for x in preview),
        "name_proposals": preview,
        "production_alias_export": "disabled",
        "new_oto_timings": "none",
    }


def _save_json_new(path: Path, payload: dict) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch naming rules, separate source-specific alias/OTO reviews")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("template", "preview", "compile"):
        sub = commands.add_parser(name)
        sub.add_argument("--pack", type=Path, required=True)
        if name != "template":
            sub.add_argument("--batch-reviews", type=Path, required=True)
        if name == "compile":
            sub.add_argument("--reviewed-scheme", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _manifest, action, directives, records, fingerprint = load_pack(args.pack)
        if args.command == "template":
            result = make_batch_review(action, directives, records, fingerprint)
            _save_json_new(args.output, result)
            print("Batch naming groups:", result["group_count"])
            print("Source-specific candidates:", result["target_count"])
        else:
            batch = _read_json(args.batch_reviews.expanduser())
            expanded, preview = expand_batch_review(
                batch, action, directives, records, fingerprint)
            if args.command == "preview":
                _save_json_new(args.output, preview)
                print("Naming preview: source roles and OTO unchanged")
            else:
                reviewed = _read_json(args.reviewed_scheme.expanduser())
                # The original strict compiler rechecks each source target,
                # the independent acoustic/context review and name collisions.
                result = compile_oto_tasks(
                    expanded, action, directives, records, fingerprint, reviewed)
                result["batch_review"] = {
                    "schema": PREVIEW_SCHEMA,
                    "source_pack_sha256": fingerprint,
                    "group_decisions": preview["group_decisions"],
                    "group_count": preview["group_count"],
                }
                location = write_compiled_tasks(result, args.output)
                print("Approved manual OTO tasks:", result["approved_manual_annotation_tasks"])
                print("Task pack:", location)
        print("Production OTO/phonemizer export: disabled")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
