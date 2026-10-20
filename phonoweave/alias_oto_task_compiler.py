"""Compile reviewed source-backed alias naming into manual OTO annotation tasks.

Approval here authorizes an *annotation work item*, not an installed alias,
phonemizer route, OTO timing, new WAV, or acoustic/synthesis quality claim.
Review pack source hashes and individual OTO provenance are independently
verified. Existing legacy aliases are reserved against name collisions.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile

from .recording_scheme_decisions import _hash_plan, RESOLUTIONS

SCHEMA = "phonoweave.alias_oto_annotation_tasks.v0.1"
TEMPLATE_SCHEMA = "phonoweave.alias_oto_review_template.v0.1"
PACK_SCHEMA = "phonoweave.recording_review_pack.v0.1"
REVIEW_SCHEMA = "phonoweave.reviewed_recording_scheme.v0.1"
DIRECTIVE_SCHEMA = "phonoweave.split_decision_directives.v0.1"
ALIAS_SCHEMA = "phonoweave.lossless_alias_record.v0.1"
FILES = (
    "scheme/alias_records.jsonl",
    "scheme/recording_action_plan.json",
    "directives/split_decision_directives.json",
)
# A deliberately limited *new* alias grammar; original aliases are unmodified.
_ALIAS = re.compile(r"[A-Za-z0-9_# -]{1,80}\Z")
COLUMNS = (
    "task_id", "unit_id", "context", "line_id", "pitch_group",
    "syllable_index", "syllable", "source_wav", "source_oto",
    "oto_line_number", "original_alias", "approved_new_alias", "intended_role",
    "annotation_status", "oto_offset_ms", "oto_consonant_ms",
    "oto_cutoff_ms", "oto_preutterance_ms", "oto_overlap_ms",
)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _canonical_sha(data: dict) -> str:
    return sha256(json.dumps(data, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_pack(pack: Path) -> tuple[dict, dict, dict, list[dict], str]:
    """Read only hashed pack artifacts; abort if files changed since compilation."""
    root = pack.expanduser().resolve(strict=True)
    manifest = _read_json(root / "review_pack_manifest.json")
    if manifest.get("schema") != PACK_SCHEMA:
        raise ValueError("unsupported recording review pack schema")
    digests = manifest.get("artifact_sha256", {})
    blobs = {}
    for rel in FILES:
        path = root / rel
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("review artifact escapes its pack")
        raw = path.read_bytes()
        if digests.get(rel) != sha256(raw).hexdigest():
            raise ValueError(f"review pack artifact checksum mismatch: {rel}")
        blobs[rel] = raw.decode("utf-8")
    records = [json.loads(s) for s in blobs[FILES[0]].splitlines() if s.strip()]
    action = json.loads(blobs[FILES[1]])
    directives = json.loads(blobs[FILES[2]])
    if directives.get("schema") != DIRECTIVE_SCHEMA:
        raise ValueError("unsupported split directive schema")
    fingerprint = _canonical_sha({rel: digests[rel] for rel in FILES})
    return manifest, action, directives, records, fingerprint


def _line_number(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("invalid OTO line number")
    try:
        number = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid OTO line number") from exc
    if number < 1 or str(value) not in {str(number), str(float(number))}:
        raise ValueError("invalid OTO line number")
    return number


def _source_index(records: list[dict]) -> tuple[dict, set[str]]:
    sources, taken = {}, set()
    for item in records:
        if item.get("schema") != ALIAS_SCHEMA:
            raise ValueError("unexpected lossless alias schema")
        source, role = item["source"], item["role"]
        keys = ("source_oto", "line_number", "original_alias")
        if any(source.get(k) != role.get(k) for k in keys):
            raise ValueError("source alias/role provenance mismatch")
        key = (source.get("source_oto"), _line_number(source.get("line_number")),
               source.get("original_alias"))
        if key in sources:
            raise ValueError("duplicate source OTO identity")
        sources[key] = item
        taken.add(source["original_alias"])
    return sources, taken


def build_review_template(action: dict, directives: dict,
                          records: list[dict], fingerprint: str) -> dict:
    """Create pending, source-verified internal VC naming review targets."""
    if action.get("schema") != "phonoweave.recording_scheme_action_plan.v0.1":
        raise ValueError("unsupported source action plan")
    if directives.get("schema") != DIRECTIVE_SCHEMA:
        raise ValueError("unsupported directive schema")
    sources, _ = _source_index(records)
    targets, seen = [], set()
    for row in directives["directives"]:
        if row.get("evidence_kind") != "source_linked_internal_VC_candidate":
            continue  # Initial lexical cues are NOT verified OTO entries.
        for candidate in row.get("candidate_realizations", []):
            if candidate.get("evidence_kind") != row["evidence_kind"]:
                raise ValueError("inconsistent directive evidence")
            key = (candidate.get("source_oto"), _line_number(candidate.get("oto_line_number")),
                   candidate.get("original_alias"))
            source = sources.get(key)
            if (source is None or source["source"].get("source_wav") != row.get("source_wav")
                    or source["role"].get("role") != "vc_transition"
                    or source["role"].get("pitch_suffix") != row.get("pitch_group")
                    or candidate.get("alias_mapping_approved") is not False
                    or candidate.get("unit_id") is None):
                raise ValueError("internal alias evidence does not match source OTO")
            target_key = (row["line_id"], row["pitch_group"],
                          row["syllable_index"], candidate["unit_id"], key)
            if target_key in seen:
                raise ValueError("duplicate line-specific alias target")
            seen.add(target_key)
            targets.append({
                "task_id": f"ALT-{len(targets)+1:04d}",
                "line_id": row["line_id"], "pitch_group": row["pitch_group"],
                "syllable_index": row["syllable_index"], "syllable": row["syllable"],
                "unit_id": candidate["unit_id"], "context": candidate["context"],
                "source_wav": row["source_wav"], "source_oto": key[0],
                "oto_line_number": key[1], "original_alias": key[2],
                "decision": "pending", "new_alias": "", "intended_role": "",
                "role_verified": False, "boundary_verified": False,
                "reviewer_note": "",
            })
    if len(targets) != directives.get("source_linked_internal_candidates"):
        raise ValueError("source-linked candidate count mismatch")
    return {
        "schema": TEMPLATE_SCHEMA,
        "speaker_id": action["speaker_id"],
        "source_pack_sha256": fingerprint,
        "source_action_plan_sha256": _hash_plan(action),
        "reviewer_id": "", "targets": targets,
        "target_count": len(targets),
        "scope": "manual_annotation_work_only",
    }


def _approved_internal_reviews(action: dict, reviewed: dict) -> set[tuple[str, str]]:
    if (reviewed.get("schema") != REVIEW_SCHEMA or
            reviewed.get("speaker_id") != action.get("speaker_id") or
            reviewed.get("source_plan_sha256") != _hash_plan(action) or
            reviewed.get("production_alias_export") != "disabled"):
        raise ValueError("reviewed scheme does not match frozen action plan")
    task_index = {(t["task_id"], pitch): t
                  for t in action["tasks"] for pitch in t["pitch_groups"]}
    if len(task_index) != action["source_review_rows"]:
        raise ValueError("action plan task count mismatch")
    resolved = reviewed.get("resolved_pitch_reviews", [])
    pending = reviewed.get("pending_pitch_reviews", [])
    if (not isinstance(resolved, list) or not isinstance(pending, list) or
            len(resolved) != reviewed.get("reviewed_pitch_cells") or
            len(pending) != reviewed.get("pending_pitch_cells")):
        raise ValueError("reviewed scheme counts do not match records")
    reviews = resolved + pending
    found, accepted = set(), set()
    for row in reviews:
        key = row.get("task_id"), row.get("pitch_group")
        task = task_index.get(key)
        if key in found or task is None:
            raise ValueError("unexpected/duplicate reviewed pitch task")
        found.add(key)
        if (row.get("resolution") not in RESOLUTIONS[task["review_action"]] or
                ((row in resolved) != (row.get("resolution") != "pending")) or
                row.get("unit_id") != task["unit_id"] or
                row.get("review_action") != task["review_action"] or
                row.get("base_unit") != task["base_unit"]):
            raise ValueError("reviewed task provenance mismatch")
        if (row.get("resolution") == "candidate_accepted" and
                row["review_action"] == "review_internal_candidate_mapping" and
                row.get("reviewer_note", "").strip() and
                row.get("evidence_reference", "").strip()):
            accepted.add((row["unit_id"], row["pitch_group"]))
    if found != set(task_index):
        raise ValueError("reviewed scheme omits pitch-specific task records")
    if accepted and not reviewed.get("reviewer_id", "").strip():
        raise ValueError("accepted reviews require reviewer identity")
    return accepted


def compile_oto_tasks(template: dict, action: dict, directives: dict,
                      records: list[dict], fingerprint: str, reviewed: dict) -> dict:
    expected = build_review_template(action, directives, records, fingerprint)
    if (template.get("schema") != TEMPLATE_SCHEMA or
            template.get("speaker_id") != expected["speaker_id"] or
            template.get("source_pack_sha256") != expected["source_pack_sha256"] or
            template.get("source_action_plan_sha256") != expected["source_action_plan_sha256"]):
        raise ValueError("stale or incompatible alias review template")
    targets = template.get("targets")
    if not isinstance(targets, list) or len(targets) != len(expected["targets"]):
        raise ValueError("alias review target count mismatch")
    accepted = _approved_internal_reviews(action, reviewed)
    _sources, taken = _source_index(records)
    approved, held, proposed_names = [], [], set()
    reviewer = template.get("reviewer_id", "")
    for actual, frozen in zip(targets, expected["targets"]):
        # Only explicitly editable review fields may differ from the plan.
        editable = {"decision", "new_alias", "intended_role", "role_verified",
                    "boundary_verified", "reviewer_note"}
        if {k: v for k, v in actual.items() if k not in editable} != {
            k: v for k, v in frozen.items() if k not in editable
        } or not editable <= actual.keys():
            raise ValueError("source task identity changed from frozen template")
        decision = actual.get("decision")
        if decision not in {"pending", "approve_annotation", "reject"}:
            raise ValueError("invalid alias task decision")
        task = {k: frozen[k] for k in frozen if k not in editable}
        if not isinstance(actual["new_alias"], str) or not isinstance(actual["intended_role"], str) or not isinstance(actual["reviewer_note"], str):
            raise ValueError("invalid alias review field type")
        task.update({"decision": decision, "new_alias": actual["new_alias"],
                     "intended_role": actual["intended_role"]})
        if decision != "approve_annotation" and (
                actual["new_alias"] or actual["intended_role"] or
                actual["role_verified"] or actual["boundary_verified"]):
            raise ValueError("unapproved alias task contains approval fields")
        if decision == "approve_annotation":
            alias = actual["new_alias"]
            if (not isinstance(alias, str) or not _ALIAS.fullmatch(alias) or
                    alias != alias.strip() or alias in taken or alias in proposed_names):
                raise ValueError("unsafe or colliding proposed alias")
            if (actual["intended_role"] != "internal_cv" or
                    actual["role_verified"] is not True or
                    actual["boundary_verified"] is not True or
                    not actual["reviewer_note"].strip() or
                    not isinstance(reviewer, str) or not reviewer.strip()):
                raise ValueError("approval requires explicit role/boundary verification and reviewer")
            if (task["unit_id"], task["pitch_group"]) not in accepted:
                raise ValueError("annotation approval lacks accepted source context review")
            proposed_names.add(alias)
            task.update({"annotation_status": "manual_oto_measurement_requested",
                         "oto_parameters_ms": None, "phonemizer_routing": None,
                         "reviewer_note": actual["reviewer_note"]})
            approved.append(task)
        else:
            if decision == "reject" and not actual["reviewer_note"].strip():
                raise ValueError("rejected annotation task requires rationale")
            task.update({"annotation_status": "pending_review" if decision == "pending" else "rejected",
                         "oto_parameters_ms": None, "phonemizer_routing": None})
            held.append(task)
    return {
        "schema": SCHEMA, "source_pack_sha256": fingerprint,
        "speaker_id": action["speaker_id"],
        "target_count": len(targets),
        "approved_manual_annotation_tasks": len(approved),
        "pending_or_rejected_tasks": len(held),
        "annotation_tasks": approved, "unapproved_targets": held,
        "oto_ini_export": "disabled", "phonemizer_export": "disabled",
        "audio_mutation": "none",
        "notes": [
            "An annotation approval orders manual review/measurement, not installation of a new alias.",
            "The referenced VC alias is not automatically a valid independent internal CV sample.",
            "All new OTO timing fields remain null until separately measured and verified.",
            "Tuner routing remains disabled until annotated samples and listening tests are approved.",
        ],
    }


def _csv_safe(value: object) -> str:
    value = "" if value is None else str(value)
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def write_compiled_tasks(report: dict, output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phonoweave-alias-oto-", dir=output.parent) as temp:
        stage = Path(temp)
        (stage / "alias_oto_annotation_tasks.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        with (stage / "ANNOTATOR_OTO_TASKS.csv").open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=COLUMNS)
            writer.writeheader()
            for row in report["annotation_tasks"]:
                record = {k: row.get(k, "") for k in COLUMNS}
                record["approved_new_alias"] = row["new_alias"]
                record["annotation_status"] = row["annotation_status"]
                writer.writerow({k: _csv_safe(v) for k, v in record.items()})
        tuner = [
            "# PhonoWeave — Pending alias routing", "",
            "Manual annotation approval is not phonemizer or playback approval.",
            "Only audition aliases after OTO segmentation and listening tests.",
            "No automatic fallback or source substitution is authorized.", "",
        ]
        for row in report["annotation_tasks"]:
            tuner.append(
                f"- {row['task_id']} ({row['pitch_group']} / {row['unit_id']}): "
                f"new name `{row['new_alias']}`; original VC "
                f"`{row['original_alias']}`; routing **pending**"
            )
        if not report["annotation_tasks"]:
            tuner.append("No annotation tasks have been approved.")
        (stage / "TUNER_PENDING_ROUTING.md").write_text(
            "\n".join(tuner) + "\n", encoding="utf-8")
        (stage / "README.txt").write_text(
            "REVIEW-ONLY alias naming and OTO measurement assignments.\n"
            "No original WAV/OTO modified. Empty task CSV is valid.\n",
            encoding="utf-8")
        if output.exists():
            raise FileExistsError("destination appeared during compilation")
        os.rename(stage, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile approval-gated OTO annotation work items")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("template", "compile"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--pack", type=Path, required=True)
        if name == "compile":
            cmd.add_argument("--reviews", type=Path, required=True)
            cmd.add_argument("--reviewed-scheme", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _manifest, action, directives, records, fingerprint = load_pack(args.pack)
        if args.command == "template":
            result = build_review_template(action, directives, records, fingerprint)
            with args.output.expanduser().open("x", encoding="utf-8") as out:
                json.dump(result, out, ensure_ascii=False, indent=2)
                out.write("\n")
            print("Alias/OTO review template:", args.output)
            print("Pending source-backed targets:", result["target_count"])
        else:
            template = _read_json(args.reviews.expanduser())
            reviewed = _read_json(args.reviewed_scheme.expanduser())
            report = compile_oto_tasks(template, action, directives, records, fingerprint, reviewed)
            location = write_compiled_tasks(report, args.output)
            print("Alias/OTO task pack:", location)
            print("Approved manual annotation tasks:", report["approved_manual_annotation_tasks"])
            print("Production alias/OTO export: disabled")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
