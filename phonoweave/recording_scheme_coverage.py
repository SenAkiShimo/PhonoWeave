"""Pitch-scoped recording-scheme coverage from traceable CV alias evidence.

This is an actionable review matrix of *alias references*, not a claim of
physically independent WAV takes or tested synthesis coverage. Role-scoped
inventory groups remain abstentions, never auto-labeled as absent recordings.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import re

SCHEMA = "phonoweave.recording_scheme_coverage.v0.1"
ROLES = ("initial_cv", "standalone_cv")
_PITCH = re.compile(r"^[A-Ga-g][#b]?-?\d+$")

COVERAGE_FIELDS = (
    "pitch_group", "unit_id", "base_unit", "contexts", "decision",
    "coverage_status", "initial_cv_aliases", "standalone_cv_aliases",
    "referenced_source_wavs", "internal_candidate_links",
    "internal_candidate_source_wavs", "internal_candidate_syllables",
    "initial_cv_syllables", "standalone_cv_syllables",
    "missing_initial_cv_vs_other_pitches", "missing_cv_vs_other_pitches",
    "recommended_review_action", "evidence_limit",
)

ACTION_FIELDS = (
    "pitch_group", "unit_id", "base_unit", "review_action",
    "suggested_syllables", "reason", "priority",
)


def build_recording_coverage(
    units, source_aliases: list[dict], compiled_aliases: list[dict],
    internal_links: list[dict] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Connect source CV alias roles with provisional inventory and pitch.

    The only usable direct links are explicitly candidate-mapped initial/CV
    source rows and lossless existing-alias reconstructions. There is no
    artificial fallback mapping for internal, unknown or empty observations.
    """
    if len(source_aliases) != len(compiled_aliases):
        raise ValueError("source/compiled alias row count mismatch")
    unit_ids = [u.id for u in units]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("duplicate neutral unit IDs")
    valid_ids = set(unit_ids)
    units_by_id = {u.id: u for u in units}
    internal_links = internal_links or []
    coverage: dict[tuple[str, str, str], dict[str, set]] = defaultdict(
        lambda: {"syllables": set(), "aliases": set(), "wavs": set()}
    )
    pitches = set()
    source_refs = {}
    for source, comp in zip(source_aliases, compiled_aliases):
        if any(source.get(k) != comp.get(k) for k in
               ("source_oto", "line_number", "original_alias")):
            raise ValueError("source/compiled alias provenance mismatch")
        provenance = (source.get("source_oto"), source.get("line_number"),
                      source.get("source_wav"), source.get("original_alias"))
        source_refs[provenance] = comp
        pitch = comp.get("pitch_suffix") or ""
        if pitch and _PITCH.fullmatch(pitch):
            pitches.add(pitch)
        unit_id = comp.get("candidate_unit_id", "")
        if not (unit_id in valid_ids and comp.get("compile_status") ==
                "roundtrip_review_only" and comp.get("role") in ROLES
                and source.get("mapping_status") in
                {"candidate_review_only", "unresolved_review_only"} and
                pitch in pitches):
            continue
        role = comp["role"]
        syllable = comp.get("right_token", "")
        if not syllable:
            continue
        group = coverage[(unit_id, pitch, role)]
        group["syllables"].add(syllable)
        group["aliases"].add(comp["original_alias"])
        if source.get("source_wav") and source["source_wav"] != "[external_path]":
            group["wavs"].add(source["source_wav"])

    pitches = sorted(pitches)
    # Link evidence was produced by an independent conservative sequence
    # aligner. Re-check its source identity and role before using it to
    # change the review queue. No synthesized mapping is emitted.
    internal_evidence = defaultdict(
        lambda: {"refs": set(), "wavs": set(), "syllables": set()}
    )
    for link in internal_links:
        uid = link.get("unit_id")
        pitch = link.get("pitch_group")
        origin = (link.get("source_oto"), link.get("line_number"),
                  link.get("source_wav"), link.get("original_alias"))
        comp = source_refs.get(origin)
        unit = units_by_id.get(uid)
        if not (unit is not None and pitch in pitches and
                link.get("context") in unit.contexts and
                any(context.startswith("internal:") for context in unit.contexts)
                and link.get("base_unit") == unit.base_unit and
                link.get("review_status") == "candidate_link_not_synthesis_validated"
                and comp is not None and comp.get("pitch_suffix") == pitch
                and comp.get("role") == "vc_transition"
                and comp.get("compile_status") == "roundtrip_review_only"):
            raise ValueError("internal candidate link provenance or context mismatch")
        group = internal_evidence[(uid, pitch)]
        group["refs"].add(origin)
        group["wavs"].add(link["source_wav"])
        group["syllables"].add(link["syllable"])
    rows, actions = [], []
    unit_with_any = {
        uid for (uid, _pitch, _role), group in coverage.items()
        if group["aliases"]
    }
    for unit in sorted(units, key=lambda x: (x.base_unit, x.id)):
        pooled = {
            role: set().union(*(coverage[(unit.id, p, role)]["syllables"]
                                for p in pitches))
            for role in ROLES
        }
        scoped = any(":" in context for context in unit.contexts)
        for pitch in pitches:
            internal = internal_evidence[(unit.id, pitch)]
            initial = coverage[(unit.id, pitch, "initial_cv")]
            standalone = coverage[(unit.id, pitch, "standalone_cv")]
            missing_initial = sorted(pooled["initial_cv"] - initial["syllables"])
            missing_cv = sorted(pooled["standalone_cv"] - standalone["syllables"])
            if scoped and internal["refs"]:
                status, action, priority = (
                    "internal_candidate_evidence",
                    "review_internal_candidate_mapping", 2,
                )
                reason = ("Sequence-aligned internal CV candidates exist; "
                          "manual role and phonetic verification required")
            elif scoped:
                status, action, priority = (
                    "mapping_scope_unresolved", "review_role_scoped_mapping", 2
                )
                reason = ("Inventory is role-scoped; no sequence-verified "
                          "internal CV candidate is linked")
            elif unit.id not in unit_with_any:
                status, action, priority = (
                    "not_traceably_mapped", "review_context_mapping", 2
                )
                reason = ("No evidence-linked CV alias for this unit; "
                          "absence of a recording is NOT established")
            elif missing_initial or missing_cv:
                status, action, priority = (
                    "cross_pitch_alias_gap", "review_cross_pitch_coverage", 1
                )
                reason = ("A CV syllable is linked to this unit in another "
                          "pitch group but not this one")
            elif not initial["aliases"] or not standalone["aliases"]:
                status, action, priority = (
                    "incomplete_role_coverage", "review_cv_role_coverage", 1
                )
                reason = "Only one CV role is traceably represented"
            else:
                status = "both_cv_roles_observed"
                if unit.decision == "unresolved":
                    action, priority = "review_provisional_unit_decision", 3
                    reason = "CV aliases exist, but the split/merge decision is unresolved"
                else:
                    action, priority = "retain_existing_aliases", 4
                    reason = "Both existing CV roles are traceably observed"

            references = initial["wavs"] | standalone["wavs"]
            row = {
                "pitch_group": pitch, "unit_id": unit.id,
                "base_unit": unit.base_unit, "contexts": "|".join(unit.contexts),
                "decision": unit.decision,
                "coverage_status": status,
                "initial_cv_aliases": len(initial["aliases"]),
                "standalone_cv_aliases": len(standalone["aliases"]),
                "referenced_source_wavs": len(references),
                "internal_candidate_links": len(internal["refs"]),
                "internal_candidate_source_wavs": len(internal["wavs"]),
                "internal_candidate_syllables": "|".join(
                    sorted(internal["syllables"])),
                "initial_cv_syllables": "|".join(sorted(initial["syllables"])),
                "standalone_cv_syllables": "|".join(sorted(standalone["syllables"])),
                "missing_initial_cv_vs_other_pitches": "|".join(missing_initial),
                "missing_cv_vs_other_pitches": "|".join(missing_cv),
                "recommended_review_action": action,
                "evidence_limit": "OTO_alias_references_only_not_recording_QC",
            }
            rows.append(row)
            if action != "retain_existing_aliases":
                examples = sorted(
                    internal["syllables"] if scoped and internal["refs"]
                    else set(missing_initial + missing_cv))
                actions.append({
                    "pitch_group": pitch, "unit_id": unit.id,
                    "base_unit": unit.base_unit,
                    "review_action": action,
                    "suggested_syllables": "|".join(examples),
                    "reason": reason,
                    "priority": priority,
                })
    actions.sort(key=lambda row: (row["priority"], row["pitch_group"],
                                  row["base_unit"], row["unit_id"]))
    return rows, actions, {
        "schema": SCHEMA,
        "pitch_groups": pitches,
        "unit_pitch_cells": len(rows),
        "units_with_traceable_cv_links": len(unit_with_any),
        "units_without_traceable_cv_links": len(valid_ids - unit_with_any),
        "internal_candidate_unit_pitch_cells": len({
            cell for cell, group in internal_evidence.items() if group["refs"]
        }),
        "internal_units_with_candidate_links": len({
            uid for (uid, _pitch), group in internal_evidence.items()
            if group["refs"]
        }),
        "internal_candidate_alias_links": sum(
            len(group["refs"]) for group in internal_evidence.values()),
        "coverage_status_counts": dict(sorted(
            Counter(row["coverage_status"] for row in rows).items())),
        "review_action_counts": dict(sorted(
            Counter(row["review_action"] for row in actions).items())),
        "actions": len(actions),
        "source_metric": "CV_alias_references_and_candidate_sequence_links_not_recording_QC",
        "recording_requests_auto_authorized": 0,
        "status": "manual_review_required",
    }
