"""Non-destructive speaker-adaptive recording-scheme review package.

This is an *output compiler* for existing analysis components, not a renderer
or a validated alias/OTO author. It never rewrites voicebank source files.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
import tempfile

from .alias_role_preview import classify_alias_role
from .diagnostic_selector import DiagnosticSelection, select_diagnostic_items
from .evidence_gap import build_evidence_completion_plan
from .inventory import VoicebankInventoryAnalysis, analyze_voicebank_inventory
from .mandarin import MandarinObservation, collect_observations, context_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .profile import build_speaker_profile
from .supplement_plan import build_supplement_plan
from .supplement_reclist import build_supplement_reclist
from .synthesis_inventory import (
    SynthesisInventory, build_synthesis_inventory, synthesis_inventory_yaml,
)

SCHEMA = "phonoweave.recording_scheme_review.v0.1"

PLAN_FIELDS = (
    "unit_id", "base_unit", "contexts", "decision", "confidence",
    "acoustic_evidence", "synthesis_evidence", "review_action",
)
ALIAS_FIELDS = (
    "source_oto", "line_number", "source_wav", "original_alias",
    "alias_shape", "base_unit", "final", "inferred_context", "candidate_unit_id",
    "mapping_status", "decision",
)
ROLE_FIELDS = (
    "source_oto", "line_number", "original_alias", "surface_alias",
    "pitch_suffix", "pitch_suffix_status", "role", "left_token",
    "right_token", "role_review_status",
)
OTO_FIELDS = (
    "source_oto", "line_number", "source_wav", "original_alias",
    "offset_ms", "consonant_ms", "cutoff_ms", "preutterance_ms", "overlap_ms",
    "review_action", "reason",
)
SUPPLEMENT_FIELDS = (
    "syllable", "base_unit", "final", "context_family", "role_scope",
    "replicate", "existing_observations", "recording_line",
)


def _rel(root: Path, target: Path) -> str:
    """Never publish unexpected external local paths in review artifacts."""
    try:
        return target.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return "[external_path]"


def _safe_csv(value: object) -> str:
    """Prevent spreadsheet formula injection without changing source WAVs."""
    # Signed numeric OTO fields must remain numeric in review CSVs.
    # Only untrusted text strings are formula-escaped.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = "" if value is None else str(value)
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _safe_csv(row.get(k)) for k in fields})


def _action(decision: str) -> str:
    return {
        "split_recommended": "review_separate_units",
        "merge_supported": "review_shared_unit",
        "unresolved": "collect_evidence_before_choice",
        "three_realizations_provisional": "review_three_realizations",
    }.get(decision, "manual_review")


def _alias_shape(alias: str, oto_path: Path) -> str:
    """Describe surface structure without interpreting an unknown alias."""
    stem = oto_path.parent.name.removesuffix(" EX")
    name = alias.strip()
    if stem and name.endswith(stem):
        name = name[:-len(stem)].strip()
    parts = name.split()
    if not parts:
        return "pitch_only_or_empty"
    if len(parts) >= 2:
        return "multi_token_transition_or_modifier"
    if parts[0].startswith("_"):
        return "underscore_prefixed"
    return "other_single_token"


def _observed_context(base: str, final: str) -> str | None:
    context = context_for(base, final)
    if base == "x" and context == "plain":
        return "front_unrounded"
    return context


@dataclass(frozen=True)
class ReviewPackage:
    summary: dict
    inventory: SynthesisInventory
    plan_rows: list[dict]
    alias_rows: list[dict]
    oto_rows: list[dict]
    role_rows: list[dict]
    supplement_rows: list[dict]
    supplement_text: str


def compile_review_package(
    inventory: SynthesisInventory,
    entries: list[OtoEntry],
    observations: list[MandarinObservation],
    selection: DiagnosticSelection,
    *,
    oto_warnings: int = 0,
) -> ReviewPackage:
    """Compile deterministic previews without guessing production aliases."""
    root = inventory.source_voicebank.expanduser().resolve()
    units = sorted(inventory.units, key=lambda x: (x.base_unit, x.id))
    by_base: dict[str, list] = defaultdict(list)
    for unit in units:
        by_base[unit.base_unit].append(unit)

    plan = [
        {
            "unit_id": u.id, "base_unit": u.base_unit,
            "contexts": "|".join(u.contexts), "decision": u.decision,
            "confidence": u.confidence, "acoustic_evidence": u.acoustic_evidence,
            "synthesis_evidence": u.synthesis_evidence,
            "review_action": _action(u.decision),
        }
        for u in units
    ]
    obs_by_origin: dict[tuple[str, int], MandarinObservation] = {
        (str(o.entry.oto_path.resolve()), o.entry.line_number): o
        for o in observations
    }
    aliases: list[dict] = []
    timings: list[dict] = []
    roles: list[dict] = []
    for entry in sorted(entries, key=lambda x: (str(x.oto_path), x.line_number)):
        origin = (str(entry.oto_path.resolve()), entry.line_number)
        obs = obs_by_origin.get(origin)
        base = obs.base_unit if obs else ""
        final = obs.final if obs else ""
        context = _observed_context(base, final) if obs else None
        candidates = [
            u for u in by_base.get(base, ())
            if context is not None and context in u.contexts
        ]
        unit = candidates[0] if len(candidates) == 1 else None
        if not obs:
            status = "unclassified_alias"
        elif not by_base.get(base):
            status = "outside_inventory"
        elif context is None:
            # Role-scoped and other context groups cannot safely be assigned
            # from an orthographic OTO alias alone.
            status = "context_not_inferable"
        elif len(candidates) != 1:
            status = "ambiguous_or_missing_unit"
        elif unit.decision == "unresolved":
            status = "unresolved_review_only"
        else:
            status = "candidate_review_only"
        if not entry.wav_path.is_file():
            status = "missing_source_wav"
        row = {
            "source_oto": _rel(root, entry.oto_path),
            "line_number": entry.line_number,
            "source_wav": _rel(root, entry.wav_path),
            "original_alias": entry.alias,
            "alias_shape": _alias_shape(entry.alias, entry.oto_path),
            "base_unit": base, "final": final,
            "inferred_context": context or "",
            "candidate_unit_id": unit.id if unit else "",
            "mapping_status": status,
            "decision": unit.decision if unit else "",
        }
        aliases.append(row)
        parsed_role = classify_alias_role(entry.alias, entry.oto_path)
        roles.append({
            "source_oto": row["source_oto"],
            "line_number": entry.line_number,
            "original_alias": entry.alias,
            "surface_alias": parsed_role.surface,
            "pitch_suffix": parsed_role.pitch_suffix,
            "pitch_suffix_status": parsed_role.pitch_suffix_status,
            "role": parsed_role.role,
            "left_token": parsed_role.left_token,
            "right_token": parsed_role.right_token,
            "role_review_status": parsed_role.review_status,
        })
        timings.append({
            "source_oto": row["source_oto"],
            "line_number": entry.line_number,
            "source_wav": row["source_wav"],
            "original_alias": entry.alias,
            "offset_ms": entry.offset, "consonant_ms": entry.consonant,
            "cutoff_ms": entry.cutoff, "preutterance_ms": entry.preutterance,
            "overlap_ms": entry.overlap,
            "review_action": (
                "inspect_missing_source" if status == "missing_source_wav"
                else "manual_boundary_review" if status == "unresolved_review_only"
                else "retain_no_automatic_change"
            ),
            "reason": status,
        })

    # An alias that maps to different neutral realization IDs is not a
    # collision-free mapping, even if each individual source WAV is valid.
    alias_to_ids: dict[str, set[str]] = defaultdict(set)
    for row in aliases:
        if row["candidate_unit_id"]:
            alias_to_ids[row["original_alias"]].add(row["candidate_unit_id"])
    collisions = {a for a, ids in alias_to_ids.items() if len(ids) > 1}
    for row, timing in zip(aliases, timings):
        if row["original_alias"] in collisions:
            row["mapping_status"] = "alias_collision_review"
            timing["review_action"] = "manual_alias_collision_review"
            timing["reason"] = "alias_collision_review"

    lines = build_supplement_reclist(selection)
    supplement = [
        {
            "syllable": item.syllable, "base_unit": item.base_unit,
            "final": item.final, "context_family": item.context_family,
            "role_scope": item.role_scope or "",
            "replicate": item.replicate,
            "existing_observations": item.existing_observations,
            "recording_line": line.text,
        }
        for item, line in zip(selection.items, lines.lines)
    ]
    summary = {
        "schema": SCHEMA,
        "speaker_id": inventory.speaker_id,
        "language": inventory.language,
        "units": len(units),
        "decisions": {d: sum(u.decision == d for u in units)
                      for d in sorted({u.decision for u in units})},
        "oto_entries": len(entries),
        "oto_warnings": oto_warnings,
        "alias_review_statuses": {
            s: sum(row["mapping_status"] == s for row in aliases)
            for s in sorted({row["mapping_status"] for row in aliases})
        },
        "distinct_colliding_aliases": len(collisions),
        "alias_role_counts": {
            role: sum(row["role"] == role for row in roles)
            for role in sorted({row["role"] for row in roles})
        },
        "alias_role_preview_status": "read_only_surface_grammar",
        "unclassified_alias_shapes": {
            shape: sum(row["mapping_status"] == "unclassified_alias"
                       and row["alias_shape"] == shape for row in aliases)
            for shape in sorted({row["alias_shape"] for row in aliases
                                 if row["mapping_status"] == "unclassified_alias"})
        },
        "suggested_diagnostic_lines": len(supplement),
        "unfilled_diagnostics": list(selection.unfilled),
        "production_alias_compilation": "not_implemented",
        "oto_timing_changes": "none",
        "status": "manual_review_required",
        "limitations": [
            "Alias-to-unit mapping is a candidate classification, not an OpenUtau alias installation.",
            "Role-scoped context cannot be inferred safely from spelling alone.",
            "Source OTO timing is copied for review; no invented replacement timings.",
            "Existing inventory decisions are provisional until synthesis/perceptual validation.",
        ],
    }
    return ReviewPackage(summary, inventory, plan, aliases, timings,
                         roles, supplement, lines.text())


def build_from_voicebank(root: Path) -> ReviewPackage:
    """Run existing analysis once, then compile its downstream output."""
    root = root.expanduser().resolve(strict=True)
    analysis: VoicebankInventoryAnalysis = analyze_voicebank_inventory(root)
    profile = build_speaker_profile(root, analysis)
    inventory = build_synthesis_inventory(root, profile)
    entries, warnings = load_voicebank(root)
    obs = collect_observations(entries, affix_pairs(load_prefix_maps(root)))
    gaps = build_evidence_completion_plan(analysis)
    selection = select_diagnostic_items(root, build_supplement_plan(gaps))
    return compile_review_package(
        inventory, entries, obs, selection, oto_warnings=len(warnings)
    )


def write_review_package(package: ReviewPackage, output: Path) -> Path:
    """Publish a complete review directory, never overwrite previous work."""
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phonoweave-review-",
                                     dir=output.parent) as staging_name:
        staging = Path(staging_name)
        (staging / "synthesis_inventory.yaml").write_text(
            synthesis_inventory_yaml(package.inventory), encoding="utf-8"
        )
        (staging / "summary.json").write_text(
            json.dumps(package.summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_csv(staging / "recording_units_review.csv",
                   PLAN_FIELDS, package.plan_rows)
        _write_csv(staging / "observed_alias_review.csv",
                   ALIAS_FIELDS, package.alias_rows)
        _write_csv(staging / "oto_timing_review.csv",
                   OTO_FIELDS, package.oto_rows)
        _write_csv(staging / "alias_roles_preview.csv",
                   ROLE_FIELDS, package.role_rows)
        _write_csv(staging / "supplemental_targets.csv",
                   SUPPLEMENT_FIELDS, package.supplement_rows)
        (staging / "supplemental_reclist.txt").write_text(
            package.supplement_text, encoding="utf-8"
        )
        (staging / "README.txt").write_text(
            "REVIEW ONLY - DO NOT IMPORT INTO A LIVE VOICEBANK\n"
            "recording_units_review.csv: neutral unit candidates and evidence\n"
            "observed_alias_review.csv: source aliases and tentative unit matches\n"
            "oto_timing_review.csv: unchanged OTO values; no new timing proposals\n"
            "supplemental_targets.csv / supplemental_reclist.txt: "
            "diagnostic recording proposals for unresolved evidence\n"
            "synthesis_inventory.yaml: existing neutral profile compiler output\n"
            "A collision or unresolved decision requires manual review.\n"
            "CSV cells beginning with formula prefixes are quoted for safety.\n",
            encoding="utf-8",
        )
        # Rename only after all files succeed; a competing destination is
        # treated as an error rather than intentionally overwritten.
        if output.exists():
            raise FileExistsError(f"output directory appeared: {output}")
        os.rename(staging, output)
    return output
