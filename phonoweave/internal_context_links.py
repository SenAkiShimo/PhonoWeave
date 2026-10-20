"""Conservative internal-context links from verified reclist sequence order.

A V->C alias is not an internal-CV synthesis sample by itself. Only when a
recording filename's complete syllable sequence aligns with ordered boundary
aliases do we emit a *candidate* context link. No WAV is modified, no inferred
unit is authorized for synthesis, and no missing recording is claimed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import PurePosixPath
import re

from .mandarin import _split_syllable

SCHEMA = "phonoweave.internal_context_links.v0.1"
FIELDS = (
    "pitch_group", "unit_id", "base_unit", "context", "syllable",
    "source_oto", "line_number", "source_wav", "original_alias",
    "previous_syllable", "next_syllable", "boundary_order",
    "evidence", "review_status",
)
_BOUNDARY = frozenset({
    "initial_cv", "initial_vowel", "vc_transition",
    "vv_transition", "nonstandard_transition",
})
_VALID_WORD = re.compile(r"^[a-z]+$")


def _family(base: str, final: str) -> str | None:
    """Match only a context family explicitly exposed by the inventory."""
    if base in {"b", "p", "m", "d", "t"}:
        if final.startswith("i"):
            return "internal:i_series"
        if final.startswith("u"):
            return "internal:u_series"
        if final.startswith("v") or final == "ü":
            return None
        return "internal:other"
    if base in {"g", "k"}:
        return "internal:u_series" if final.startswith("u") else "internal:other"
    if base in {"f", "h"}:
        rounded = final.startswith("u") or final in {"o", "ou", "ong"}
        return "internal:rounded" if rounded else "internal:other"
    return None


def link_internal_contexts(units, aliases: list[dict], roles: list[dict],
                           oto_rows: list[dict], compiled_rows: list[dict]):
    """Return traceable review-only links, never executable unit mappings."""
    count = len(aliases)
    if len(roles) != count or len(oto_rows) != count or len(compiled_rows) != count:
        raise ValueError("source/role/timing/compiled row counts differ")
    index = defaultdict(list)
    for source, role, timing, compiled in zip(
            aliases, roles, oto_rows, compiled_rows):
        provenance = ("source_oto", "line_number", "original_alias")
        if any(source.get(key) != record.get(key)
               for record in (role, timing, compiled) for key in provenance):
            raise ValueError("source-alias role/timing provenance mismatch")
        wav = source.get("source_wav", "")
        if wav and wav != "[external_path]":
            index[(source.get("source_oto", ""), wav)].append(
                (source, role, timing, compiled)
            )

    unit_map = defaultdict(list)
    for unit in units:
        for context in unit.contexts:
            if context.startswith("internal:"):
                unit_map[(unit.base_unit, context)].append(unit.id)

    links, rejections = [], Counter()
    aligned_wavs = 0
    for (_oto, wav), records in sorted(index.items()):
        boundaries = [r for r in records if r[1].get("role") in _BOUNDARY]
        if len(boundaries) < 2:
            rejections["too_few_boundary_aliases"] += 1
            continue
        pitch = boundaries[0][1].get("pitch_suffix", "")
        stem = PurePosixPath(wav).stem
        if not (pitch and stem.startswith("_") and stem.endswith(pitch)):
            rejections["unsupported_reclist_filename"] += 1
            continue
        words = stem[:-len(pitch)].lstrip("_").split("_")
        if not words or any(not _VALID_WORD.fullmatch(w) for w in words):
            rejections["unsupported_reclist_tokens"] += 1
            continue
        if len(words) != len(boundaries):
            rejections["boundary_count_mismatch"] += 1
            continue
        if any(r[1].get("pitch_suffix_status") != "matched" or
               r[1].get("pitch_suffix") != pitch or
               r[3].get("compile_status") not in
               {"roundtrip_review_only", "unsupported_role"}
               for r in boundaries):
            rejections["unverified_pitch_or_alias"] += 1
            continue
        try:
            boundaries.sort(key=lambda r: float(r[2]["offset_ms"]))
        except (TypeError, ValueError, KeyError):
            rejections["non_numeric_oto_offset"] += 1
            continue
        offsets = [float(r[2]["offset_ms"]) for r in boundaries]
        if any(b <= a for a, b in zip(offsets, offsets[1:])):
            rejections["non_monotone_boundaries"] += 1
            continue
        first = boundaries[0][1]
        if (first.get("role") not in {"initial_cv", "initial_vowel"} or
                first.get("right_token") != words[0] or
                any(r[1].get("role") in {"initial_cv", "initial_vowel"}
                    for r in boundaries[1:])):
            rejections["sequence_anchor_mismatch"] += 1
            continue
        aligned_wavs += 1
        for position, (source, role, _timing, comp) in enumerate(boundaries[1:], 1):
            if role.get("role") != "vc_transition":
                continue
            if comp.get("compile_status") != "roundtrip_review_only":
                continue
            parsed = _split_syllable(words[position])
            if parsed is None:
                continue
            base, final = parsed
            if role.get("right_token") != base:
                continue
            left = role.get("left_token") or ""
            if not left or not words[position - 1].endswith(left):
                # Orthographic-to-phonetic variants need a declared
                # phonemizer dictionary, not a guessed equivalence.
                continue
            context = _family(base, final)
            candidates = unit_map.get((base, context), ())
            if context is None or len(candidates) != 1:
                continue
            links.append({
                "pitch_group": pitch, "unit_id": candidates[0],
                "base_unit": base, "context": context,
                "syllable": words[position],
                "source_oto": source["source_oto"],
                "line_number": source["line_number"],
                "source_wav": source["source_wav"],
                "original_alias": source["original_alias"],
                "previous_syllable": words[position - 1],
                "next_syllable": words[position],
                "boundary_order": position,
                "evidence": "ordered_reclist_filename_and_oto_boundary",
                "review_status": "candidate_link_not_synthesis_validated",
            })

    links.sort(key=lambda x: (x["pitch_group"], x["unit_id"],
                              x["source_wav"], x["boundary_order"]))
    return links, {
        "schema": SCHEMA,
        "source_wav_groups_inspected": len(index),
        "sequence_aligned_wav_groups": aligned_wavs,
        "candidate_links": len(links),
        "linked_internal_units": len({x["unit_id"] for x in links}),
        "linked_unit_pitch_cells": len({
            (x["unit_id"], x["pitch_group"]) for x in links
        }),
        "abstained_groups": dict(sorted(rejections.items())),
        "evidence_scope": "filename_and_oto_order_not_independent_acoustic_proof",
        "production_mapping": "disabled",
    }
