"""Strict, collision-checked preview of an existing UTAU/OpenUtau alias scheme.

No new aliases are inferred. Successful output is a lossless rendering of
already observed aliases and their pitch-subbank structure, never an oto.ini
writer or a claim that a synthesis unit can be safely merged or removed.
"""
from __future__ import annotations

from collections import Counter, defaultdict

SCHEMA = "phonoweave.alias_compilation_preview.v0.1"
ROLE_ALLOWLIST = frozenset({
    "initial_cv", "initial_vowel", "standalone_cv", "standalone_vowel",
    "prefixed_vowel", "vc_transition", "vv_transition",
})

FIELDS = (
    "source_oto", "line_number", "original_alias", "role",
    "pitch_suffix", "left_token", "right_token",
    "surface_alias", "rendered_alias", "compile_status",
    "candidate_unit_id", "unit_mapping_status",
)


def _render(role: str, left: str, right: str) -> str:
    if role in {"initial_cv", "initial_vowel"} and left == "-":
        return "- " + right
    if role in {"standalone_cv", "standalone_vowel"} and not left:
        return right
    if role == "prefixed_vowel" and not left:
        return "_" + right
    if role in {"vc_transition", "vv_transition"} and left and right:
        return left + " " + right
    return ""


def compile_alias_preview(
    aliases: list[dict], roles: list[dict],
) -> tuple[list[dict], dict]:
    """Produce only round-trip-verified existing aliases.

    Matching (source OTO, source line, original alias) ties role parsing to
    immutable OTO provenance. Identical generated aliases, even from separate
    subbanks, are never silently deduplicated or declared import-ready.
    """
    if len(aliases) != len(roles):
        raise ValueError("alias and role row counts differ")
    result = []
    for source, role in zip(aliases, roles):
        identity = ("source_oto", "line_number", "original_alias")
        if any(source.get(k) != role.get(k) for k in identity):
            raise ValueError("source-alias/role provenance does not match")
        status = "roundtrip_review_only"
        rendered = ""
        if role.get("role") not in ROLE_ALLOWLIST:
            status = "unsupported_role"
        elif role.get("pitch_suffix_status") != "matched" or not role.get("pitch_suffix"):
            status = "unverified_pitch_suffix"
        else:
            surface = _render(
                role["role"], role.get("left_token", ""),
                role.get("right_token", ""),
            )
            if not surface or surface != role.get("surface_alias"):
                status = "surface_roundtrip_mismatch"
            else:
                rendered = surface + role["pitch_suffix"]
                if rendered != role["original_alias"]:
                    status = "alias_roundtrip_mismatch"
                    rendered = ""
        candidate = source.get("candidate_unit_id") or ""
        mapping_status = source.get("mapping_status", "")
        # Internal VC/VV roles do not themselves identify an inventory CV
        # realization. Existing candidate links in those roles are withheld.
        if role.get("role") not in {"standalone_cv", "initial_cv"}:
            candidate = ""
            if mapping_status in {"candidate_review_only", "unresolved_review_only"}:
                mapping_status = "not_cv_unit_link"
        result.append({
            "source_oto": role["source_oto"],
            "line_number": role["line_number"],
            "original_alias": role["original_alias"],
            "role": role["role"],
            "pitch_suffix": role.get("pitch_suffix", ""),
            "left_token": role.get("left_token", ""),
            "right_token": role.get("right_token", ""),
            "surface_alias": role.get("surface_alias", ""),
            "rendered_alias": rendered,
            "compile_status": status,
            "candidate_unit_id": candidate,
            "unit_mapping_status": mapping_status,
        })

    # An alias is a singer-visible namespace key: collisions across source
    # oto.ini files are possible even when per-file alias spelling is valid.
    targets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(result):
        if row["compile_status"] == "roundtrip_review_only":
            targets[row["rendered_alias"]].append(index)
    collisions = {alias: indices for alias, indices in targets.items()
                  if len(indices) > 1}
    for indices in collisions.values():
        for index in indices:
            result[index]["compile_status"] = "global_alias_collision"

    counts = dict(sorted(Counter(x["compile_status"] for x in result).items()))
    summary = {
        "schema": SCHEMA,
        "rows": len(result),
        "status_counts": counts,
        "globally_colliding_aliases": len(collisions),
        "collision_affected_rows": sum(map(len, collisions.values())),
        "roundtrip_verified_rows": counts.get("roundtrip_review_only", 0),
        "scope": "existing_alias_reconstruction_only",
        "production_export": "disabled",
        "oto_timing_changes": "none",
    }
    return result, summary
