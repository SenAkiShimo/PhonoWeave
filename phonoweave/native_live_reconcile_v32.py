"""v3.2: reconcile a native recording plan with a fresh WAV/OTO source scan.

File-level consistency audit only. No inference of phonemes, missing human
performances, required new recording time, or safe unit substitution.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .native_live_assets_v31 import SCHEMA as LIVE_SCHEMA, scan_live_voicebank
from .native_recording_plan_v20 import (
    SCHEMA as NATIVE_SCHEMA,
    build_native_recording_workflow,
    read_bank_alias_prefixes,
    read_targets,
)

SCHEMA = "phonoweave.native_live_recording_reconciliation.v3.2"
HEALTHY = {"readable_source_pcm"}
FILE_GAPS = {"missing_referenced_wav"}
SOURCE_REVIEWS = {
    "unreadable_referenced_wav",
    "cross_subbank_wav_reference_review",
    "source_outside_voicebank_not_scanned",
    "unreferenced_wav_needs_oto_review",
}
PLAN_STATUSES = {
    "audio_present_registered", "missing_audio", "invalid_audio",
    "audio_present_needs_oto_review", "unverified_template_mapping",
    "ambiguous_file_names", "ambiguous_oto_alias",
}


def reconcile_native_live(
    plan: dict[str, Any],
    live: dict[str, Any],
    expected_reclist: Path,
) -> dict[str, Any]:
    """Cross-check exact author aliases and source PCM, never alter the plan.

    The two snapshots are read consecutively, not atomically. A mismatch can
    indicate a concurrent file edit, so the output only requests investigation.
    """
    if (not isinstance(plan, dict) or plan.get("schema") != NATIVE_SCHEMA or
            not isinstance(live, dict) or live.get("schema") != LIVE_SCHEMA):
        raise ValueError("requires a native recording plan and current live WAV/OTO scan")
    native_root = Path(plan["voicebank"]["root"]).resolve()
    if native_root != Path(live["voicebank_root"]).resolve():
        raise ValueError("refusing to compare different voicebank directories")

    targets = read_targets(expected_reclist)
    prefixes = read_bank_alias_prefixes(expected_reclist)
    expected = {row["line"]: row for row in targets}
    cases = plan["coverage"]["template_states"]
    if len(cases) != len(expected) or {c["line"] for c in cases} != set(expected):
        raise ValueError("recording plan and native template rows do not match")

    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    unreferenced: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for route in live["original_alias_routes"]:
        bank = route["bank"]
        for alias in route["original_aliases"]:
            index[(bank, alias)].append(route)
        if route["routing_status"] == "unreferenced_wav_needs_oto_review":
            for option in route["observed_source_options"]:
                unreferenced[(bank, option["source_wav"])].append(option)

    comparison = []
    states = Counter()
    for target_state in cases:
        line = target_state["line"]
        target = expected[line]
        for bank_row in target_state["banks"]:
            bank = bank_row["bank"]
            recorded_state = bank_row["status"]
            if recorded_state not in PLAN_STATUSES:
                raise ValueError("unknown native recording coverage status")
            aliases = [line]
            prefix = prefixes.get(bank)
            if prefix:
                aliases.append(prefix + line)
            related = []
            seen = set()
            for raw_alias in aliases:
                for route in index.get((bank, raw_alias), ()):
                    key = (route["bank"], tuple(route["original_aliases"]))
                    if key not in seen:
                        related.append(route)
                        seen.add(key)

            options = sorted(
                [dict(source, original_alias=route["original_aliases"][0])
                 for route in related for source in route["observed_source_options"]],
                key=lambda x: (x["original_alias"], x["source_wav"]),
            )
            healthy = [s for s in options if s["source_status"] in HEALTHY]
            # A readable path in another pitch subbank is NOT a verified
            # same-bank vocal take even if its digital PCM can be decoded.
            matching_verified = [
                s for s in healthy
                if (s.get("source_pcm_sha256") is not None
                    and s["source_pcm_sha256"] == bank_row.get("pcm_sha256"))
            ]
            expected_path = ((bank + "/" if bank != "." else "") + target["wav"])
            unregistered_wav = unreferenced.get((bank, expected_path), [])
            ambiguous_source = (
                len({s["source_wav"] for s in options}) > 1 or
                any(r["routing_status"] == "multiple_observed_source_options_review"
                    for r in related)
            )
            observed_file_gap = any(
                s["source_status"] in FILE_GAPS for s in options)
            observed_source_review = any(
                s["source_status"] in SOURCE_REVIEWS for s in options)
            if recorded_state == "audio_present_registered":
                if not matching_verified:
                    outcome, reason = (
                        "possible_inconsistency",
                        "planner_verified_audio_but_live_alias_pcm_not_confirmed",
                    )
                elif ambiguous_source:
                    outcome, reason = (
                        "source_choice_review",
                        "same_alias_has_multiple_source_WAV_options",
                    )
                else:
                    outcome, reason = (
                        "source_verified_file_level",
                        "registered_alias_and_PCM_agree",
                    )
            elif recorded_state == "missing_audio":
                if healthy:
                    outcome, reason = (
                        "possible_inconsistency",
                        "planner_missing_file_but_live_alias_has_readable_same_bank_source",
                    )
                elif observed_file_gap:
                    outcome, reason = (
                        "source_file_missing",
                        "explicit_registered_WAV_path_missing_in_both_scans",
                    )
                else:
                    outcome, reason = (
                        "not_proven",
                        "no_readable_matching_alias_source_in_current_scan",
                    )
            elif recorded_state == "unverified_template_mapping":
                if healthy:
                    outcome, reason = (
                        "possible_inconsistency",
                        "planner_unmapped_but_live_alias_has_readable_source",
                    )
                else:
                    outcome, reason = (
                        "source_mapping_review",
                        "long_WAV_or_alias_mapping_not_proven",
                    )
            elif recorded_state == "invalid_audio":
                outcome, reason = (
                    "source_repair_review",
                    "unreadable_source_needs_check_before_rerecording",
                )
            elif recorded_state == "audio_present_needs_oto_review":
                outcome, reason = (
                    "oto_or_source_review",
                    "existing_audio_needs_registration_or_cross_bank_check",
                )
            else:
                outcome, reason = (
                    "source_choice_review",
                    "multiple_alias_or_filename_source_options",
                )
            states[outcome] += 1
            comparison.append({
                "original_recording_line": line,
                "bank": bank,
                "expected_source_wav": target["wav"],
                "plan_status": recorded_state,
                "candidate_original_aliases": aliases,
                "live_observed_options": options,
                "unreferenced_expected_file_options": unregistered_wav,
                "live_multiple_sources": ambiguous_source,
                "live_has_explicit_missing_WAV": observed_file_gap,
                "live_has_source_status_to_review": observed_source_review,
                "result": outcome,
                "reason": reason,
                "automatic_new_recording_authorized": False,
            })

    compare_count = sum(states.values())
    if compare_count != plan["coverage"]["subbank_template_checks"]:
        raise ValueError("reconciliation does not cover every native template/bank check")
    problems = states["possible_inconsistency"]
    reviews = compare_count - states["source_verified_file_level"] - problems
    return {
        "schema": SCHEMA,
        "scope": "file_and_OTO_source_consistency_not_phonetic_completeness",
        "voicebank_root": str(native_root),
        "native_template_check_count": compare_count,
        "file_level_agreement_count": states["source_verified_file_level"],
        "possible_inconsistency_count": problems,
        "open_review_or_missing_source_count": reviews,
        "outcome_counts": dict(sorted(states.items())),
        "checks": comparison,
        "live_oto_entries_read": live["oto_entries_read"],
        "live_oto_parse_warnings": live["oto_parse_warnings"],
        "live_unreferenced_wav_routes": live["unreferenced_wav_routes"],
        "live_exact_PCM_duplicate_path_groups": len(
            live["exact_pcm_duplicate_path_groups"]),
        "result": ("possible_file_level_discrepancies_require_review" if problems else
                   "no_detected_file_level_discrepancy_but_reviews_remain" if reviews
                   else "no_detected_file_level_discrepancy"),
        "no_new_recording_lines_from_reconciliation": True,
        "automatic_unit_merges_or_splits": [],
        "original_voicebank_modified": False,
        "limitations": [
            "Planner and scanner read the same source sequentially, not atomically.",
            "Matching paths and PCM do not verify phoneme identity or OTO timing.",
            "Missing source WAV is not proof that the singer never performed the sound.",
            "Existing unreferenced WAV may hold content not proven by its filename.",
            "No automatic rerecording, new phonemes, audio substitution or unit edits.",
        ],
    }


def build_reconciled_native_plan(
    voicebank: Path, *, expected_reclist: Path,
    max_lines: int = 8, bank_names: tuple[str, ...] | None = None,
    existing_reclist: Path | None = None,
) -> dict[str, Any]:
    """One-call current voicebank preview, same file/bank/manifest scope."""
    report = build_native_recording_workflow(
        voicebank, expected_reclist=expected_reclist,
        max_lines=max_lines, bank_names=bank_names,
        existing_reclist=existing_reclist,
    )
    live = scan_live_voicebank(voicebank)
    consistency = reconcile_native_live(report, live, expected_reclist)
    return {**report, "source_consistency_review": consistency}
