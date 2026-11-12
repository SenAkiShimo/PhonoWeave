"""v2.9: conservative alias-to-physical-WAV routing ledger for existing units.

This is a provenance inventory, NOT cross-context sound substitution, phoneme
discovery, or a claim of minimized actual recording time. No audio/OTO edits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import json
from typing import Any

from .native_unit_budget_v25 import _read_archive

SCHEMA = "phonoweave.native_asset_routing_ledger.v2.9"


def build_asset_routing_ledger(
    rows: list[dict[str, Any]], original_labels: list[str], *,
    archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Index *observed* OTO aliases by existing PCM source and original label.

    A long WAV can contain many DIFFERENT performed/recorded phonetic units.
    Co-storage reduces redundant file referencing; it is never evidence that
    different labels, contexts, or temporal segments are interchangeable.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("source-linked rows required")
    if (not isinstance(original_labels, list)
            or not 2 <= len(original_labels) <= 5
            or any(not isinstance(x, str) or not x.strip() or x != x.strip()
                   for x in original_labels)
            or len(set(original_labels)) != len(original_labels)):
        raise ValueError("provide 2–5 distinct original author-defined labels")
    if archive_sha256 is not None and (
        not isinstance(archive_sha256, str) or len(archive_sha256) != 64
        or any(c not in "0123456789abcdef" for c in archive_sha256)
    ):
        raise ValueError("invalid source archive SHA-256")
    labels = set(original_labels)
    route_sources: dict[tuple, dict[tuple, set[str]]] = defaultdict(
        lambda: defaultdict(set))
    physical_assets: dict[tuple, dict[str, set]] = defaultdict(
        lambda: {"wav_paths": set(), "alias_references": set(), "unit_contexts": set()})
    co_storage: dict[tuple, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set))
    wav_fingerprints: dict[tuple, str] = {}
    present_labels: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid source-index row")
        if row.get("symbol") not in labels:
            continue
        mandatory = ("bank", "symbol", "context", "source_wav",
                     "source_pcm_sha256", "source_alias")
        if any(not isinstance(row.get(k), str) or not row[k] for k in mandatory):
            raise ValueError("missing source WAV / original alias / context identity")
        bank, label, context = row["bank"], row["symbol"], row["context"]
        wav, pcm, alias = row["source_wav"], row["source_pcm_sha256"], row["source_alias"]
        if len(pcm) != 64 or any(c not in "0123456789abcdef" for c in pcm):
            raise ValueError("invalid decoded PCM source SHA-256")
        prior = wav_fingerprints.setdefault((bank, wav), pcm)
        if prior != pcm:
            raise ValueError("same bank/WAV path has incompatible PCM source hashes")
        present_labels.add(label)
        asset_key = (bank, pcm)
        source_key = (pcm, wav)
        route_sources[(bank, label, context)][source_key].add(alias)
        asset = physical_assets[asset_key]
        asset["wav_paths"].add(wav)
        asset["alias_references"].add((label, context, alias))
        asset["unit_contexts"].add((label, context))
        co_storage[(bank, context, pcm)][label].add((wav, alias))
    missing_labels = [label for label in original_labels if label not in present_labels]
    if missing_labels:
        raise ValueError("some original labels do not occur in source-index evidence")

    routes = []
    for (bank, label, context), sources in sorted(route_sources.items()):
        aliases = sorted({alias for source_aliases in sources.values() for alias in source_aliases})
        source_refs = [{
            "source_wav": wav, "source_pcm_sha256": pcm,
            "observed_original_aliases": sorted(source_aliases),
        } for (pcm, wav), source_aliases in sorted(sources.items())]
        distinct_pcm = {key[0] for key in sources}
        ambiguous = len(distinct_pcm) != 1 or len(aliases) != 1
        routes.append({
            "bank": bank, "original_unit_label": label,
            "following_context": context,
            "original_aliases": aliases,
            "observed_source_options": source_refs,
            "observed_distinct_pcm_assets": len(distinct_pcm),
            "routing_status": (
                "multiple_observed_source_options_review"
                if ambiguous else "single_observed_source_in_filtered_index"
            ),
            "unit_identity_must_be_preserved": True,
            "cross_context_audio_substitution_authorized": False,
        })
    route_by_key = {
        (route["bank"], route["original_unit_label"], route["following_context"]):
        route for route in routes
    }
    assets = []
    for (bank, pcm), asset in sorted(physical_assets.items()):
        paths = sorted(asset["wav_paths"])
        assets.append({
            "bank": bank,
            "source_pcm_sha256": pcm,
            "observed_source_wav_paths": paths,
            "possible_exact_PCM_duplicate_paths": len(paths) > 1,
            "distinct_label_context_routes": len(asset["unit_contexts"]),
            "original_labels_in_source": [
                label for label in original_labels
                if any(item[0] == label for item in asset["unit_contexts"])
            ],
            "following_contexts": sorted({ctx for _, ctx in asset["unit_contexts"]}),
            "original_alias_reference_count": len(asset["alias_references"]),
            "reuse_scope": "existing_wav_container_only",
        })
    co_recorded = []
    for (bank, context, pcm), by_label in sorted(co_storage.items()):
        if not all(label in by_label for label in original_labels):
            continue
        wavs = {wav for recordings in by_label.values() for wav, _ in recordings}
        # All labels must occur on the same named WAV within this same PCM.
        common = set.intersection(*(
            {wav for wav, _ in by_label[label]} for label in original_labels
        ))
        if not common:
            continue
        for wav in sorted(common):
            co_recorded.append({
                "bank": bank, "following_context": context,
                "source_wav": wav, "source_pcm_sha256": pcm,
                "original_aliases_by_label": {
                    label: sorted(alias for path, alias in by_label[label] if path == wav)
                    for label in original_labels
                },
                "any_label_has_multiple_source_options": any(
                    route_by_key[(bank, label, context)]["routing_status"]
                    != "single_observed_source_in_filtered_index"
                    for label in original_labels
                ),
                "co_container_not_same_phonetic_unit": True,
                "safe_to_merge_aliases_or_segments": False,
            })
    bank_summaries = []
    all_banks = sorted({route["bank"] for route in routes})
    for bank in all_banks:
        subset_routes = [route for route in routes if route["bank"] == bank]
        subset_assets = [asset for asset in assets if asset["bank"] == bank]
        bank_summaries.append({
            "bank": bank,
            "observed_label_context_routes": len(subset_routes),
            "distinct_existing_PCM_source_containers": len(subset_assets),
            "routes_with_multiple_source_options": sum(
                route["routing_status"] == "multiple_observed_source_options_review"
                for route in subset_routes
            ),
            "observed_same_container_all_label_contexts": sum(
                row["bank"] == bank for row in co_recorded
            ),
        })
    return {
        "schema": SCHEMA,
        "original_user_labels": original_labels[:],
        "source_archive_sha256": archive_sha256,
        "input_source_index_windows": len(rows),
        "scope": "existing_OTO_relative_filtered_source_index_only",
        "per_bank": bank_summaries,
        "total_existing_alias_context_routes": len(routes),
        "distinct_existing_bank_PCM_source_containers": len(assets),
        "routes_requiring_source_choice_review": sum(
            route["routing_status"] == "multiple_observed_source_options_review"
            for route in routes
        ),
        "same_container_all_original_labels_context_groups": len(co_recorded),
        "original_alias_routes": routes,
        "existing_source_containers": assets,
        "same_container_cross_label_examples": co_recorded,
        "proposed_new_recording_lines": [],
        "approved_unit_substitutions": [],
        "approved_alias_merges": [],
        "approved_unit_splits": [],
        "source_sound_modified": False,
        "limitations": [
            "Source links are from filtered OTO-derived historical records, not fresh full-WAV verification.",
            "An original long WAV can contain multiple distinct sung segments or takes.",
            "Co-storage of i/ir/i0 must NOT be interpreted as shared unit acoustics.",
            "Different context aliases stored in the same WAV are not validated interchangeable donors.",
            "Numbers of WAV containers do not measure performance count or minutes of recording saved.",
            "No new phonemes, OTO edits, alternate donor selection or new recording tasks are authorized.",
        ],
    }


def run_archive(archive: Path, labels: list[str]) -> dict[str, Any]:
    rows, digest = _read_archive(archive)
    return build_asset_routing_ledger(rows, labels, archive_sha256=digest)


def attach_asset_routing_ledger(
    recording_report: dict[str, Any], archive: Path, labels: list[str],
) -> dict[str, Any]:
    """Add a separate conservative ledger, preserving every recording decision."""
    if not isinstance(recording_report, dict) or not isinstance(
        recording_report.get("recording_plan"), dict
    ):
        raise ValueError("existing recording plan required")
    if "existing_source_asset_routing" in recording_report:
        raise ValueError("asset routing already attached; refusing overwrite")
    author = recording_report.get("native_author_feedback_review")
    if isinstance(author, dict):
        confirmed = author.get("author_confirmation", {})
        if (confirmed.get("original_alias_mapping") != "yes"
                or confirmed.get("desire_to_keep_existing_realization_labels_distinct") != "yes"
                or labels != author.get("original_user_labels")):
            raise ValueError("asset routing labels require matching author confirmation")
    review = recording_report.get("existing_sound_unit_review")
    if isinstance(review, dict) and set(labels) != set(review.get("original_user_labels", [])):
        raise ValueError("asset routing labels disagree with prior user unit review")
    return {
        **recording_report,
        "existing_source_asset_routing": run_archive(archive, labels)
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Read-only original alias to existing physical WAV mapping"
    )
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--label", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    out = args.output.expanduser().resolve()
    if out.exists() or not out.parent.is_dir():
        p.error("output must be in an existing directory and not already exist")
    report = run_archive(args.source_archive, args.label)
    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({
        "existing_source_containers": report["distinct_existing_bank_PCM_source_containers"],
        "alias_context_routes": report["total_existing_alias_context_routes"],
        "source_options_to_review": report["routes_requiring_source_choice_review"],
        "automatic_unit_changes": 0,
        "new_recordings": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
