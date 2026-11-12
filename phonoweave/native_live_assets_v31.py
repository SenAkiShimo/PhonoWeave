"""v3.1: read-only current-voicebank OTO -> WAV/PCM routing audit.

Shows original aliases without assuming their phonetic segmentation. An optional
author-supplied JSON template with bank_alias_prefixes permits strict parsing of
pre-existing unit labels: no IPA conversion, phoneme creation or recording edits.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import unicodedata
import wave
from typing import Any

from .oto import load_voicebank
from .native_recording_plan_v20 import read_bank_alias_prefixes
from .source_aware_recording_v17 import pcm_identity

SCHEMA = "phonoweave.live_voicebank_asset_routing.v3.1"
UNRESOLVED = "(original alias only)"


def _labels(value: list[str] | None) -> list[str]:
    if value is None:
        return []
    if (not isinstance(value, list) or len(value) > 5
            or any(not isinstance(x, str) or not x or x != x.strip()
                   or any(ord(c) < 32 for c in x) for x in value)):
        raise ValueError("use zero to five nonempty original author labels")
    normalized = [unicodedata.normalize("NFC", x) for x in value]
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate original labels")
    return normalized


def _split_alias(alias: str, bank: str, labels: list[str], prefixes: dict[str, str]
                 ) -> tuple[str, str, str]:
    """Never infer bank prefixes or split known labels inside longer strings."""
    label_text = unicodedata.normalize("NFC", alias)
    prefix = prefixes.get(bank, "")
    # If no prefix was author-declared, only parse unprefixed exact labels.
    if prefix:
        if not label_text.startswith(prefix):
            return UNRESOLVED, "", "unresolved_no_matching_declared_prefix"
        label_text = label_text[len(prefix):]
    for label in sorted(labels, key=lambda x: (-len(x), x)):
        if label_text == label:
            return label, "", ("author_declared_bank_prefix" if prefix else "exact_unprefixed_label")
        if label_text.startswith(label + " "):
            return label, label_text[len(label) + 1:], (
                "author_declared_bank_prefix" if prefix else "exact_unprefixed_label")
    return UNRESOLVED, "", "unresolved_original_alias"


def scan_live_voicebank(voicebank: Path, *,
                        labels: list[str] | None = None,
                        native_reclist: Path | None = None) -> dict[str, Any]:
    root = Path(voicebank).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("voicebank must be an existing directory")
    labels = _labels(labels)
    prefixes = read_bank_alias_prefixes(native_reclist) if native_reclist else {}
    for bank in prefixes:
        directory = (root if bank == "." else root / bank).resolve()
        if not directory.is_dir() or not directory.is_relative_to(root):
            raise ValueError("declared prefix subbank is absent or outside the voicebank: " + bank)
    entries, warnings = load_voicebank(root)

    # Only scan local regular source files; never follow a symlink out of bank.
    wav_paths: dict[Path, str] = {}
    for file in sorted(root.rglob("*")):
        if file.suffix.lower() != ".wav" or not file.is_file():
            continue
        resolved = file.resolve()
        if resolved.is_relative_to(root):
            wav_paths[resolved] = resolved.relative_to(root).as_posix()

    hashes: dict[Path, str | None] = {}
    def fingerprint(path: Path) -> str | None:
        if path not in hashes:
            try:
                hashes[path] = pcm_identity(path)
            except (OSError, EOFError, ValueError, wave.Error):
                hashes[path] = None
        return hashes[path]

    # Keep each author's raw alias. Without a declared mapping, grouping is
    # strictly by (bank, original alias), never by speculative phone names.
    grouped: dict[tuple[str, str], dict[Path, dict[str, Any]]] = defaultdict(dict)
    referenced: set[Path] = set()
    alias_meanings = {}
    for entry in entries:
        folder = entry.oto_path.parent.resolve()
        wav = entry.wav_path.resolve()
        if not folder.is_relative_to(root):
            continue
        bank = folder.relative_to(root).as_posix() or "."
        alias = unicodedata.normalize("NFC", entry.alias)
        if not alias:
            continue
        label, context, interpretation = _split_alias(alias, bank, labels, prefixes)
        alias_meanings[(bank, alias)] = (label, context, interpretation)
        if not wav.is_relative_to(root):
            status = "source_outside_voicebank_not_scanned"
            pcm = None
            path_label = "(outside voicebank: not read)"
        else:
            path_label = wav.relative_to(root).as_posix()
            referenced.add(wav)
            pcm = fingerprint(wav) if wav.is_file() else None
            if not wav.is_file():
                status = "missing_referenced_wav"
            elif pcm is None:
                status = "unreadable_referenced_wav"
            elif wav.parent != folder:
                status = "cross_subbank_wav_reference_review"
            else:
                status = "readable_source_pcm"
        current = grouped[(bank, alias)].get(wav)
        if current is None:
            grouped[(bank, alias)][wav] = {
                "source_wav": path_label, "source_pcm_sha256": pcm,
                "observed_original_aliases": [alias],
                "source_status": status,
                "oto_row_count": 1,
            }
        else:
            current["oto_row_count"] += 1

    # Also report unreferenced WAV files as configuration review, without
    # inventing an alias, unit identity or rerecording requirement.
    for wav, path_label in wav_paths.items():
        if wav in referenced:
            continue
        folder = wav.parent
        bank = folder.relative_to(root).as_posix() or "."
        unique_key = (bank, "\x00unreferenced:" + path_label)
        grouped[unique_key][wav] = {
            "source_wav": path_label,
            "source_pcm_sha256": fingerprint(wav),
            "observed_original_aliases": [],
            "source_status": "unreferenced_wav_needs_oto_review",
            "oto_row_count": 0,
        }

    routes = []
    assets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"wav_paths": set(), "aliases": set()})
    for (bank, alias), sources_by_path in sorted(grouped.items()):
        sources = sorted(sources_by_path.values(), key=lambda x: x["source_wav"])
        hashes_for_route = {x["source_pcm_sha256"] for x in sources
                            if x["source_pcm_sha256"] is not None}
        statuses = {x["source_status"] for x in sources}
        unresolved_file = any(x != "readable_source_pcm" for x in statuses)
        multiple = len(sources) > 1
        if alias.startswith("\x00unreferenced:"):
            label, context, interpretation = UNRESOLVED, "", "no_oto_alias"
            aliases = []
            status = "unreferenced_wav_needs_oto_review"
        else:
            label, context, interpretation = alias_meanings[(bank, alias)]
            aliases = [alias]
            if multiple:
                status = "multiple_observed_source_options_review"
            elif unresolved_file:
                status = next(iter(statuses))
            else:
                status = "single_current_oto_source"
        routes.append({
            "bank": bank, "original_unit_label": label,
            "following_context": context, "original_aliases": aliases,
            "alias_interpretation": interpretation,
            "observed_source_options": sources,
            "observed_distinct_pcm_assets": len(hashes_for_route),
            "routing_status": status,
            "unit_identity_must_be_preserved": True,
            "cross_context_audio_substitution_authorized": False,
        })
        for src in sources:
            pcm = src["source_pcm_sha256"]
            if pcm is not None:
                asset = assets[(bank, pcm)]
                asset["wav_paths"].add(src["source_wav"])
                asset["aliases"].update(aliases)

    bank_names = sorted({route["bank"] for route in routes})
    banks = []
    for bank in bank_names:
        bank_rows = [r for r in routes if r["bank"] == bank]
        banks.append({
            "bank": bank, "observed_label_context_routes": len(bank_rows),
            "distinct_existing_PCM_source_containers": sum(k[0] == bank for k in assets),
            "routes_with_multiple_source_options": sum(
                r["routing_status"] == "multiple_observed_source_options_review"
                for r in bank_rows),
            "unresolved_original_aliases": sum(
                r["original_unit_label"] == UNRESOLVED for r in bank_rows),
        })
    duplicate_assets = [{
        "bank": bank, "source_pcm_sha256": pcm,
        "source_wav_paths": sorted(a["wav_paths"])
    } for (bank, pcm), a in sorted(assets.items()) if len(a["wav_paths"]) > 1]
    return {
        "schema": SCHEMA,
        "scope": "current_voicebank_local_WAV_OTO_source_scan_not_phone_verification",
        "voicebank_root": str(root),
        "source_archive_sha256": None,
        "original_user_labels": labels,
        "declared_prefix_banks": sorted(prefixes),
        "oto_entries_read": len(entries),
        "oto_parse_warnings": len(warnings),
        "total_existing_alias_context_routes": len(routes),
        "distinct_existing_bank_PCM_source_containers": len(assets),
        "routes_requiring_source_choice_review": sum(
            r["routing_status"] != "single_current_oto_source"
            for r in routes),
        "multiple_source_alias_routes": sum(
            r["routing_status"] == "multiple_observed_source_options_review"
            for r in routes),
        "unreferenced_wav_routes": sum(
            r["routing_status"] == "unreferenced_wav_needs_oto_review"
            for r in routes),
        "unresolved_alias_routes": sum(r["original_unit_label"] == UNRESOLVED
                                       for r in routes),
        "per_bank": banks,
        "original_alias_routes": routes,
        "exact_pcm_duplicate_path_groups": duplicate_assets,
        "proposed_new_recording_lines": [],
        "approved_unit_substitutions": [],
        "approved_alias_merges": [],
        "approved_unit_splits": [],
        "source_sound_modified": False,
        "limitations": [
            "Live WAV/OTO files were scanned, but OTO boundaries and sound identity are not verified.",
            "Unknown author-native labels are preserved as raw original aliases, never guessed.",
            "Unregistered WAVs, missing files and cross-bank paths need review, not automatic recording.",
            "Identical PCM copies and multiple aliases in one WAV do not prove independent human takes.",
            "No new alias, phoneme, split/merge or recording task is generated.",
        ],
    }
