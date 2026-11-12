"""Read-only, source-provenance-focused GUI presentation for v2.9 asset ledger.

No conversion of source co-storage into phonetic substitutions or recordings.
"""
from __future__ import annotations

import csv
from io import StringIO
from typing import Any

CSV_COLUMNS = ("bank", "original_unit_label", "following_context",
               "routing_status", "source_wav", "source_pcm_sha256",
               "original_aliases", "source_status")

def _cell(value: Any) -> str:
    """Defuse spreadsheet formula injection when CSV is opened in a spreadsheet."""
    text = str(value)
    if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")):
        return "'" + text
    return text

def asset_routes_csv(report: dict[str, Any]) -> str:
    """Export one row per observed source option, never an invented recording."""
    if report.get("schema") not in ("phonoweave.native_asset_routing_ledger.v2.9",
                                    "phonoweave.live_voicebank_asset_routing.v3.1"):
        raise ValueError("expected validated v2.9 source ledger or v3.1 live voicebank scan")
    routes = report.get("original_alias_routes")
    if not isinstance(routes, list):
        raise ValueError("invalid asset ledger routes")
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for route in routes:
        if not isinstance(route, dict):
            raise ValueError("invalid source route")
        sources = route.get("observed_source_options")
        if not isinstance(sources, list):
            raise ValueError("invalid source options")
        for source in sources:
            values = (
                route["bank"], route["original_unit_label"],
                route["following_context"], route["routing_status"],
                source["source_wav"], source["source_pcm_sha256"],
                " | ".join(source["observed_original_aliases"]),
                source.get("source_status", "historical_source_link"),
            )
            writer.writerow([_cell(v) for v in values])
    return stream.getvalue()

def asset_ledger_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Lightweight metadata, with the full ledger available for filtering."""
    if report.get("schema") not in ("phonoweave.native_asset_routing_ledger.v2.9",
                                    "phonoweave.live_voicebank_asset_routing.v3.1"):
        raise ValueError("expected validated source archive or live voicebank scan")
    return {
        "schema": report["schema"],
        "source_archive_sha256": report["source_archive_sha256"],
        "original_user_labels": report["original_user_labels"],
        "source_scope": report["scope"],
        "total_existing_alias_context_routes": report["total_existing_alias_context_routes"],
        "distinct_existing_bank_PCM_source_containers": report["distinct_existing_bank_PCM_source_containers"],
        "routes_requiring_source_choice_review": report["routes_requiring_source_choice_review"],
        "per_bank": report["per_bank"],
        "original_alias_routes": report["original_alias_routes"],
        "historic_index_not_live_voicebank_validation": report["schema"] == "phonoweave.native_asset_routing_ledger.v2.9",
        "live_voicebank_scan": report["schema"] == "phonoweave.live_voicebank_asset_routing.v3.1",
        "unresolved_alias_routes": report.get("unresolved_alias_routes", 0),
        "unreferenced_wav_routes": report.get("unreferenced_wav_routes", 0),
        "oto_parse_warnings": report.get("oto_parse_warnings", 0),
        "no_new_recording_or_alias_changes": True,
    }
