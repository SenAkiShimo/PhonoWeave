"""Lossless machine-readable source alias records for downstream compilers.

CSV review tables are spreadsheet-safe presentation artifacts. This JSONL
stream preserves source strings and numeric OTO values byte-for-byte at
the Unicode string level. The source voicebank is never modified.
"""
from __future__ import annotations

import json
from pathlib import Path


def write_alias_records(path: Path, aliases: list[dict], roles: list[dict],
                        compiled: list[dict], timings: list[dict]) -> int:
    streams = (aliases, roles, compiled, timings)
    if len(set(map(len, streams))) != 1:
        raise ValueError("alias records are not aligned")
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for source, role, compilation, timing in zip(*streams):
            identity = ("source_oto", "line_number", "original_alias")
            if any(
                row.get(key) != source.get(key)
                for row in (role, compilation, timing)
                for key in identity
            ):
                raise ValueError("alias source/role/compilation/timing mismatch")
            if source.get("source_wav") != timing.get("source_wav"):
                raise ValueError("alias WAV provenance mismatch")
            record = {
                "schema": "phonoweave.lossless_alias_record.v0.1",
                "source": source,
                "role": role,
                "compilation_preview": compilation,
                "original_oto_timing": timing,
            }
            handle.write(json.dumps(record, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")) + "\n")
            count += 1
    return count
