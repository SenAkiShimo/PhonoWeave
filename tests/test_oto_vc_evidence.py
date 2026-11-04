"""Read-only ZIP/OTO-to-phonetic-graph interface tests with synthetic speech."""
from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import wave
import zipfile

import numpy as np
import pytest

from phonoweave.oto_vc_evidence import (
    SCHEMA, LINK_SCHEMA, REVIEW_SCHEMA, analyze_archived_voicebank,
    map_reviewed_graph_edges, write_audit,
)
from phonoweave.phonetic_transition_graph import compile_transition_budget, LEXICON_SCHEMA
from phonoweave.reclist import parse_reclist


def _wav(duration_s=1.0):
    sr = 44100
    t = np.arange(round(sr * duration_s), dtype=float) / sr
    sample = (0.15 * np.sin(2 * np.pi * 420 * t)
              + 0.08 * np.sin(2 * np.pi * 3300 * t))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sr)
        stream.writeframes((sample * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def _archive(tmp_path):
    path = tmp_path / "voicebank.zip"
    records = [
        "source.wav=D4i sh,100,100,300,90,60",
        "source.wav=D4ir sh,300,100,150,80,52",
        "source.wav=D4i0 sh,500,90,100,40,35",
        "source.wav=D4- shi,700,100,90,70,40",
        "source.wav=D4shi,750,80,20,60,40",
        "source.wav=D4i sh,120,80,250,88,62",
    ]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Singer/source.wav", _wav())
        z.writestr("Singer/oto.ini", "\r\n".join(records) + "\r\n")
    return path


def test_zip_report_and_duplicate_alias_audit(tmp_path):
    archive = _archive(tmp_path)
    report = analyze_archived_voicebank(archive, prefix="D4")
    assert report["schema"] == SCHEMA
    summary = report["summary"]
    assert summary["oto_records_total"] == 6
    assert summary["vc_shape_records"] == 4
    assert summary["vc_shape_distinct_aliases"] == 3
    assert summary["duplicate_alias_names"] == 1
    assert summary["extra_records_on_duplicate_aliases"] == 1
    assert summary["spectral_status"]["provisional_oto_relative_only"] == 3
    assert summary["spectral_status"]["too_short_for_fixed_16ms_proxy"] == 1
    assert all(row["alias"].startswith("D4") for row in report["candidates"])
    assert all(row["wav"] == "source.wav" for row in report["candidates"])
    assert report["production_alias_export"] == "disabled"
    assert analyze_archived_voicebank(archive, prefix="D4") == report


def test_review_queue_write_non_destructive(tmp_path):
    archive = _archive(tmp_path)
    report = analyze_archived_voicebank(archive, prefix="D4")
    path = tmp_path / "new_review"
    write_audit(report, path)
    assert (path / "vc_oto_evidence.json").is_file()
    assert (path / "vc_review_queue.csv").is_file()
    template = (path / "identity_mapping_template.csv").read_text(encoding="utf-8")
    assert "D4ir sh" in template
    assert "false,false" in template
    with pytest.raises(FileExistsError):
        write_audit(report, path)


def test_graph_link_review_is_required(tmp_path):
    report = analyze_archived_voicebank(_archive(tmp_path), prefix="D4")
    lexicon = {
        "schema": LEXICON_SCHEMA,
        "language": "synthetic-phonetic-review",
        "entries": {
            "shi": [
                {"symbol": "ʂ", "role": "C", "grapheme": "sh"},
                {"symbol": "ɻ̩", "role": "V", "grapheme": "i"},
            ],
            "ma": [
                {"symbol": "m", "role": "C", "grapheme": "m"},
                {"symbol": "a", "role": "V", "grapheme": "a"},
            ],
        },
    }
    graph = compile_transition_budget(parse_reclist("shi ma\n"), lexicon, 1)
    review = {
        "schema": REVIEW_SCHEMA,
        "links": [{
            "alias": "D4ir sh",
            "left_surface": "ir", "right_surface": "sh",
            "phase": "across_syllables", "pattern": "VC",
            "left_phone": "ɻ̩", "right_phone": "ʂ",
            "identity_reviewed": False,
        }]
    }
    # Original graph has only the transition [ɻ̩]→[m], not [ɻ̩]→[ʂ].
    with pytest.raises(ValueError, match="identity-reviewed"):
        map_reviewed_graph_edges(report, graph, review)
    review["links"][0]["identity_reviewed"] = True
    with pytest.raises(ValueError, match="nonexisting graph"):
        map_reviewed_graph_edges(report, graph, review)

    lexicon["entries"]["sha"] = [
        {"symbol": "ʂ", "role": "C", "grapheme": "sh"},
        {"symbol": "a", "role": "V", "grapheme": "a"},
    ]
    graph = compile_transition_budget(parse_reclist("shi sha\n"), lexicon, 2)
    linked = map_reviewed_graph_edges(report, graph, review)
    assert linked["schema"] == LINK_SCHEMA
    assert len(linked["matched_links"]) == 1
    assert linked["matched_links"][0]["source_oto_lines"] == [2]
    assert not linked["production_ready"]


def test_bad_archive_fails_closed(tmp_path):
    invalid = tmp_path / "invalid.zip"
    invalid.write_bytes(b"not zip")
    with pytest.raises(ValueError, match="not a ZIP"):
        analyze_archived_voicebank(invalid, prefix="D4")
    with pytest.raises(ValueError, match="prefix"):
        analyze_archived_voicebank(_archive(tmp_path), prefix=" ")
