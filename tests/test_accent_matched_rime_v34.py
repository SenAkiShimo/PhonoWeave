"""v3.4 paired-rime auditing tests: never infer new phone identities."""
from __future__ import annotations

import json
from pathlib import Path
import wave

import numpy as np
import pytest

from phonoweave.accent_matched_rime_v34 import (
    SCHEMA, matched_pairs, export,
)
from phonoweave.accent_context_comparison_v33 import scan_voicebank


def _row(label, context, stem, source, *, bank="D4", alias=None, oto=1):
    value = np.zeros(36, dtype=float)
    value[0] = 0.5 if context == "rounded" else 0
    value[24] = 0.7 if context == "rounded" else 0
    return {
        "base_unit": label, "context": context, "stimulus": stem,
        "bank": bank, "source_pcm_sha256": source,
        "original_alias": alias or stem, "original_wav": source + ".wav",
        "oto_line": oto, "feature": value.tolist(),
    }


def test_only_exact_same_source_and_matching_rime():
    rows = [
        _row("sh", "plain", "sha", "same"),
        _row("sh", "rounded", "shua", "same", oto=2),
        _row("sh", "plain", "shai", "other", oto=3),
        _row("sh", "rounded", "shuai", "different", oto=4),
        _row("x", "plain", "xia", "other2"),
        _row("x", "rounded", "xue", "other2", oto=5),
    ]
    report = matched_pairs(rows)
    assert report["schema"] == SCHEMA
    assert report["matched_pair_count"] == 1
    case = report["pairs"][0]
    assert case["base_unit"] == "sh"
    assert (case["plain_final"], case["rounded_final"]) == ("a", "ua")
    assert case["same_source_pcm_sha256"] == "same"
    assert report["new_phonemes"] == []
    assert report["new_recording_lines"] == []
    assert report["base_summaries"][0]["recording_split_or_share"] == "UNRESOLVED"


def test_many_aliases_in_one_wav_do_not_multiply_independent_repeats():
    rows = []
    for index in range(20):
        rows.append(_row("sh", "plain", "sha", "source", oto=index + 1))
        rows.append(_row("sh", "rounded", "shua", "source", oto=index + 30))
    report = matched_pairs(rows)
    assert report["matched_pair_count"] == 1
    pair = report["pairs"][0]
    assert pair["plain_aliases_collapsed"] == 20
    assert pair["rounded_aliases_collapsed"] == 20
    assert report["base_summaries"][0]["distinct_source_filename_families"] == 1


def _write_wav(path:Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    hz = 48000
    t = np.arange(hz, dtype=np.float64) / hz
    pcm = (32767*.12*np.sin(2*np.pi*700*t)).astype("<i2")
    with wave.open(str(path),"wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(hz)
        w.writeframes(pcm.tobytes())


def test_actual_wav_oto_pair_and_preview_clips_are_read_only(tmp_path):
    root = tmp_path / "bank"
    subbank = root / "D4"
    subbank.mkdir(parents=True)
    wav = subbank / "a.wav"
    _write_wav(wav)
    original = wav.read_bytes()
    oto = subbank / "oto.ini"
    oto.write_text("a.wav=sha,200,120,100,130,80\n"
                   "a.wav=shua,600,120,100,130,80\n", encoding="utf-8")
    before = oto.read_bytes()
    rows, scan = scan_voicebank(root, only_bases=("sh",))
    assert scan["usable_candidate_regions"] == 2
    report = matched_pairs(rows)
    assert report["matched_pair_count"] == 1
    target = tmp_path / "audition"
    export(report, target, root)
    restored = json.loads((target / "matched_rime_audit.json").read_text())
    assert len(restored["listening_source_excerpts"]) == 1
    item = restored["listening_source_excerpts"][0]
    for key in ("a_path", "b_path"):
        audio = target / item[key]
        assert audio.exists()
        with wave.open(str(audio), "rb") as f:
            assert f.getnframes() > 0
    assert wav.read_bytes() == original
    assert oto.read_bytes() == before
    with pytest.raises(FileExistsError):
        export(report, target, root)
    with pytest.raises(ValueError, match="outside"):
        export(report, root / "inside", root)


def test_no_pair_on_mismatched_banks_or_source_pcm():
    rows = [
        _row("sh", "plain", "sha", "first", bank="D4"),
        _row("sh", "rounded", "shua", "first", bank="G4"),
        _row("sh", "rounded", "shua", "second", bank="D4"),
    ]
    report = matched_pairs(rows)
    assert report["matched_pair_count"] == 0


def test_invalid_features_are_rejected():
    a=_row("sh","plain","sha","a")
    a["feature"][0]=float("nan")
    with pytest.raises(ValueError,match="feature"):
        matched_pairs([a])
