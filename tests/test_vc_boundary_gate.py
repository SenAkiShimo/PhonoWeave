"""Synthetic regression suite: source integrity and transition review abstention."""
from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
import wave
import zipfile

import numpy as np
import pytest

from phonoweave.vc_boundary_gate import compile_boundary_review, SCHEMA


def _wav(freq: float) -> bytes:
    rate = 44100
    t = np.arange(rate * 2, dtype=float) / rate
    waveform = (0.12 * np.sin(2 * np.pi * freq * t)
                + 0.09 * np.sin(2 * np.pi * (freq + 1100) * t))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((waveform * 32767).astype('<i2').tobytes())
    return buf.getvalue()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    zip_path = tmp_path / "voicebank.zip"
    oto_lines = []
    for i, label in enumerate(("i", "ir", "i0")):
        # Same-WAV m, cross-WAV sh for ir.
        oto_lines.append(f"common.wav=D4{label} m,{600+i*250},120,10,115,80")
        fname = "different.wav" if label == "ir" else "common.wav"
        oto_lines.append(f"{fname}=D4{label} sh,{650+i*250},130,10,120,90")
    oto = ("\n".join(oto_lines) + "\n").encode('utf-8')
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("voicebank/oto.ini", oto)
        z.writestr("voicebank/common.wav", _wav(210))
        z.writestr("voicebank/different.wav", _wav(335))
    labels = {
        "schema": "phonoweave.author_confirmed_vc_nucleus_labels.v0.1",
        "source_oto_sha256": sha256(oto).hexdigest(),
        "alias_prefix": "D4", "review_scope": "nucleus_label_semantics_only",
        "labels": [{"alias_left_surface": label,
                    "nucleus_class_id": f"test_{label}", "author_confirmed": True}
                   for label in ("i", "ir", "i0")],
    }
    label_file = tmp_path / "labels.json"
    label_file.write_text(json.dumps(labels), encoding="utf-8")
    return zip_path, label_file


def test_real_audio_excerpts_and_feasibility_gate(tmp_path):
    src, labels = _fixture(tmp_path)
    out = tmp_path / "review"
    report = compile_boundary_review(src, labels, out, right_labels=("m", "sh"))
    assert report["schema"] == SCHEMA
    assert report["status"] == "auditory_boundary_review_required_no_split_share_evidence"
    assert len(report["review_groups"]) == 2
    assert report["review_groups"][0]["same_wav_for_three_nuclei"] is True
    assert report["review_groups"][1]["different_source_WAVs_confounded"] is True
    assert report["research_gate"]["can_claim_split_share_benefit"] is False
    assert "unreviewed" in (out / "review_annotations.csv").read_text()
    for group in report["review_groups"]:
        for record in group["clip_records"]:
            clip = out / record["excerpt_filename"]
            assert clip.exists()
            assert record["excerpt_sha256"] == sha256(clip.read_bytes()).hexdigest()
            with wave.open(str(clip), "rb") as w:
                assert w.getframerate() == 44100
                assert w.getnframes() == int(44100 * 0.6)
    assert len((out / "right_sh" / "timecourse.csv").read_text().splitlines()) == 52
    assert compile_boundary_review(src, labels, tmp_path / "second",
                                   right_labels=("m",))["review_groups"][0]["common_source_wav"] == "common.wav"
    with pytest.raises(FileExistsError, match="exists"):
        compile_boundary_review(src, labels, out)


def test_oto_hash_change_rejected(tmp_path):
    src, labels = _fixture(tmp_path)
    mapping = json.loads(labels.read_text())
    mapping["source_oto_sha256"] = "0"*64
    labels.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="hash"):
        compile_boundary_review(src, labels, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_missing_right_hand_family_fails(tmp_path):
    src, labels = _fixture(tmp_path)
    with pytest.raises(ValueError, match="not all confirmed"):
        compile_boundary_review(src, labels, tmp_path / "bad", right_labels=("q",))
    assert not (tmp_path / "bad").exists()


def test_input_alias_integrity_and_no_oto_rewriting(tmp_path):
    src, labels = _fixture(tmp_path)
    before = sha256(src.read_bytes()).hexdigest()
    report = compile_boundary_review(src, labels, tmp_path / "review")
    assert sha256(src.read_bytes()).hexdigest() == before
    assert report["production_alias_export"] == "disabled"
    assert report["oto_timing_changes"] == "none"
    lineids = [x["oto_line"] for group in report["review_groups"] for x in group["clip_records"]]
    assert len(lineids) == 6 and len(set(lineids)) == 6
