"""S01-only executable renderer tests using synthetic WAVs, not singing data."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from phonoweave.synthesis_relevance_controlled_corpus import (
    FINALS, PAIRS, REPEATS, make_plan,
)
from phonoweave.synthesis_relevance_rendered_intake import (
    PLAN_SCHEMA, _check_receipt, _digest, stage_blind_session
)
from phonoweave.synthesis_relevance_reference_renderer import render_reference_plan


def fixture(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    data = make_plan()
    for i, row in enumerate(data["recordings"]):
        path = corpus / row["relative_wav"]
        path.parent.mkdir(parents=True, exist_ok=True)
        # 1-second non-silent, distinct synthetic signals with onset and vowel.
        rate = 48000
        time = np.arange(rate, dtype=np.float64) / rate
        sig = (
            0.1 * np.sin(2 * np.pi * (1300 + i * 13) * time)
            + 0.17 * np.sin(2 * np.pi * (220 + i * 5) * time)
        )
        pcm = (sig * 32767).astype("<i2").tobytes()
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes(pcm)
        row["wav_sha256"] = _digest(path)
        row["frication_onset_ms"] = 100.0
        row["cv_boundary_ms"] = 300.0
    frozen_sha = sha256(json.dumps(
        data, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    hashes = {r["take_id"]: r["wav_sha256"] for r in data["recordings"]}
    tasks = []
    for final in FINALS:
        pair = next(x for x in PAIRS if final in x)
        for repeat in range(1, REPEATS + 1):
            take = f"S02_sh{final}_{repeat:02d}"
            tasks.append({
                "target_take_id": take, "target_final": final,
                "source_reference_sha256": hashes[take],
                "allowed_S01_pooled_donor_take_ids": sorted(
                    f"S01_sh{f}_{n:02d}" for f in pair for n in range(1, REPEATS + 1)),
                "allowed_S01_split_donor_take_ids": sorted(
                    f"S01_sh{final}_{n:02d}" for n in range(1, REPEATS + 1)),
                "render_wavs": {
                    "pooled": f"{take}/pooled.wav",
                    "split": f"{take}/split.wav",
                },
            })
    plan = {
        "schema": PLAN_SCHEMA,
        "source_manifest_sha256": frozen_sha,
        "source_audio_sha256_by_take": hashes,
        "speaker_id": data["speaker_id"],
        "pitch_subbank": data["pitch_subbank"],
        "task_count": len(tasks), "tasks": tasks,
    }
    return corpus, data, plan


def test_end_to_end_reference_render_and_blind_pack(tmp_path):
    corpus, manifest, plan = fixture(tmp_path)
    renders = tmp_path / "reference_renders"
    result = render_reference_plan(plan, manifest, corpus, renders)
    assert result["rendered_wavs"] == 48
    assert not result["actual_singing_resampler_used"]
    assert result["S02_wav_waveforms_used_for_rendering"] is False
    receipt = json.loads((renders / "renderer_receipt.json").read_text())
    rows = _check_receipt(plan, receipt, renders)
    assert len(rows) == 24
    for record in rows:
        a = record["renders"]["pooled"]["file"]
        b = record["renders"]["split"]["file"]
        assert a.read_bytes() != b.read_bytes()
        assert record["receipt"]["pooled_donor_take_ids_used"][0].startswith("S01_")
        assert record["receipt"]["split_donor_take_ids_used"][0].startswith("S01_")
        with wave.open(str(a), "rb") as wa, wave.open(str(b), "rb") as wb:
            assert wa.getnframes() == wb.getnframes()
            assert wa.getframerate() == wb.getframerate() == 48000
    public, key = stage_blind_session(
        plan, receipt, renders,
        tmp_path / "blind", tmp_path / "private.json",
        corpus=corpus, seed_hex="a"*64,
    )
    assert len(public["trials"]) == 24
    assert sum(x["A_condition"] == "pooled" for x in key["trials"]) == 12
    with pytest.raises(FileExistsError):
        render_reference_plan(plan, manifest, corpus, renders)


def test_source_tamper_and_corpus_write_guards(tmp_path):
    corpus, manifest, plan = fixture(tmp_path)
    with pytest.raises(ValueError, match="inside original capture corpus"):
        render_reference_plan(plan, manifest, corpus, corpus / "renders")
    tampered = deepcopy(plan)
    tampered["source_manifest_sha256"] = "0"*64
    with pytest.raises(ValueError, match="stale"):
        render_reference_plan(tampered, manifest, corpus, tmp_path / "bad")
    path = corpus / "audio/S01/sha_01.wav"
    path.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum mismatch"):
        render_reference_plan(plan, manifest, corpus, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_donor_annotation_and_identity_guard(tmp_path):
    corpus, manifest, plan = fixture(tmp_path)
    bad = deepcopy(manifest)
    for row in bad["recordings"]:
        if row["session_id"] == "S01":
            row["cv_boundary_ms"] = None
            break
    badplan = deepcopy(plan)
    badplan["source_manifest_sha256"] = sha256(json.dumps(
        bad, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    with pytest.raises(ValueError, match="annotation"):
        render_reference_plan(badplan, bad, corpus, tmp_path / "out")


def test_S02_audio_is_not_decoded(tmp_path, monkeypatch):
    corpus, manifest, plan = fixture(tmp_path)
    from phonoweave import synthesis_relevance_reference_renderer as renderer
    old_read = renderer.read_wav

    def checked(path):
        assert "/S02/" not in path.as_posix(), "heldout WAV leaked into renderer!"
        return old_read(path)

    monkeypatch.setattr(renderer, "read_wav", checked)
    result = render_reference_plan(plan, manifest, corpus, tmp_path / "renders")
    assert result["rendered_wavs"] == 48
