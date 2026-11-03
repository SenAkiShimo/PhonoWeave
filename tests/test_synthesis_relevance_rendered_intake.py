"""Synthetic integration tests for external-render intake, no sound quality claims."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import wave

import pytest

from phonoweave.synthesis_relevance_rendered_intake import (
    PLAN_SCHEMA, _canonical, _check_receipt, _digest, create_render_plan,
    make_receipt_template, stage_blind_session, write_render_plan,
)
from phonoweave.synthesis_relevance_blind_score import compile_responses


def _wav(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        # Length and rate identical, waveform content intentionally distinct.
        w.writeframes((int(value).to_bytes(2, "little", signed=True)) * 9600)


def _fixture(tmp_path):
    from phonoweave.synthesis_relevance_controlled_corpus import FINALS, PAIRS, REPEATS
    root = tmp_path / "renders"
    root.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    original_hashes = {}
    for si, sid in enumerate(("S01", "S02")):
        for fi, final in enumerate(FINALS):
            for repeat in range(1, REPEATS + 1):
                name = f"{sid}_sh{final}_{repeat:02d}"
                path = corpus / f"audio/{sid}/sh{final}_{repeat:02d}.wav"
                _wav(path, 100 + si * 1000 + fi * 100 + repeat)
                original_hashes[name] = _digest(path)

    tasks = []
    for final in FINALS:
        pair = next(p for p in PAIRS if final in p)
        for repeat in range(1, REPEATS + 1):
            take = f"S02_sh{final}_{repeat:02d}"
            tasks.append({
                "target_take_id": take,
                "target_final": final,
                "source_reference_sha256": original_hashes[take],
                "allowed_S01_pooled_donor_take_ids": sorted(
                    f"S01_sh{f}_{j:02d}" for f in pair for j in range(1, REPEATS + 1)
                ),
                "allowed_S01_split_donor_take_ids": sorted(
                    f"S01_sh{final}_{j:02d}" for j in range(1, REPEATS + 1)
                ),
                "render_wavs": {
                    "pooled": f"{take}/pooled.wav",
                    "split": f"{take}/split.wav",
                },
            })
    plan = {
        "schema": PLAN_SCHEMA,
        "task_count": 24,
        "source_audio_sha256_by_take": original_hashes,
        "tasks": tasks,
    }
    receipt = make_receipt_template(plan)
    receipt.update({
        "engine_name": "test_render_engine",
        "engine_version": "0",
        "resampler_name": "test",
        "resampler_version": "0",
        "score_template_sha256": "a" * 64,
        "render_configuration_sha256": "b" * 64,
        "same_score_and_render_settings_between_arms_confirmed": True,
        "S02_audio_not_used_as_render_donor_or_conditioning_confirmed": True,
    })
    for i, task in enumerate(tasks):
        item = receipt["trial_receipts"][i]
        item["pooled_donor_take_ids_used"] = task["allowed_S01_pooled_donor_take_ids"]
        item["split_donor_take_ids_used"] = task["allowed_S01_split_donor_take_ids"]
        for j, arm in enumerate(("pooled", "split")):
            path = root / task["render_wavs"][arm]
            _wav(path, (i + 1) * 10 + j + 1)
            item[f"{arm}_wav_sha256"] = _digest(path)
    return root, corpus, plan, receipt

def test_receipts_validated_and_split_training_only(tmp_path):
    render, _, plan, receipt = _fixture(tmp_path)
    files = _check_receipt(plan, receipt, render)
    assert len(files) == 24
    assert files[0]["renders"]["pooled"]["format"][:2] == (48000, 9600)
    forged = deepcopy(receipt)
    forged["trial_receipts"][0]["split_donor_take_ids_used"] = ["S02_a"]
    with pytest.raises(ValueError, match="unapproved"):
        _check_receipt(plan, forged, render)
    forged = deepcopy(receipt)
    forged["trial_receipts"][0]["pooled_wav_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        _check_receipt(plan, forged, render)


def test_reject_duplicate_condition_and_tampering(tmp_path):
    render, _, plan, receipt = _fixture(tmp_path)
    src = render / plan["tasks"][0]["render_wavs"]["pooled"]
    target = render / plan["tasks"][0]["render_wavs"]["split"]
    target.write_bytes(src.read_bytes())
    receipt["trial_receipts"][0]["split_wav_sha256"] = _digest(target)
    with pytest.raises(ValueError, match="identical"):
        _check_receipt(plan, receipt, render)


def test_blinding_and_response_scoring(tmp_path):
    render, corpus, plan, receipt = _fixture(tmp_path)
    package = tmp_path / "participant"
    keypath = tmp_path / "private" / "key.json"
    manifest, key = stage_blind_session(
        plan, receipt, render, package, keypath, corpus=corpus, seed_hex="1"*64
    )
    assert len(manifest["trials"]) == 24
    assert len(key["trials"]) == 24
    assert "condition" not in json.dumps(manifest)
    assert not (package / "key.json").exists()
    assert (package / "response_template.json").exists()
    assert sum(t["A_condition"] == "pooled" for t in key["trials"]) == 12
    public = json.loads((package / "manifest.json").read_text())
    assert public == manifest
    assert key == json.loads(keypath.read_text())
    template = json.loads((package / "response_template.json").read_text())
    template["listener_id"] = "listener01"
    for t in template["votes"]:
        t["choice"] = "A"
    scores = compile_responses(manifest, key, [template], package)
    assert scores["listeners"] == 1
    assert scores["votes"]["pooled"] == 12
    assert scores["votes"]["split"] == 12
    assert scores["unrated_or_missing_votes"] == 0
    with pytest.raises(ValueError, match="repeated listener"):
        compile_responses(manifest, key, [template, template], package)
    with pytest.raises(FileExistsError):
        stage_blind_session(
            plan, receipt, render, package, tmp_path / "another_key.json",
            corpus=corpus, seed_hex="2"*64
        )


def test_private_key_and_media_protection(tmp_path):
    render, corpus, plan, receipt = _fixture(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        stage_blind_session(plan, receipt, render, render / "public",
                            tmp_path / "key", corpus=corpus, seed_hex="0"*64)
    with pytest.raises(ValueError, match="separate"):
        stage_blind_session(plan, receipt, render, tmp_path / "public",
                            tmp_path / "public" / "key", corpus=corpus,
                            seed_hex="0"*64)
    manifest, key = stage_blind_session(
        plan, receipt, render, tmp_path / "public",
        tmp_path / "secret.json", corpus=corpus, seed_hex="0"*64
    )
    (tmp_path / "public" / manifest["trials"][0]["A"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        compile_responses(manifest, key, [], tmp_path / "public")


def test_source_corpus_tampering_blocks_blind_publication(tmp_path):
    render, corpus, plan, receipt = _fixture(tmp_path)
    source = corpus / "audio/S01/sha_01.wav"
    source.write_bytes(b"not the original source")
    with pytest.raises(ValueError, match="source audio checksum mismatch"):
        stage_blind_session(
            plan, receipt, render, tmp_path / "participant",
            tmp_path / "answer.json", corpus=corpus, seed_hex="a"*64,
        )
    assert not (tmp_path / "participant").exists()


def test_forged_donor_pool_blocks_blind_publication(tmp_path):
    render, corpus, plan, receipt = _fixture(tmp_path)
    plan["tasks"][0]["allowed_S01_split_donor_take_ids"] = ["S01_shua_01"]
    with pytest.raises(ValueError, match="donor pools"):
        stage_blind_session(
            plan, receipt, render, tmp_path / "participant",
            tmp_path / "answer.json", corpus=corpus, seed_hex="a"*64,
        )


def test_incomplete_corpus_cannot_authorize_render_plan(tmp_path):
    from phonoweave.synthesis_relevance_controlled_corpus import make_plan
    with pytest.raises(ValueError, match="not audit-ready"):
        create_render_plan(make_plan(), {}, tmp_path)


def test_plan_write_guards(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    plan = {
        "schema": PLAN_SCHEMA, "task_count": 24, "tasks": [
            {"target_take_id": str(i)} for i in range(24)
        ]
    }
    out = tmp_path / "plan"
    write_render_plan(plan, out, corpus=root)
    assert (out / "render_receipt_template.json").exists()
    with pytest.raises(FileExistsError):
        write_render_plan(plan, out, corpus=root)
    with pytest.raises(ValueError, match="outside"):
        write_render_plan(plan, root / "nested", corpus=root)
