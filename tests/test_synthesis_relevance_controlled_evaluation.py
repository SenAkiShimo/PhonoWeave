"""Friction-only, session-isolated pilot acoustic proxy safety tests."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import wave

import numpy as np
import pytest

from phonoweave.synthesis_relevance_controlled_corpus import make_plan
from phonoweave.synthesis_relevance_controlled_evaluation import (
    evaluate_controlled_manifest, run_controlled_proxy, PAIR_NAMES, SCHEMA
)
from phonoweave.synthesis_relevance_controlled_evaluation_cli import main


def _record(root, data):
    for i, session in enumerate(data["sessions"]):
        session.update(
            capture_date=f"2026-10-{14+i}", device_id="mic1",
            room_id="quiet", gain_setting="fixed", target_pitch_hz=220,
            setup_reset_confirmed=True,
        )
    sr = 48000
    t = np.arange(sr, dtype=np.float64) / sr
    for i, record in enumerate(data["recordings"]):
        path = root / record["relative_wav"]
        path.parent.mkdir(parents=True, exist_ok=True)
        # Distinct, non-silent 1-second stimuli with all bands populated.
        # Synthetic integrity fixture; not real capture or study evidence.
        sig = (
            0.17 * np.sin(2 * np.pi * (1730+i*7) * t)
            + 0.15 * np.sin(2 * np.pi * (3100+i*11) * t)
            + 0.11 * np.sin(2 * np.pi * (4600+i*13) * t)
        )
        pcm = np.clip(sig * 32767, -32768, 32767).astype("<i2").tobytes()
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(pcm)
        record["wav_sha256"] = sha256(path.read_bytes()).hexdigest()
        record["frication_onset_ms"] = 100
        record["cv_boundary_ms"] = 300


def test_empty_manifest_abstains(tmp_path):
    result = evaluate_controlled_manifest(make_plan(), tmp_path)
    assert result["status"] == "insufficient"
    assert result["budget_curves"] == []
    assert result["outcome_status"] == "not_evaluated"


def test_complete_grid_and_budget_parity(tmp_path):
    data = make_plan()
    _record(tmp_path, data)
    result = evaluate_controlled_manifest(data, tmp_path)
    assert result["status"] == "evaluable_acoustic_proxy"
    assert result["n_heldout_WAVs"] == 24
    assert len(result["budget_curves"]) == 15
    for k in range(5):
        chunk = [c for c in result["budget_curves"]
                 if c["budget_split_pairs"] == k]
        assert len(chunk) == 3
        assert {c["n_heldout_WAVs"] for c in chunk} == {24}
        assert all(len(c["selected_pairs"]) == k for c in chunk)
    assert result["protocol"]["heldout_role_used_for_selection"] is False
    assert result["rankings"]["fixed_lexicographic"] == sorted(PAIR_NAMES)
    assert all(np.isfinite(v) for v in result["train_only_utility"].values())
    assert evaluate_controlled_manifest(data, tmp_path) == result


def test_S02_audio_changes_do_not_affect_training_rank(tmp_path):
    data = make_plan()
    _record(tmp_path, data)
    original = evaluate_controlled_manifest(data, tmp_path)
    row = next(r for r in data["recordings"] if r["session_id"] == "S02")
    path = tmp_path / row["relative_wav"]
    with wave.open(str(path), "rb") as wav:
        frames, rate = wav.readframes(wav.getnframes()), wav.getframerate()
    samples = np.frombuffer(frames, dtype="<i2").copy()
    samples[6000:15000] = (samples[6000:15000] // 3)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate)
        wav.writeframes(samples.astype("<i2").tobytes())
    row["wav_sha256"] = sha256(path.read_bytes()).hexdigest()
    altered = evaluate_controlled_manifest(data, tmp_path)
    assert altered["rankings"] == original["rankings"]
    assert altered["train_only_utility"] == original["train_only_utility"]
    assert altered["source_manifest_sha256"] != original["source_manifest_sha256"]


def test_bad_annotation_and_hash_abstain(tmp_path):
    data = make_plan()
    _record(tmp_path, data)
    bad = deepcopy(data)
    bad["recordings"][0]["wav_sha256"] = "bad"
    assert evaluate_controlled_manifest(bad, tmp_path)["status"] == "insufficient"
    bad = deepcopy(data)
    bad["recordings"][0]["cv_boundary_ms"] = None
    assert evaluate_controlled_manifest(bad, tmp_path)["status"] == "insufficient"


def test_too_short_annotated_frication_rejected(tmp_path):
    data = make_plan()
    _record(tmp_path, data)
    data["recordings"][0]["cv_boundary_ms"] = 120
    with pytest.raises(ValueError, match="too short"):
        evaluate_controlled_manifest(data, tmp_path)


def test_noncanonical_capture_filename_rejected(tmp_path):
    data = make_plan()
    _record(tmp_path, data)
    row = data["recordings"][0]
    row["take_id"] = "forged"
    with pytest.raises(ValueError, match="noncanonical"):
        evaluate_controlled_manifest(data, tmp_path)


def test_cli_nonoverwrite_and_report(tmp_path, capsys):
    data = make_plan()
    manifest, dest = tmp_path / "manifest.json", tmp_path / "result.json"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    assert main([str(manifest), "--output", str(dest)]) == 2
    assert json.loads(dest.read_text())["status"] == "insufficient"
    with pytest.raises(SystemExit) as exc:
        main([str(manifest), "--output", str(dest)])
    assert exc.value.code == 2
