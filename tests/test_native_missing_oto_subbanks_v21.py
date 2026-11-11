"""Regression checks: explicitly declared voicebank subbanks without oto.ini.

The template declares recording targets; no IPA or phoneme identification is inferred.
"""
import json
import wave
from pathlib import Path

import pytest

from phonoweave.native_recording_plan_v20 import (
    TARGET_SCHEMA, build_native_recording_workflow,
)


def pcm(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(44100)
        writer.writeframes(b"\x01\x00" * 500)


def manifest(tmp_path: Path, bank: str = "D4") -> Path:
    path = tmp_path / "native_targets.json"
    path.write_text(json.dumps({
        "schema": TARGET_SCHEMA,
        "targets": [{"line": "ŋa", "wav": "ŋa.wav", "subbanks": [bank]}],
    }, ensure_ascii=False), encoding="utf-8")
    return path


def test_declared_audio_only_subbank_means_oto_review_not_rerecord(tmp_path):
    root = tmp_path / "voicebank"
    pcm(root / "D4" / "ŋa.wav")
    report = build_native_recording_workflow(
        root, expected_reclist=manifest(tmp_path),
    )
    state = report["coverage"]["template_states"][0]["banks"][0]
    assert state["status"] == "audio_present_needs_oto_review"
    assert report["voicebank"]["subbanks_missing_oto"] == ["D4"]
    assert report["recording_plan"]["recording_lines"] == []
    assert report["recording_plan"]["oto_review_without_rerecording"] == [
        {"line": "ŋa", "banks": ["D4"]},
    ]


def test_declared_empty_bank_means_missing_audio(tmp_path):
    root = tmp_path / "voicebank"
    (root / "D4").mkdir(parents=True)
    report = build_native_recording_workflow(
        root, expected_reclist=manifest(tmp_path), max_lines=1,
    )
    assert report["voicebank"]["subbanks_missing_oto"] == ["D4"]
    assert report["recording_plan"]["recording_lines"] == ["ŋa"]
    assert report["recording_plan"]["selected"][0]["missing_audio_banks"] == ["D4"]


def test_explicit_bank_flag_checks_audio_only_bank_with_plain_text_template(tmp_path):
    root = tmp_path / "voicebank"
    pcm(root / "D4" / "ŋa.wav")
    template = tmp_path / "native_reclist.txt"
    template.write_text("ŋa\n", encoding="utf-8")
    report = build_native_recording_workflow(
        root, expected_reclist=template, bank_names=("D4",),
    )
    assert report["voicebank"]["subbanks"] == ["D4"]
    assert report["coverage"]["registration_or_ambiguity_review_checks"] == 1
    assert report["recording_plan"]["selected_count"] == 0


def test_nonexistent_declared_bank_is_an_error_not_fake_absence(tmp_path):
    root = tmp_path / "voicebank"
    root.mkdir()
    with pytest.raises(ValueError, match="directory is missing"):
        build_native_recording_workflow(root, expected_reclist=manifest(tmp_path))


def test_undeclared_audio_dir_is_not_assumed_to_be_a_pitch_bank(tmp_path):
    root = tmp_path / "voicebank"
    pcm(root / "D4" / "ŋa.wav")
    template = tmp_path / "native_reclist.txt"
    template.write_text("ŋa\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot infer target subbanks"):
        build_native_recording_workflow(root, expected_reclist=template)


def test_existing_registered_bank_is_still_covered_without_changes(tmp_path):
    root = tmp_path / "voicebank"
    pcm(root / "D4" / "ŋa.wav")
    (root / "D4" / "oto.ini").write_text(
        "ŋa.wav=ŋa,0,0,0,0,0\n", encoding="utf-8"
    )
    report = build_native_recording_workflow(
        root, expected_reclist=manifest(tmp_path),
    )
    assert report["voicebank"]["subbanks_missing_oto"] == []
    assert report["coverage"]["template_states"][0]["banks"][0]["status"] == "audio_present_registered"
    assert report["recording_plan"]["recording_lines"] == []
