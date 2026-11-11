"""v20 opaque recording-template and GUI/CLI regression tests.

These tests verify file/template coverage, NOT new phone discovery.
"""
from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import pytest

from phonoweave.native_recording_plan_v20 import (
    SCHEMA, TARGET_SCHEMA, read_targets, build_native_recording_workflow,
)


def wav(path: Path, sample: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(44100)
        out.writeframes(struct.pack("<300h", *([sample] * 300)))


def example(tmp_path: Path):
    root = tmp_path / "bank"
    root.mkdir()
    for name in ("A3", "D4"):
        sub = root / name
        sub.mkdir()
        (sub / "oto.ini").write_text(
            "ŋa.wav=ŋa,0,0,0,0,0\n"
            "tɕi.wav=tɕi,0,0,0,0,0\n", encoding="utf-8"
        )
    wav(root / "A3" / "ŋa.wav")
    wav(root / "A3" / "tɕi.wav", 80)
    wav(root / "D4" / "ŋa.wav")
    template = tmp_path / "original.txt"
    template.write_text("ŋa\ntɕi\n", encoding="utf-8")
    return root, template


def test_native_unicode_missing_in_one_bank(tmp_path):
    root, template = example(tmp_path)
    plan = build_native_recording_workflow(root, expected_reclist=template)
    assert plan["schema"] == SCHEMA
    assert plan["coverage"]["expected_template_count"] == 2
    assert plan["coverage"]["subbank_template_checks"] == 4
    assert plan["recording_plan"]["recording_lines"] == ["tɕi"]
    assert plan["recording_plan"]["selected"][0]["missing_audio_banks"] == ["D4"]
    assert plan["restrictions"]["no_new_phoneme_or_alias_generated"]


def test_pcm_duplicate_does_not_count_as_separate_performance(tmp_path):
    root, template = example(tmp_path)
    plan = build_native_recording_workflow(root, expected_reclist=template)
    item = next(x for x in plan["coverage"]["template_states"] if x["line"] == "ŋa")
    assert item["identical_pcm_in_multiple_subbanks"] is True
    assert item["existing_distinct_pcm_sources"] == 1


def test_scheduled_is_not_recorded(tmp_path):
    root, template = example(tmp_path)
    scheduled = tmp_path / "scheduled.txt"
    scheduled.write_text("tɕi\n", encoding="utf-8")
    plan = build_native_recording_workflow(
        root, expected_reclist=template, existing_reclist=scheduled)
    assert plan["recording_plan"]["recording_lines"] == []
    assert plan["coverage"]["missing_audio_checks"] == 1
    assert plan["recording_plan"]["deferred_or_skipped"][0]["reason"] == "already_scheduled_not_verified_audio"


def test_zero_budget_never_recovers_missing_audio_silently(tmp_path):
    root, template = example(tmp_path)
    data = build_native_recording_workflow(root, expected_reclist=template, max_lines=0)
    assert data["recording_plan"]["recording_lines"] == []
    assert data["coverage"]["missing_audio_checks"] == 1
    assert data["recording_plan"]["deferred_or_skipped"][0]["reason"] == "recording_budget_reached"


def test_select_single_pitch_subbank(tmp_path):
    root, template = example(tmp_path)
    data = build_native_recording_workflow(root, expected_reclist=template, bank_names=("A3",))
    assert data["recording_plan"]["selected_count"] == 0
    assert data["voicebank"]["subbanks"] == ["A3"]


def test_existing_wav_without_oto_is_review_not_rerecord(tmp_path):
    root, template = example(tmp_path)
    wav(root / "D4" / "附加.wav", 32)
    template.write_text("附加\n", encoding="utf-8")
    data = build_native_recording_workflow(root, expected_reclist=template, bank_names=("D4",))
    assert data["recording_plan"]["recording_lines"] == []
    assert data["recording_plan"]["oto_review_without_rerecording"][0]["banks"] == ["D4"]


def test_explicit_json_template_without_phoneme_parsing(tmp_path):
    root, template = example(tmp_path)
    extra = root / "D4" / "file_009.wav"
    wav(extra, 23)
    with (root / "D4" / "oto.ini").open("a", encoding="utf-8") as out:
        out.write("file_009.wav=上海话原样符号,0,0,0,0,0\n")
    obj = {"schema": TARGET_SCHEMA, "targets": [
        {"line": "上海话原样符号", "wav": "file_009.wav", "subbanks": ["D4"]}]}
    manifest = tmp_path / "native.json"
    manifest.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    result = build_native_recording_workflow(root, expected_reclist=manifest)
    assert result["recording_plan"]["selected_count"] == 0
    assert result["coverage"]["template_states"][0]["banks"][0]["status"] == "audio_present_registered"


def test_exact_oto_alias_fallback_avoids_rerecord(tmp_path):
    root, template = example(tmp_path)
    wav(root / "D4" / "nonstandard.wav", 25)
    with (root / "D4" / "oto.ini").open("a", encoding="utf-8") as out:
        out.write("nonstandard.wav=原名,0,0,0,0,0\n")
    template.write_text("原名\n", encoding="utf-8")
    report = build_native_recording_workflow(root, expected_reclist=template, bank_names=("D4",))
    assert report["recording_plan"]["selected_count"] == 0
    assert report["coverage"]["template_states"][0]["banks"][0]["match"] == "exact_oto_alias"


def test_broken_pcm_is_not_recorded_evidence(tmp_path):
    root, template = example(tmp_path)
    (root / "D4" / "tɕi.wav").write_bytes(b"not a wave")
    report = build_native_recording_workflow(root, expected_reclist=template)
    assert report["recording_plan"]["recording_lines"] == ["tɕi"]
    assert report["coverage"]["template_states"][1]["banks"][1]["status"] == "invalid_audio"


@pytest.mark.parametrize("bad", ["../a", "/foo", "a\\b", "", "a\x00b"])
def test_unsafe_target_wav_names_fail(tmp_path, bad):
    path = tmp_path / "template.json"
    path.write_text(json.dumps({"schema": TARGET_SCHEMA, "targets": [
        {"line": "okay", "wav": bad}]}))
    with pytest.raises(ValueError):
        read_targets(path)


def test_duplicate_labels_fail_closed(tmp_path):
    path = tmp_path / "reclist.txt"
    path.write_text("ŋa\nŋa\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_targets(path)


def test_unknown_subbank_and_absent_oto_fail(tmp_path):
    root, template = example(tmp_path)
    with pytest.raises(ValueError, match="subbank"):
        build_native_recording_workflow(root, expected_reclist=template, bank_names=("invalid",))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="oto.ini"):
        build_native_recording_workflow(empty, expected_reclist=template)


def test_never_invents_non_author_script_line(tmp_path):
    root, template = example(tmp_path)
    report = build_native_recording_workflow(root, expected_reclist=template)
    assert set(report["recording_plan"]["recording_lines"]) <= {r["line"] for r in read_targets(template)}


def test_existing_gui_and_cli_native_routes_present():
    from phonoweave.recording_plan_page_v19 import HTML
    from phonoweave import gui
    from phonoweave import recording_workflow_v18
    from inspect import getsource
    assert 'id="native-reclist"' in HTML
    assert "native_reclist" in getsource(gui._Handler.do_POST)
    assert "--native-reclist" in getsource(recording_workflow_v18.main)


def test_wav_used_by_another_alias_is_review_not_verified(tmp_path):
    root, template = example(tmp_path)
    sub = root / "D4"
    wav(sub / "another.wav", 38)
    with (sub / "oto.ini").open("a", encoding="utf-8") as out:
        out.write("another.wav=not_the_target,0,0,0,0,0\n")
    obj = {"schema": TARGET_SCHEMA, "targets": [
        {"line": "required_custom_alias", "wav": "another.wav", "subbanks": ["D4"]}]}
    spec = tmp_path / "targets.json"
    spec.write_text(json.dumps(obj), encoding="utf-8")
    result = build_native_recording_workflow(root, expected_reclist=spec)
    assert result["recording_plan"]["recording_lines"] == []
    assert result["coverage"]["template_states"][0]["banks"][0]["status"] == "audio_present_needs_oto_review"
