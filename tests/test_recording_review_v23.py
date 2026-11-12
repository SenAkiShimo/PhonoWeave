"""v2.3: separate actual recording tasks from non-recording repair work.

Fixtures are synthetic PCM and OTO configurations, NOT validated phonemes.
"""
from __future__ import annotations

import json
import wave
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

from phonoweave import gui
from phonoweave.native_recording_plan_v20 import (
    TARGET_SCHEMA, build_native_recording_workflow,
)
from phonoweave.recording_review_v23 import format_review_tasks, grouped_review_tasks
from phonoweave.recording_workflow_v18 import export_recording_workflow
from phonoweave.recording_plan_page_v19 import HTML


def make_wav(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(44100)
        writer.writeframes(b"\x00\x01" * 500)


def scenario(tmp_path):
    root = tmp_path / "voicebank"
    d4 = root / "D4"
    d4.mkdir(parents=True)
    make_wav(d4 / "good.wav")
    make_wav(d4 / "unconfigured.wav")
    (d4 / "bad.wav").write_bytes(b"not a WAV")
    (d4 / "oto.ini").write_text(
        "good.wav=ŋa,0,0,0,0,0\n"
        "bad.wav=坏音,0,0,0,0,0\n", encoding="utf-8"
    )
    e4 = root / "E4"
    e4.mkdir()
    obj = {
        "schema": TARGET_SCHEMA,
        "targets": [
            {"line": "ŋa", "wav": "good.wav", "subbanks": ["D4"]},
            {"line": "坏音", "wav": "bad.wav", "subbanks": ["D4"]},
            {"line": "待登记", "wav": "unconfigured.wav", "subbanks": ["D4"]},
            {"line": "未映射", "wav": "other.wav", "subbanks": ["D4"]},
            {"line": "确实未录", "wav": "absent.wav", "subbanks": ["E4"]},
        ]
    }
    manifest = tmp_path / "original.json"
    manifest.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return root, manifest


def test_real_gap_only_one_line_while_three_items_need_review(tmp_path):
    root, manifest = scenario(tmp_path)
    report = build_native_recording_workflow(root, expected_reclist=manifest)
    assert report["recording_plan"]["recording_lines"] == ["确实未录"]
    tasks = grouped_review_tasks(report)
    assert [x["line"] for x in tasks["oto_configuration"]] == ["待登记"]
    assert set(x["line"] for x in tasks["source_verification"]) == {"坏音", "未映射"}
    assert report["recording_plan"]["oto_configuration_count"] == 1
    assert report["recording_plan"]["source_verification_count"] == 2
    assert report["coverage"]["missing_audio_checks"] == 1


def test_corrupted_audio_does_not_become_a_new_recording(tmp_path):
    root, manifest = scenario(tmp_path)
    report = build_native_recording_workflow(root, expected_reclist=manifest)
    assert "坏音" not in report["recording_plan"]["recording_lines"]
    tasks = grouped_review_tasks(report)
    task = next(x for x in tasks["source_verification"] if x["line"] == "坏音")
    assert task["reason"] == "unreadable_wav"


def test_review_file_is_not_the_reclist(tmp_path):
    root, manifest = scenario(tmp_path)
    report = build_native_recording_workflow(root, expected_reclist=manifest)
    path = export_recording_workflow(report, tmp_path / "export", voicebank=root)
    reclist = (path / "supplement_reclist.txt").read_text(encoding="utf-8")
    review = (path / "recording_review.txt").read_text(encoding="utf-8")
    assert reclist == "确实未录\n"
    assert "坏音" in review and "待登记" in review and "未映射" in review
    assert "确实未录" not in review
    assert json.loads((path / "recording_plan.json").read_text(encoding="utf-8")) == report


def test_zero_budget_still_reports_repairs(tmp_path):
    root, manifest = scenario(tmp_path)
    report = build_native_recording_workflow(root, expected_reclist=manifest, max_lines=0)
    assert report["recording_plan"]["recording_lines"] == []
    assert len(report["recording_plan"]["review_tasks"]) == 3
    assert report["recording_plan"]["deferred_or_skipped"][0]["line"] == "确实未录"


def test_review_text_preserves_original_user_labels(tmp_path):
    root, manifest = scenario(tmp_path)
    report = build_native_recording_workflow(root, expected_reclist=manifest)
    review = format_review_tasks(report)
    assert "待登记 | D4 | audio_present_needs_oto_review" in review
    assert "坏音 | D4 | invalid_audio | unreadable_wav" in review
    assert "未映射 | D4 | unverified_template_mapping" in review
    assert "NOT additional recording instructions" in review


def test_empty_legacy_review_is_not_an_all_clear():
    report = {"recording_plan": {"recording_lines": ["fa"]}}
    assert grouped_review_tasks(report) == {
        "oto_configuration": [], "source_verification": []
    }
    assert "not proof of complete phonetic coverage" in format_review_tasks(report)


def test_gui_has_separate_review_sections_and_download_link():
    assert 'id="oto-review"' in HTML
    assert 'id="source-review"' in HTML
    assert '/api/recording-plan/download/review' in HTML
    assert "textContent" in HTML
    assert "innerHTML=" not in HTML


def test_gui_review_download_returns_only_review_tasks(tmp_path):
    root, manifest = scenario(tmp_path)
    report = build_native_recording_workflow(root, expected_reclist=manifest)
    server = ThreadingHTTPServer(("127.0.0.1", 0), gui._Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    with gui._STATE_LOCK:
        gui._STATE.recording_report = report
    worker.start()
    try:
        address = "http://127.0.0.1:" + str(server.server_address[1])
        with urlopen(address + "/api/recording-plan/download/review", timeout=8) as response:
            text = response.read().decode("utf-8")
            assert response.headers.get_content_type() == "text/plain"
            assert "待登记" in text and "坏音" in text
            assert "确实未录" not in text
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
        with gui._STATE_LOCK:
            gui._STATE.recording_report = None


def test_explicit_registered_missing_wav_is_a_gap_even_if_other_wavs_exist(tmp_path):
    root = tmp_path / "voicebank"
    d4 = root / "D4"
    d4.mkdir(parents=True)
    make_wav(d4 / "other.wav")
    (d4 / "oto.ini").write_text(
        "absent.wav=确实缺源,0,0,0,0,0\n"
        "other.wav=其他,0,0,0,0,0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "targets.json"
    manifest.write_text(json.dumps({
        "schema": TARGET_SCHEMA,
        "targets": [{"line": "确实缺源", "wav": "absent.wav", "subbanks": ["D4"]}],
    }, ensure_ascii=False), encoding="utf-8")
    report = build_native_recording_workflow(root, expected_reclist=manifest)
    assert report["recording_plan"]["recording_lines"] == ["确实缺源"]
    state = report["coverage"]["template_states"][0]["banks"][0]
    assert state["status"] == "missing_audio"
    assert state["reason"] == "registered_source_wav_missing"
    assert report["recording_plan"]["review_tasks"] == []
