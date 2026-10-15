"""Controlled pilot grid and integrity stop conditions."""
from copy import deepcopy
from hashlib import sha256
import wave

from phonoweave.synthesis_relevance_controlled_corpus import (
    FINALS, audit_manifest, make_plan, write_plan,
)


def test_frozen_48_take_grid_and_roles(tmp_path):
    data = make_plan()
    assert len(data["recordings"]) == 48
    assert len({r["take_id"] for r in data["recordings"]}) == 48
    assert len({r["relative_wav"] for r in data["recordings"]}) == 48
    assert {r["final"] for r in data["recordings"]} == set(FINALS)
    assert {s["role"] for s in data["sessions"]} == {"train", "heldout"}
    report = audit_manifest(data, tmp_path)
    assert not report["analysis_ready"]
    assert report["outcome_status"] == "not_evaluated"
    assert report["planned_takes"] == 48


def test_existing_plan_cannot_be_overwritten(tmp_path):
    path = tmp_path / "manifest.json"
    write_plan(path)
    try:
        write_plan(path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing capture manifest was overwritten")


def test_tampered_pairs_and_session_dates_rejected(tmp_path):
    data = make_plan()
    data["final_pairs"][0] = ["a", "e"]
    for session in data["sessions"]:
        session["capture_date"] = "2026-10-14"
    report = audit_manifest(data, tmp_path)
    assert "predeclared_pair_set_modified" in report["errors"]
    assert "sessions_must_use_distinct_capture_dates" in report["errors"]


def _populate(data, root):
    for j, session in enumerate(data["sessions"]):
        session.update(capture_date=f"2026-10-{14+j}", device_id="mic01",
                       room_id="room01", gain_setting="fixed",
                       target_pitch_hz=220, setup_reset_confirmed=True)
    for i, item in enumerate(data["recordings"]):
        path = root / item["relative_wav"]
        path.parent.mkdir(parents=True, exist_ok=True)
        # Distinct PCM content without changing rate, count, or format.
        frames = bytes([i, 0]) * 48000
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(48000)
            audio.writeframes(frames)
        item["wav_sha256"] = sha256(path.read_bytes()).hexdigest()
        item["frication_onset_ms"] = 20.0
        item["cv_boundary_ms"] = 200.0


def test_complete_manifest_requires_integrity_and_annotations(tmp_path):
    data = make_plan()
    _populate(data, tmp_path)
    report = audit_manifest(data, tmp_path)
    assert report["analysis_ready"]
    assert report["verified_wav_files"] == 48
    assert all(v == 3 for fs in report["per_session_final_verified"].values()
               for v in fs.values())
    item = data["recordings"][0]
    item["wav_sha256"] = "invalid"
    assert not audit_manifest(data, tmp_path)["analysis_ready"]
    item["wav_sha256"] = sha256((tmp_path / item["relative_wav"]).read_bytes()).hexdigest()
    item["cv_boundary_ms"] = None
    assert not audit_manifest(data, tmp_path)["analysis_ready"]


def test_duplicate_physical_content_and_missing_wav_block(tmp_path):
    data = make_plan()
    _populate(data, tmp_path)
    first, second = data["recordings"][:2]
    (tmp_path / second["relative_wav"]).write_bytes(
        (tmp_path / first["relative_wav"]).read_bytes())
    second["wav_sha256"] = first["wav_sha256"]
    result = audit_manifest(data, tmp_path)
    assert any("duplicate_pcm_file_hash" in issue for issue in result["errors"])
    (tmp_path / first["relative_wav"]).unlink()
    assert not audit_manifest(data, tmp_path)["analysis_ready"]


def test_wrong_target_grid_never_passes(tmp_path):
    data = make_plan()
    _populate(data, tmp_path)
    bad = deepcopy(data)
    bad["recordings"][0]["repeat"] = 2
    assert not audit_manifest(bad, tmp_path)["analysis_ready"]
