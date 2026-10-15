"""Guided capture safety: no overwrite, resume and split-date isolation."""
from hashlib import sha256
import json

import pytest

from phonoweave.synthesis_relevance_controlled_corpus import make_plan
from phonoweave.synthesis_relevance_controlled_capture import capture_session


def _manifest(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(make_plan()), encoding="utf-8")
    return target


def _dummy(frames, device=None):
    return b"\x00\x01" * frames


def test_capture_session_resumable_and_no_overwrite(tmp_path):
    manifest = _manifest(tmp_path)
    responses = iter(["", "q"])
    count = capture_session(
        manifest, "S01", room_id="studio", gain_setting="fixed",
        target_pitch_hz=220, duration=0.3, device_id="mic",
        confirm_reset=True, recorder=_dummy,
        ask=lambda message: next(responses), capture_date="2026-10-14")
    assert count == 1
    data = json.loads(manifest.read_text())
    acquired = [item for item in data["recordings"] if item["wav_sha256"]]
    assert len(acquired) == 1
    target = tmp_path / acquired[0]["relative_wav"]
    before = target.read_bytes()
    assert acquired[0]["wav_sha256"] == sha256(before).hexdigest()
    responses = iter(["q"])
    assert capture_session(
        manifest, "S01", room_id="studio", gain_setting="fixed",
        target_pitch_hz=220, duration=0.3, device_id="mic",
        confirm_reset=True, recorder=_dummy,
        ask=lambda message: next(responses), capture_date="2026-10-14") == 0
    assert target.read_bytes() == before


def test_second_session_cannot_start_before_first(tmp_path):
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="S01 must be completed"):
        capture_session(
            manifest, "S02", room_id="studio", gain_setting="fixed",
            target_pitch_hz=220, duration=0.3, device_id="mic",
            confirm_reset=True, recorder=_dummy,
            ask=lambda _: "q", capture_date="2026-10-15")


def test_resume_metadata_change_is_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    capture_session(
        manifest, "S01", room_id="studio", gain_setting="fixed",
        target_pitch_hz=220, duration=0.3, device_id="mic",
        confirm_reset=True, recorder=_dummy,
        ask=lambda _: "q", capture_date="2026-10-14")
    with pytest.raises(ValueError, match="metadata changed"):
        capture_session(
            manifest, "S01", room_id="other_room", gain_setting="fixed",
            target_pitch_hz=220, duration=0.3, device_id="mic",
            confirm_reset=True, recorder=_dummy,
            ask=lambda _: "q", capture_date="2026-10-14")


def test_unregistered_existing_wav_refuses_overwrite(tmp_path):
    manifest = _manifest(tmp_path)
    data = json.loads(manifest.read_text())
    # Precreate all expected WAV paths to ensure whichever randomized take
    # comes first cannot be overwritten.
    for row in data["recordings"]:
        if row["session_id"] == "S01":
            target = tmp_path / row["relative_wav"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"do-not-overwrite")
    with pytest.raises(FileExistsError, match="without matching manifest hash"):
        capture_session(
            manifest, "S01", room_id="studio", gain_setting="fixed",
            target_pitch_hz=220, duration=0.3, device_id="mic",
            confirm_reset=True, recorder=_dummy,
            ask=lambda _: "", capture_date="2026-10-14")
