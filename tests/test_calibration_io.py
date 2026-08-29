import json
import struct
from pathlib import Path

from phonoweave.calibration_io import (
    create_calibration_session,
    protocol_payload,
    recording_filename,
    save_calibration_recording,
)
from phonoweave.calibration_protocol import live_calibration_protocol


def _wav() -> bytes:
    data = b"\x00\x00" * 8
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 48000, 96000, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


def test_protocol_payload_exposes_tokens_and_frozen_metadata() -> None:
    payload = protocol_payload(live_calibration_protocol())
    assert payload["version"] == "0.1"
    assert payload["automatic_supplement_rounds"] == 1
    first = payload["prompts"][0]
    assert first["spoken_pattern"] == "zha a zha a"
    assert first["tokens"] == ["zha", "a", "zha", "a"]


def test_create_session_writes_protocol_and_recordings_dir(tmp_path: Path) -> None:
    session_id, session_dir = create_calibration_session(
        live_calibration_protocol(), tmp_path
    )
    assert session_dir.name == session_id
    assert (session_dir / "recordings").is_dir()
    payload = json.loads((session_dir / "protocol.json").read_text(encoding="utf-8"))
    assert payload["name"] == "live_calibration"


def test_recording_filename_is_stable() -> None:
    prompt = live_calibration_protocol().prompts[0]
    assert recording_filename(0, prompt) == "001_zh_plain.wav"


def test_save_recording_writes_wav_and_metadata(tmp_path: Path) -> None:
    protocol = live_calibration_protocol()
    session_id, _ = create_calibration_session(protocol, tmp_path)
    output = save_calibration_recording(
        tmp_path, session_id, 0, protocol.prompts[0], _wav(), 48000
    )
    assert output.name == "001_zh_plain.wav"
    wav = output.read_bytes()
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert struct.unpack("<I", wav[4:8])[0] == len(wav) - 8
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["base_unit"] == "zh"
    assert metadata["context_family"] == "plain"
    assert metadata["repetitions"] == 3


def test_save_recording_rejects_non_wav(tmp_path: Path) -> None:
    protocol = live_calibration_protocol()
    session_id, _ = create_calibration_session(protocol, tmp_path)
    try:
        save_calibration_recording(
            tmp_path, session_id, 0, protocol.prompts[0], b"not a wav", 48000
        )
    except ValueError as exc:
        assert "WAV" in str(exc)
    else:
        raise AssertionError("expected invalid recording to be rejected")
