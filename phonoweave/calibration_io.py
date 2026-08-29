from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .calibration_protocol import CalibrationPrompt, CalibrationProtocol


_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def protocol_payload(protocol: CalibrationProtocol) -> dict[str, object]:
    return {
        "name": protocol.name,
        "version": protocol.version,
        "automatic_supplement_rounds": protocol.automatic_supplement_rounds,
        "stop_rule": protocol.stop_rule,
        "prompts": [
            {
                **asdict(prompt),
                "spoken_pattern": prompt.spoken_pattern,
                "tokens": prompt.spoken_pattern.split(),
                "target_occurrences": prompt.target_occurrences,
            }
            for prompt in protocol.prompts
        ],
    }


def prompt_from_payload(payload: dict[str, object]) -> CalibrationPrompt:
    return CalibrationPrompt(
        base_unit=str(payload["base_unit"]),
        context_family=str(payload["context_family"]),
        syllable=str(payload["syllable"]),
        role_scope=str(payload["role_scope"]),
        carrier=str(payload["carrier"]),
        repeats=int(payload.get("repeats", 1)),
    )


def create_calibration_session(
    protocol: CalibrationProtocol,
    root: Path | None = None,
) -> tuple[str, Path]:
    if root is None:
        root = Path.home() / "Downloads" / "PhonoWeaveCalibration"
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_dir = root / session_id
    session_dir.mkdir(parents=False, exist_ok=False)
    (session_dir / "protocol.json").write_text(
        json.dumps(protocol_payload(protocol), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (session_dir / "recordings").mkdir()
    return session_id, session_dir


def load_calibration_session(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    protocol_path = session_dir / "protocol.json"
    recordings_dir = session_dir / "recordings"
    if not protocol_path.is_file() or not recordings_dir.is_dir():
        raise ValueError("selected folder is not a PhonoWeave calibration session")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    prompts = protocol.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("calibration protocol is missing prompts")

    recordings: dict[int, dict[str, object]] = {}
    for metadata_path in sorted(recordings_dir.glob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            index = int(metadata["prompt_index"])
            wav_name = str(metadata["wav"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if index < 0 or index >= len(prompts):
            continue
        wav_path = recordings_dir / wav_name
        if not wav_path.is_file():
            continue
        recordings[index] = {
            "wav": wav_name,
            "sample_rate": metadata.get("sample_rate"),
        }

    return {
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "protocol": protocol,
        "recordings": recordings,
        "completed_indices": sorted(recordings),
    }


def recording_filename(index: int, prompt: CalibrationPrompt) -> str:
    return f"{index + 1:03d}_{prompt.base_unit}_{prompt.context_family}.wav"


def recording_path_for_session(session_dir: Path, index: int) -> Path:
    session = load_calibration_session(session_dir)
    recordings = session["recordings"]
    if not isinstance(recordings, dict) or index not in recordings:
        raise ValueError("recording does not exist")
    item = recordings[index]
    if not isinstance(item, dict):
        raise ValueError("recording metadata is invalid")
    path = session_dir.expanduser().resolve() / "recordings" / str(item["wav"])
    if not path.is_file():
        raise ValueError("recording does not exist")
    return path


def _save_recording_to_dir(
    session_dir: Path,
    index: int,
    prompt: CalibrationPrompt,
    wav_bytes: bytes,
    sample_rate: int,
) -> Path:
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("recording is not a WAV file")
    recordings = session_dir.expanduser().resolve() / "recordings"
    if not recordings.is_dir():
        raise ValueError("calibration session does not exist")
    output = recordings / recording_filename(index, prompt)
    output.write_bytes(wav_bytes)
    metadata = {
        "prompt_index": index,
        "base_unit": prompt.base_unit,
        "context_family": prompt.context_family,
        "syllable": prompt.syllable,
        "role_scope": prompt.role_scope,
        "carrier": prompt.carrier,
        "repetitions": prompt.repeats,
        "target_occurrences": prompt.target_occurrences,
        "spoken_pattern": prompt.spoken_pattern,
        "sample_rate": sample_rate,
        "wav": output.name,
    }
    (recordings / f"{output.stem}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def save_calibration_recording(
    session_root: Path,
    session_id: str,
    index: int,
    prompt: CalibrationPrompt,
    wav_bytes: bytes,
    sample_rate: int,
) -> Path:
    if not _SESSION_RE.fullmatch(session_id):
        raise ValueError("invalid calibration session id")
    return _save_recording_to_dir(
        session_root.expanduser().resolve() / session_id,
        index,
        prompt,
        wav_bytes,
        sample_rate,
    )


def save_calibration_recording_to_session(
    session_dir: Path,
    index: int,
    prompt: CalibrationPrompt,
    wav_bytes: bytes,
    sample_rate: int,
) -> Path:
    return _save_recording_to_dir(
        session_dir,
        index,
        prompt,
        wav_bytes,
        sample_rate,
    )
