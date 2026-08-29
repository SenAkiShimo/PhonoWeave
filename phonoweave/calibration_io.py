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
            }
            for prompt in protocol.prompts
        ],
    }


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


def recording_filename(index: int, prompt: CalibrationPrompt) -> str:
    return f"{index + 1:03d}_{prompt.base_unit}_{prompt.context_family}.wav"


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
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("recording is not a WAV file")
    session_dir = session_root.expanduser().resolve() / session_id
    recordings = session_dir / "recordings"
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
        "spoken_pattern": prompt.spoken_pattern,
        "sample_rate": sample_rate,
        "wav": output.name,
    }
    (recordings / f"{output.stem}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output
