from __future__ import annotations

import json
from pathlib import Path
import wave

import numpy as np

from phonoweave.calibration_io import protocol_payload
from phonoweave.calibration_manual_labels import (
    DEV_ANCHOR_TARGETS,
    load_manual_labels,
    save_manual_label,
)
from phonoweave.calibration_protocol import live_calibration_protocol


def _write_wav(path: Path, sample_rate: int = 8000) -> None:
    duration_s = 7.0
    samples = np.zeros(int(sample_rate * duration_s), dtype=np.float64)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((samples * 32767).astype("<i2").tobytes())


def _session(tmp_path: Path) -> Path:
    session = tmp_path / "session"
    recordings = session / "recordings"
    recordings.mkdir(parents=True)
    protocol = protocol_payload(live_calibration_protocol())
    (session / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    for index, prompt in enumerate(protocol["prompts"]):
        wav_name = f"{index + 1:03d}_{prompt['base_unit']}_{prompt['context_family']}.wav"
        _write_wav(recordings / wav_name)
        metadata = {
            "prompt_index": index,
            "wav": wav_name,
            "sample_rate": 8000,
        }
        (recordings / f"{Path(wav_name).stem}.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
    return session


def test_fixed_manual_selection_has_16_unique_targets() -> None:
    assert len(DEV_ANCHOR_TARGETS) == 16
    assert len(set(DEV_ANCHOR_TARGETS)) == 16


def test_manual_label_save_uncertain_and_clear(tmp_path: Path, monkeypatch) -> None:
    session = _session(tmp_path)

    class FakeEvent:
        def __init__(self, time_ms: float) -> None:
            self.time_ms = time_ms

    class FakeSequence:
        events = tuple(FakeEvent(250.0 + 833.333 * i) for i in range(8))

    monkeypatch.setattr(
        "phonoweave.calibration_manual_labels.detect_beep_sequence",
        lambda samples, sample_rate, token_count: FakeSequence(),
    )

    base, context = DEV_ANCHOR_TARGETS[0]
    protocol = json.loads((session / "protocol.json").read_text(encoding="utf-8"))
    prompt_index = next(
        i
        for i, prompt in enumerate(protocol["prompts"])
        if prompt["base_unit"] == base and prompt["context_family"] == context
    )

    saved = save_manual_label(session, prompt_index, 1, 180.0, "ok")
    row = saved["labels"][str(prompt_index)]["occurrences"]["1"]
    assert row["anchor_ms_after_cue"] == 180.0
    assert row["status"] == "ok"

    saved = save_manual_label(session, prompt_index, 1, 180.0, "uncertain")
    row = saved["labels"][str(prompt_index)]["occurrences"]["1"]
    assert row["status"] == "uncertain"

    save_manual_label(session, prompt_index, 1, None, "unset")
    loaded = load_manual_labels(session)
    assert "1" not in loaded["labels"][str(prompt_index)]["occurrences"]
