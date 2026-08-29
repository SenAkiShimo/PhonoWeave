import json
import wave
from pathlib import Path

import numpy as np

from phonoweave.calibration_analysis import (
    BEAT_MS,
    PRE_ROLL_MS,
    _token_expected_ms,
    analyze_calibration_session,
)
from phonoweave.calibration_io import create_calibration_session, save_calibration_recording
from phonoweave.calibration_protocol import live_calibration_protocol


def _speech_like_wav(token_count: int, seed: int) -> bytes:
    sample_rate = 48000
    duration_s = (PRE_ROLL_MS + BEAT_MS * (token_count + 2)) / 1000.0
    samples = np.zeros(int(round(duration_s * sample_rate)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for token_index in range(token_count):
        start_ms = _token_expected_ms(token_index) + 24.0
        start = int(round(start_ms * sample_rate / 1000.0))
        length = int(round(0.18 * sample_rate))
        end = min(len(samples), start + length)
        if end <= start:
            continue
        envelope = np.hanning((end - start) * 2)[: end - start]
        carrier = np.sin(2.0 * np.pi * (2100.0 + token_index * 270.0) * np.arange(end - start) / sample_rate)
        noise = rng.normal(0.0, 0.18, end - start)
        samples[start:end] += envelope * (0.42 * carrier + noise)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2").tobytes()

    import io

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return output.getvalue()


def test_token_timing_matches_recording_count_in() -> None:
    assert _token_expected_ms(0) == PRE_ROLL_MS + BEAT_MS
    assert _token_expected_ms(2) == PRE_ROLL_MS + 3 * BEAT_MS


def test_calibration_analysis_extracts_two_targets_per_prompt(tmp_path: Path) -> None:
    protocol = live_calibration_protocol()
    session_id, session_dir = create_calibration_session(protocol, tmp_path)
    for index in (0, 1):
        prompt = protocol.prompts[index]
        save_calibration_recording(
            tmp_path,
            session_id,
            index,
            prompt,
            _speech_like_wav(len(prompt.spoken_pattern.split()), 100 + index),
            48000,
        )

    result = analyze_calibration_session(session_dir)
    summary = result["summary"]
    assert summary["recorded_prompts"] == 2
    assert summary["observations"] == 4
    assert summary["skipped_observations"] == 0
    assert summary["tested_onsets"] == 1
    assert "s" in summary["untested_onsets"]
    assert "l" in summary["untested_onsets"]

    zh = next(row for row in result["bases"] if row["base_unit"] == "zh")
    assert len(zh["pairwise"]) == 1
    pair = zh["pairwise"][0]
    assert {pair["context_a"], pair["context_b"]} == {"plain", "rounded"}
    assert pair["count_a"] == 2
    assert pair["count_b"] == 2

    output = json.loads(
        (session_dir / "analysis" / "calibration_screening_v0.1.json").read_text(
            encoding="utf-8"
        )
    )
    assert output["interpretation"] == "screening_only_not_split_merge_decision"
    assert (session_dir / "analysis" / "calibration_observations_v0.1.csv").is_file()
