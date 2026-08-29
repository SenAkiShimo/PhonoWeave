import wave
from pathlib import Path

import numpy as np

from phonoweave.calibration_analysis import BEAT_MS, PRE_ROLL_MS, _token_expected_ms
from phonoweave.calibration_analysis_v2 import analyze_calibration_session_v2
from phonoweave.calibration_io import create_calibration_session, save_calibration_recording
from phonoweave.calibration_protocol import live_calibration_protocol


def _speech_like_wav(token_count: int, seed: int) -> bytes:
    sample_rate = 48000
    duration_s = (PRE_ROLL_MS + BEAT_MS * (token_count + 2)) / 1000.0
    samples = np.zeros(int(round(duration_s * sample_rate)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for token_index in range(token_count):
        start_ms = _token_expected_ms(token_index) + 36.0
        start = int(round(start_ms * sample_rate / 1000.0))
        length = int(round(0.16 * sample_rate))
        end = min(len(samples), start + length)
        if end <= start:
            continue
        envelope = np.hanning((end - start) * 2)[: end - start]
        carrier = np.sin(
            2.0 * np.pi * (1800.0 + token_index * 250.0)
            * np.arange(end - start)
            / sample_rate
        )
        noise = rng.normal(0.0, 0.12, end - start)
        samples[start:end] += envelope * (0.45 * carrier + noise)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    import io

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return output.getvalue()


def test_v2_uses_shared_take_offset_and_keeps_two_observations(tmp_path: Path) -> None:
    protocol = live_calibration_protocol()
    session_id, session_dir = create_calibration_session(protocol, tmp_path)
    for index in (0, 1):
        prompt = protocol.prompts[index]
        save_calibration_recording(
            tmp_path,
            session_id,
            index,
            prompt,
            _speech_like_wav(len(prompt.spoken_pattern.split()), 200 + index),
            48000,
        )

    result = analyze_calibration_session_v2(session_dir)
    assert result["version"] == "0.2"
    assert result["summary"]["observations"] == 4
    assert result["summary"]["skipped_observations"] == 0
    zh = next(row for row in result["bases"] if row["base_unit"] == "zh")
    assert len(zh["pairwise"]) == 1
    pair = zh["pairwise"][0]
    assert pair["maximum_timing_repeat_spread_ms"] is not None
    assert "minimum_alignment_db_diagnostic_only" in pair
    assert (session_dir / "analysis" / "calibration_screening_v0.2.json").is_file()
    assert (session_dir / "analysis" / "calibration_observations_v0.2.csv").is_file()
