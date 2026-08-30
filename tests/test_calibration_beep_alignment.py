import io
import wave
from pathlib import Path

import numpy as np

from phonoweave.calibration_beep_alignment import (
    ACCENT_FREQUENCY_HZ,
    EXPECTED_BEAT_MS,
    PLAIN_FREQUENCY_HZ,
    analyze_beep_alignment,
    detect_beep_sequence,
    expected_beep_frequencies,
)
from phonoweave.calibration_io import create_calibration_session, save_calibration_recording
from phonoweave.calibration_protocol import live_calibration_protocol


def _synthetic_take(token_count: int, seed: int = 1) -> tuple[bytes, int, list[float]]:
    sample_rate = 48000
    rng = np.random.default_rng(seed)
    beat = EXPECTED_BEAT_MS
    beep_times = [250.0 + index * beat for index in range(token_count + 2)]
    duration_ms = beep_times[-1] + beat + 150.0
    samples = rng.normal(0.0, 0.006, int(round(duration_ms * sample_rate / 1000.0)))

    frequencies = expected_beep_frequencies(token_count)
    for time_ms, frequency in zip(beep_times, frequencies, strict=True):
        start = int(round(time_ms * sample_rate / 1000.0))
        length = int(round(0.060 * sample_rate))
        t = np.arange(length, dtype=np.float64) / sample_rate
        envelope = np.minimum(1.0, np.minimum(np.arange(length) / 80.0, np.arange(length)[::-1] / 80.0))
        end = min(len(samples), start + length)
        usable = end - start
        samples[start:end] += 0.055 * envelope[:usable] * np.sin(2.0 * np.pi * frequency * t[:usable])

    for token_index in range(token_count):
        start_ms = 250.0 + (token_index + 1) * beat + 95.0 + (token_index % 2) * 18.0
        start = int(round(start_ms * sample_rate / 1000.0))
        length = int(round(0.28 * sample_rate))
        end = min(len(samples), start + length)
        if end <= start:
            continue
        t = np.arange(end - start, dtype=np.float64) / sample_rate
        carrier = 0.022 * np.sin(2.0 * np.pi * (180.0 + token_index * 19.0) * t)
        formant = 0.018 * np.sin(2.0 * np.pi * (1350.0 + token_index * 80.0) * t)
        noise = rng.normal(0.0, 0.014, end - start)
        envelope = np.hanning((end - start) * 2)[: end - start]
        samples[start:end] += envelope * (carrier + formant + noise)

    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return output.getvalue(), sample_rate, beep_times


def test_expected_frequency_pattern_matches_recorder() -> None:
    assert expected_beep_frequencies(4) == (
        ACCENT_FREQUENCY_HZ,
        ACCENT_FREQUENCY_HZ,
        PLAIN_FREQUENCY_HZ,
        PLAIN_FREQUENCY_HZ,
        PLAIN_FREQUENCY_HZ,
        ACCENT_FREQUENCY_HZ,
    )


def test_detector_recovers_beep_sequence_from_speech_like_take(tmp_path: Path) -> None:
    wav_bytes, sample_rate, expected = _synthetic_take(5, seed=7)
    path = tmp_path / "take.wav"
    path.write_bytes(wav_bytes)
    from phonoweave.audio import read_wav

    samples, sr = read_wav(path)
    sequence = detect_beep_sequence(samples, sr, 5)
    assert len(sequence.events) == 7
    assert sequence.median_interval_ms is not None
    assert abs(sequence.median_interval_ms - EXPECTED_BEAT_MS) < 30.0
    for event, expected_ms in zip(sequence.events, expected, strict=True):
        assert abs(event.time_ms - expected_ms) < 45.0


def test_session_alignment_maps_target_occurrences_to_token_cues(tmp_path: Path) -> None:
    protocol = live_calibration_protocol()
    session_id, session_dir = create_calibration_session(protocol, tmp_path)
    prompt = protocol.prompts[0]
    token_count = len(prompt.spoken_pattern.split())
    wav_bytes, sample_rate, beep_times = _synthetic_take(token_count, seed=11)
    save_calibration_recording(
        tmp_path,
        session_id,
        0,
        prompt,
        wav_bytes,
        sample_rate,
    )

    result = analyze_beep_alignment(session_dir)
    assert result["summary"]["aligned"] == 1
    assert result["summary"]["failed"] == 0
    item = result["alignments"][0]
    assert item["detected_beeps"] == token_count + 2
    target_indices = [
        index
        for index, token in enumerate(prompt.spoken_pattern.split())
        if token == prompt.syllable
    ]
    target_cues = item["target_cues"]
    assert len(target_cues) == len(target_indices) == 2
    for cue, token_index in zip(target_cues, target_indices, strict=True):
        assert cue["token_index"] == token_index
        assert abs(cue["cue_ms"] - beep_times[token_index + 1]) < 45.0

    output = session_dir / "analysis" / "calibration_beep_alignment_v0.3.json"
    assert output.is_file()
