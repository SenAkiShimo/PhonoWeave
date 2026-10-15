"""Non-destructive capture preflight and F0 screening regression tests."""
import math
import wave
import numpy as np
import pytest

from phonoweave.synthesis_relevance_capture_screen import analyze_pcm
from phonoweave.synthesis_relevance_controlled_capture import trial_capture


def _tone(frames, device=None, freq=233.08, amp=0.35):
    x = np.arange(frames, dtype=float) / 48000
    return np.rint(32767 * amp * np.sin(2 * np.pi * freq * x)).astype("<i2").tobytes()


def test_periodic_tone_estimates_f0_without_clipping():
    result = analyze_pcm(_tone(48000 * 2), target_pitch_hz=233.08)
    assert abs(result["estimated_f0_median_hz"] - 233.08) < 4
    assert result["clipped_sample_count"] == 0
    assert result["peak_dbfs"] < -3
    assert result["screen_status"] == "screen_pass"


def test_silence_flags_unvoiced():
    result = analyze_pcm(b"\0\0" * 48000, target_pitch_hz=233.08)
    assert "no_reliable_voiced_segment" in result["flags"]
    assert result["estimated_f0_median_hz"] is None


def test_clipping_flags():
    result = analyze_pcm((32767).to_bytes(2, "little", signed=True) * 48000)
    assert result["clipped_sample_count"] == 48000
    assert "digital_clipping" in result["flags"]


def test_four_trial_wavs_never_touch_manifest(tmp_path):
    root = tmp_path / "trials"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("unchanged")
    countdowns = []
    result = trial_capture(root, duration=0.5,
                           recorder=_tone, ask=lambda _: "",
                           countdown=lambda: countdowns.append(True))
    assert len(result) == 4
    assert len(countdowns) == 4
    assert manifest.read_text() == "unchanged"
    assert sorted(p.name for p in root.glob("*.wav")) == [
        "sha.wav", "shai.wav", "shua.wav", "shuai.wav"]
    with wave.open(str(root / "sha.wav"), "rb") as audio:
        assert audio.getframerate() == 48000
        assert audio.getnchannels() == 1
        assert audio.getnframes() == 24000
    with pytest.raises(FileExistsError, match="already exists"):
        trial_capture(root, duration=0.5, recorder=_tone, ask=lambda _: "")


def test_trial_quit_before_recording(tmp_path):
    assert trial_capture(tmp_path, ask=lambda _: "q", countdown=lambda: None) == []
    assert not list(tmp_path.glob("*.wav"))
