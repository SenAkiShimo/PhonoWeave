from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phonoweave import inventory
from phonoweave.audio import AudioSegment
from phonoweave.cli import build_parser
from phonoweave.splice import _splice_segments, splice_relevance_test

SAMPLE_RATE = 44100


def _band_noise(rng: np.random.Generator, count: int, low: float, high: float) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(count))
    freqs = np.fft.rfftfreq(count, 1.0 / SAMPLE_RATE)
    spectrum[(freqs < low) | (freqs > high)] = 0
    signal = np.fft.irfft(spectrum, count)
    return signal / (np.std(signal) + 1e-9)


def _write_affricate(path: Path, rng: np.random.Generator, band: tuple[float, float]) -> float:
    closure = 0.0005 * rng.standard_normal(int(0.025 * SAMPLE_RATE))
    burst_count = int(0.006 * SAMPLE_RATE)
    burst = _band_noise(rng, burst_count, *band) * 0.5 * np.hanning(burst_count)
    friction_count = int(0.050 * SAMPLE_RATE)
    ramp = np.minimum(1.0, np.linspace(0.0, 1.0, friction_count) * 6.0)
    friction = _band_noise(rng, friction_count, *band) * 0.12 * ramp
    t = np.arange(int(0.25 * SAMPLE_RATE)) / SAMPLE_RATE
    vowel = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 12)) * 0.12
    vowel = vowel * np.minimum(1.0, t * 60.0)
    signal = np.concatenate([closure, burst, friction, vowel])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
    return 1000.0 * (len(closure) + burst_count + friction_count) / SAMPLE_RATE


def _voicebank(root: Path, plain_band: tuple[float, float], rounded_band: tuple[float, float]) -> None:
    rng = np.random.default_rng(1)
    for bank in ("C4", "C5", "C6"):
        directory = root / bank
        directory.mkdir(parents=True)
        rows = []
        for final in ("a", "ai", "an", "e", "en", "i", "u", "ua", "uo", "ui"):
            band = rounded_band if final.startswith("u") else plain_band
            for repeat in range(2):
                name = f"zh{final}{repeat}.wav"
                preutterance = _write_affricate(directory / name, rng, band)
                rows.append(
                    f"{name}=zh{final},0,{preutterance - 20:.1f},-300,{preutterance:.1f},{preutterance / 3:.1f}"
                )
        (directory / "oto.ini").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_affricate_splice_detects_a_context_dependent_onset(tmp_path: Path) -> None:
    _voicebank(tmp_path, plain_band=(4500, 9000), rounded_band=(2000, 5000))
    result = splice_relevance_test(tmp_path, "zh")

    assert result.targets == 24
    assert result.mean_delta is not None and result.mean_delta > 0
    assert result.permutation_p is not None and result.permutation_p < 0.05


def test_affricate_splice_stays_quiet_without_a_difference(tmp_path: Path) -> None:
    _voicebank(tmp_path, plain_band=(4500, 9000), rounded_band=(4500, 9000))
    result = splice_relevance_test(tmp_path, "zh")

    assert result.targets == 24
    assert result.permutation_p is not None and result.permutation_p > 0.05


def test_splice_segments_use_detected_frication_for_affricates(monkeypatch) -> None:
    friction = AudioSegment(np.zeros(4000), SAMPLE_RATE, 20.0, 110.0)
    monkeypatch.setattr(
        "phonoweave.splice.detect_affricate_frication",
        lambda entry: SimpleNamespace(frication=friction),
    )

    core, late = _splice_segments(SimpleNamespace(), "ch")

    assert core.start_ms == pytest.approx(29.0)
    assert core.end_ms == pytest.approx(69.5)
    assert late.start_ms == pytest.approx(69.5)
    assert late.end_ms == pytest.approx(110.0)


def _acoustic(balanced_accuracy: float, p_values: tuple[float, ...], distance: float):
    return SimpleNamespace(
        cross_subbank_balanced_accuracy=balanced_accuracy,
        mean_distance=distance,
        subbanks=[SimpleNamespace(permutation_p=value) for value in p_values],
    )


def _splice(delta: float, p: float, layers: tuple[tuple[float, float], ...]):
    return SimpleNamespace(
        mean_delta=delta,
        mean_relative_delta=0.2,
        permutation_p=p,
        subbanks=[SimpleNamespace(mean_delta=d, permutation_p=value) for d, value in layers],
    )


def test_affricate_with_acoustic_and_splice_support_is_split(monkeypatch) -> None:
    monkeypatch.setattr(inventory, "analyze_affricate_contrast", lambda root, base: _acoustic(0.8, (0.01, 0.02, 0.3), 0.9))
    monkeypatch.setattr(
        inventory,
        "splice_relevance_test",
        lambda root, base: _splice(0.4, 0.01, ((0.4, 0.01), (0.3, 0.03), (0.2, 0.4))),
    )

    decision = inventory._affricate_decision(Path("/tmp/bank"), "zh")

    assert decision.decision == "split_recommended"
    assert decision.acoustic_evidence == "supported"
    assert decision.synthesis_evidence == "supported_under_proxy"
    assert decision.confidence == "high"
    assert "significant_positive_pitch_layers=2/3" in decision.notes


def test_affricate_without_splice_support_stays_unresolved(monkeypatch) -> None:
    monkeypatch.setattr(inventory, "analyze_affricate_contrast", lambda root, base: _acoustic(0.8, (0.01, 0.02), 0.9))
    monkeypatch.setattr(inventory, "splice_relevance_test", lambda root, base: _splice(-0.1, 1.0, ((-0.1, 1.0),)))

    decision = inventory._affricate_decision(Path("/tmp/bank"), "c")

    assert decision.decision == "unresolved"
    assert decision.acoustic_evidence == "supported"
    assert decision.synthesis_evidence == "split_not_supported_under_proxy"


def test_affricate_with_weak_acoustics_skips_the_splice_test(monkeypatch) -> None:
    monkeypatch.setattr(inventory, "analyze_affricate_contrast", lambda root, base: _acoustic(0.55, (0.4, 0.6), 0.2))

    def fail(root, base):
        raise AssertionError("splice test should not run")

    monkeypatch.setattr(inventory, "splice_relevance_test", fail)

    decision = inventory._affricate_decision(Path("/tmp/bank"), "q")

    assert decision.decision == "unresolved"
    assert decision.acoustic_evidence == "weak_or_inconsistent"
    assert decision.synthesis_evidence == "not_tested"


def test_splice_cli_accepts_affricates() -> None:
    parser = build_parser()
    for base in ("zh", "ch", "z", "c", "j", "q"):
        assert parser.parse_args(["splice-test", "/tmp/voicebank", "--base", base]).base == base
