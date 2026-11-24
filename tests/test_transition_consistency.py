from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from phonoweave import entrypoint
from phonoweave.transition_consistency import (
    TransitionSample,
    paired_difference_reports,
    analyze_samples,
    frame_envelope,
    transition_residuals,
)

RATE = 32000
BOUNDARY_MS = 200.0


def _resonance(freqs: np.ndarray, center: float, bandwidth: float) -> np.ndarray:
    return 1.0 / np.sqrt(1.0 + ((freqs - center) / bandwidth) ** 2)


def _synth(f0: float, c_formants, v_formants, detour: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    duration = 0.4
    n = int(RATE * duration)
    control_t = np.arange(0.0, duration, 0.005)
    progress = np.clip((control_t - (BOUNDARY_MS - 20) / 1000.0) / 0.06, 0.0, 1.0)
    progress = progress * progress * (3.0 - 2.0 * progress)
    harmonics = np.arange(1, int(5000 / f0) + 1) * f0
    out = np.zeros(n)
    t = np.arange(n) / RATE
    for freq in harmonics:
        amps = np.zeros(len(control_t))
        for i, s in enumerate(progress):
            f1 = c_formants[0] * (1 - s) + v_formants[0] * s
            f2 = c_formants[1] * (1 - s) + v_formants[1] * s + detour * np.sin(np.pi * s)
            amps[i] = _resonance(np.array([freq]), f1, 120.0)[0] * 3.0 + _resonance(
                np.array([freq]), f2, 220.0
            )[0]
        env = np.interp(t, control_t, amps)
        out += env * np.sin(2 * np.pi * freq * t + rng.uniform(0, 6.28))
    out += rng.normal(0, 1e-3, n)
    return out / (np.max(np.abs(out)) + 1e-9)


def test_envelope_is_pitch_independent() -> None:
    def env(f0: float, f1: float) -> np.ndarray:
        audio = _synth(f0, (250, 1000), (f1, 1800), 0.0)
        centre = int(0.35 * RATE)
        return frame_envelope(audio[centre - 400 : centre + 400], RATE)

    same_pitch_gap = np.sqrt(np.mean((env(130, 600) - env(330, 600)) ** 2))
    vowel_gap = np.sqrt(np.mean((env(130, 600) - env(130, 800)) ** 2))
    assert same_pitch_gap < vowel_gap * 0.85


def test_transition_residuals_levels() -> None:
    audio = _synth(150, (250, 1000), (700, 1700), 0.0)
    residuals, reason = transition_residuals(audio, RATE, BOUNDARY_MS, 100.0)
    assert reason is None
    assert set(residuals) == {0.25, 0.5, 0.75}


def test_weak_contrast_is_skipped() -> None:
    audio = _synth(150, (700, 1700), (700, 1700), 0.0)
    residuals, reason = transition_residuals(audio, RATE, BOUNDARY_MS, 100.0)
    assert residuals is None
    assert reason == "weak_contrast"


def _population(detour_a: float, detour_b: float, count: int = 10) -> list[TransitionSample]:
    rng = np.random.default_rng(3)
    samples: list[TransitionSample] = []
    for layer, f0 in enumerate((130.0, 200.0, 310.0)):
        for family, detour in (("u_series", detour_a), ("i_series", detour_b)):
            for k in range(count):
                v = (rng.uniform(500, 800), rng.uniform(1500, 2400))
                audio = _synth(f0, (250, 1000), v, detour, seed=int(rng.integers(1_000_000)))
                residuals, _ = transition_residuals(audio, RATE, BOUNDARY_MS, 100.0)
                if residuals is None:
                    continue
                samples.append(
                    TransitionSample(f"layer{layer}", family, f"{family[0]}{k % 3}", "x", residuals)
                )
    return samples


def test_family_specific_path_is_detected() -> None:
    report = analyze_samples("m", _population(500.0, 0.0), Counter(), permutations=300)
    pair = next(
        p for p in report.pairs
        if {p.family_a, p.family_b} == {"i_series", "u_series"} and p.level == 0.5
    )
    assert pair.layers == 3
    assert pair.positive_layers == 3
    assert pair.p is not None and pair.p < 0.01


def test_identical_paths_are_not_flagged() -> None:
    report = analyze_samples("m", _population(0.0, 0.0), Counter(), permutations=300)
    assert report.verdict == "no_difference_detected"
    pair = next(
        p for p in report.pairs
        if {p.family_a, p.family_b} == {"i_series", "u_series"} and p.level == 0.5
    )
    assert pair.p is not None and pair.p > 0.05


def test_insufficient_data() -> None:
    report = analyze_samples("m", [], Counter(), permutations=10)
    assert report.verdict == "insufficient_data"


def test_transition_report_entrypoint(monkeypatch) -> None:
    called = {}

    def fake_main(argv):
        called["argv"] = argv
        return 0

    monkeypatch.setattr(entrypoint, "transition_main", fake_main)
    monkeypatch.setattr(sys, "argv", ["phonoweave", "transition-report", "/tmp/vb", "--base", "m"])
    assert entrypoint.main() == 0
    assert called["argv"] == ["/tmp/vb", "--base", "m"]


def _paired_population(detour_m: float, detour_n: float) -> dict[str, list[TransitionSample]]:
    rng = np.random.default_rng(11)
    out: dict[str, list[TransitionSample]] = {"m": [], "n": []}
    for layer, f0 in enumerate((130.0, 200.0, 310.0)):
        for family, shift in (("u_series", 1.0), ("i_series", 0.0)):
            for k in range(8):
                v = (rng.uniform(500, 800), rng.uniform(1500, 2400))
                for onset, base in (("m", detour_m), ("n", detour_n)):
                    audio = _synth(f0, (250, 1000), v, base * shift, seed=int(rng.integers(10**6)))
                    residuals, _ = transition_residuals(audio, RATE, BOUNDARY_MS, 100.0)
                    if residuals is None:
                        continue
                    out[onset].append(
                        TransitionSample(f"layer{layer}", family, f"{family[0]}{k}", "x", residuals)
                    )
    return out


def test_paired_difference_cancels_shared_vowel_effect() -> None:
    same = paired_difference_reports(_paired_population(500.0, 500.0), permutations=300)
    assert same[0].verdict == "no_difference_detected"


def test_paired_difference_detects_onset_specific_dependence() -> None:
    different = paired_difference_reports(_paired_population(500.0, 0.0), permutations=300)
    pair = next(
        p for p in different[0].pairs
        if {p.family_a, p.family_b} == {"i_series", "u_series"} and p.level == 0.5
    )
    assert pair.p is not None and pair.p < 0.01
