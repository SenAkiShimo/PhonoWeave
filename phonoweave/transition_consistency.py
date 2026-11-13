from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from .audio import AudioReadError, read_wav
from .mandarin import collect_observations, structure_for
from .nasal_relevance import (
    _alias_role,
    _holm_adjust,
    _identity_family,
    _oto_set_name,
    _resolve_candidates,
    _segment_key,
    _Candidate,
)
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


FAMILIES = ("i_series", "u_series", "other")
LEVELS = (0.25, 0.5, 0.75)

FRAME_MS = 25.0
HOP_MS = 5.0
CONSONANT_CORE = (0.2, 0.5)
LEAD_FRACTION = 0.9
TRANSITION_END_MS = 60.0
VOWEL_WINDOW_MS = (60.0, 100.0)
MIN_CONSONANT_MS = 50.0
LIFTER_MS = 1.2
MIN_CONTRAST_DB = 3.0
MIN_PER_FAMILY_LAYER = 3
BAND_COUNT = 24
BAND_RANGE_HZ = (150.0, 4500.0)


@dataclass(frozen=True)
class TransitionSample:
    oto_set: str
    family: str
    final: str
    alias: str
    residuals: dict[float, np.ndarray]


@dataclass(frozen=True)
class TransitionPairResult:
    family_a: str
    family_b: str
    level: float
    layers: int
    positive_layers: int
    targets_a: int
    targets_b: int
    statistic: float | None
    p: float | None
    p_holm: float | None


@dataclass(frozen=True)
class TransitionReport:
    base_unit: str
    samples: int
    skipped: int
    skip_reasons: dict[str, int]
    pairs: tuple[TransitionPairResult, ...]
    verdict: str


def _band_centers(sample_rate: int) -> np.ndarray:
    upper = min(BAND_RANGE_HZ[1], sample_rate * 0.45)
    return np.geomspace(BAND_RANGE_HZ[0], upper, BAND_COUNT)


def frame_envelope(frame: np.ndarray, sample_rate: int) -> np.ndarray:
    """Pitch-independent log spectral envelope on log-spaced bands (dB, level removed)."""
    frame = np.asarray(frame, dtype=np.float64)
    frame = (frame - np.mean(frame)) * np.hanning(len(frame))
    n_fft = 2048
    while n_fft < len(frame):
        n_fft *= 2
    log_mag = np.log(np.abs(np.fft.rfft(frame, n_fft)) + 1e-9)
    cepstrum = np.fft.irfft(log_mag, n_fft)
    keep = max(2, int(round(LIFTER_MS * sample_rate / 1000.0)))
    lifter = np.zeros(n_fft)
    lifter[: keep + 1] = 1.0
    lifter[-keep:] = 1.0
    smooth = np.fft.rfft(cepstrum * lifter, n_fft).real
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    db = (20.0 / np.log(10.0)) * np.interp(_band_centers(sample_rate), freqs, smooth)
    return db - float(np.mean(db))


def transition_residuals(
    samples: np.ndarray,
    sample_rate: int,
    boundary_ms: float,
    consonant_ms: float,
    levels: tuple[float, ...] = LEVELS,
) -> tuple[dict[float, np.ndarray] | None, str | None]:
    """Off-axis envelope residual at fixed degrees of consonant-to-vowel progress.

    Each sample is projected onto the line between its own consonant state and
    its own vowel state, so vowel identity and level are removed by construction.
    The consonant state is read from the core of the sample's own consonant
    region, so short and long consonants are treated alike.
    """
    if consonant_ms < MIN_CONSONANT_MS:
        return None, "short_consonant"
    half = int(round(FRAME_MS * sample_rate / 2000.0))
    offsets = np.arange(
        -LEAD_FRACTION * consonant_ms, VOWEL_WINDOW_MS[1] + 1e-9, HOP_MS
    )
    envelopes: list[np.ndarray] = []
    for offset in offsets:
        center = int(round((boundary_ms + offset) * sample_rate / 1000.0))
        if center - half < 0 or center + half > len(samples):
            return None, "window_outside_wav"
        envelopes.append(frame_envelope(samples[center - half : center + half], sample_rate))
    env = np.array(envelopes)

    core_start = -consonant_ms * (1.0 - CONSONANT_CORE[0])
    core_end = -consonant_ms * (1.0 - CONSONANT_CORE[1])
    core = (offsets >= core_start) & (offsets <= core_end)
    vowel_mask = (offsets >= VOWEL_WINDOW_MS[0]) & (offsets <= VOWEL_WINDOW_MS[1])
    if core.sum() < 2:
        return None, "short_consonant"
    consonant = env[core].mean(axis=0)
    vowel = env[vowel_mask].mean(axis=0)
    axis = vowel - consonant
    norm2 = float(np.dot(axis, axis))
    if np.sqrt(norm2 / len(axis)) < MIN_CONTRAST_DB:
        return None, "weak_contrast"

    mask = (offsets >= core_end) & (offsets <= TRANSITION_END_MS)
    path = env[mask] - consonant
    progress = path @ axis / norm2
    residual = path - np.outer(progress, axis)

    result: dict[float, np.ndarray] = {}
    for level in levels:
        crossing = np.flatnonzero(progress >= level)
        if len(crossing) == 0:
            return None, "level_not_reached"
        k = int(crossing[0])
        if k == 0:
            return None, "level_before_window"
        span = progress[k] - progress[k - 1]
        weight = 0.0 if span <= 1e-12 else (level - progress[k - 1]) / span
        result[level] = residual[k - 1] * (1.0 - weight) + residual[k] * weight
    return result, None


def _distance_matrix(vectors: np.ndarray) -> np.ndarray:
    diff = vectors[:, None, :] - vectors[None, :, :]
    return np.sqrt(np.mean(diff * diff, axis=2))


@dataclass
class _Layer:
    distances: np.ndarray
    labels: np.ndarray


def _layer_statistic(layer: _Layer, labels: np.ndarray) -> float:
    a = labels.astype(np.float64)
    b = 1.0 - a
    count_a = float(a.sum())
    count_b = float(b.sum())
    d = layer.distances
    between = float(a @ d @ b) / (count_a * count_b)
    within_a = float(a @ d @ a) / (count_a * (count_a - 1.0))
    within_b = float(b @ d @ b) / (count_b * (count_b - 1.0))
    return between - 0.5 * (within_a + within_b)


def _pooled(layers: list[_Layer], labels: list[np.ndarray]) -> float:
    weights = [float(l.sum()) * float((~l).sum()) for l in labels]
    stats = [_layer_statistic(layer, lab) for layer, lab in zip(layers, labels)]
    return float(np.average(stats, weights=weights))


def _pair_level(
    grouped: dict[str, list[TransitionSample]],
    family_a: str,
    family_b: str,
    level: float,
    permutations: int,
    seed: int,
) -> TransitionPairResult:
    layers: list[_Layer] = []
    total_a = total_b = 0
    for oto_set in sorted(grouped):
        rows = [s for s in grouped[oto_set] if s.family in (family_a, family_b)]
        labels = np.array([s.family == family_a for s in rows], dtype=bool)
        if labels.sum() < MIN_PER_FAMILY_LAYER or (~labels).sum() < MIN_PER_FAMILY_LAYER:
            continue
        vectors = np.array([s.residuals[level] for s in rows])
        layers.append(_Layer(_distance_matrix(vectors), labels))
        total_a += int(labels.sum())
        total_b += int((~labels).sum())

    if not layers:
        return TransitionPairResult(
            family_a, family_b, level, 0, 0, 0, 0, None, None, None
        )

    observed_labels = [layer.labels for layer in layers]
    observed = _pooled(layers, observed_labels)
    positive = sum(
        1 for layer in layers if _layer_statistic(layer, layer.labels) > 0.0
    )

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        shuffled = [rng.permutation(layer.labels) for layer in layers]
        if _pooled(layers, shuffled) >= observed:
            exceed += 1
    p = (exceed + 1) / (permutations + 1)
    return TransitionPairResult(
        family_a, family_b, level, len(layers), positive, total_a, total_b,
        observed, float(p), None,
    )


def _verdict(pairs: tuple[TransitionPairResult, ...]) -> str:
    tested = [pair for pair in pairs if pair.p is not None]
    if not tested:
        return "insufficient_data"
    by_pair: dict[tuple[str, str], list[TransitionPairResult]] = defaultdict(list)
    for pair in tested:
        by_pair[(pair.family_a, pair.family_b)].append(pair)
    for rows in by_pair.values():
        hits = [
            row for row in rows
            if row.p_holm is not None
            and row.p_holm < 0.05
            and row.positive_layers * 2 > row.layers
        ]
        if len(hits) >= 2:
            return "consistent_transition_difference"
    return "no_difference_detected"


def _build_samples(root: Path, base_unit: str) -> tuple[list[TransitionSample], Counter]:
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)

    candidates: list[_Candidate] = []
    for observation in observations:
        structure = structure_for(observation)
        if structure.onset != base_unit:
            continue
        candidates.append(
            _Candidate(
                oto_set=_oto_set_name(root, observation.entry),
                role=_alias_role(observation.entry.alias, affixes),
                family=_identity_family(structure.final),
                final=structure.final,
                alias=observation.entry.alias,
                entry=observation.entry,
            )
        )
    resolved, _, _ = _resolve_candidates(candidates)

    skipped: Counter = Counter()
    samples: list[TransitionSample] = []
    cache: dict[Path, tuple[np.ndarray, int]] = {}
    for candidate in resolved:
        if candidate.role != "internal" or candidate.family not in FAMILIES:
            continue
        entry = candidate.entry
        if entry.preutterance <= 0:
            skipped["no_preutterance"] += 1
            continue
        try:
            if entry.wav_path not in cache:
                cache[entry.wav_path] = read_wav(entry.wav_path)
            audio, rate = cache[entry.wav_path]
        except AudioReadError:
            skipped["unreadable_wav"] += 1
            continue
        if rate < 16000:
            skipped["sample_rate_too_low"] += 1
            continue
        residuals, reason = transition_residuals(
            audio, rate, entry.offset + entry.preutterance, entry.preutterance
        )
        if residuals is None:
            skipped[reason or "unknown"] += 1
            continue
        samples.append(
            TransitionSample(
                oto_set=candidate.oto_set,
                family=candidate.family,
                final=candidate.final,
                alias=candidate.alias,
                residuals=residuals,
            )
        )
    return samples, skipped


def analyze_samples(
    base_unit: str,
    samples: list[TransitionSample],
    skipped: Counter | None = None,
    permutations: int = 4000,
    seed: int = 57011,
) -> TransitionReport:
    grouped: dict[str, list[TransitionSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.oto_set].append(sample)

    pairs: list[TransitionPairResult] = []
    for family_a, family_b in combinations(FAMILIES, 2):
        for level in LEVELS:
            pairs.append(
                _pair_level(grouped, family_a, family_b, level, permutations, seed)
            )
    adjusted = _holm_adjust([pair.p for pair in pairs])
    pairs = [
        TransitionPairResult(
            pair.family_a, pair.family_b, pair.level, pair.layers,
            pair.positive_layers, pair.targets_a, pair.targets_b,
            pair.statistic, pair.p, holm,
        )
        for pair, holm in zip(pairs, adjusted)
    ]
    reasons = dict(skipped or {})
    result_pairs = tuple(pairs)
    return TransitionReport(
        base_unit=base_unit,
        samples=len(samples),
        skipped=sum(reasons.values()),
        skip_reasons=reasons,
        pairs=result_pairs,
        verdict=_verdict(result_pairs),
    )


def paired_difference_reports(
    samples_by_onset: dict[str, list[TransitionSample]],
    permutations: int = 4000,
    seed: int = 57011,
) -> list[TransitionReport]:
    """Family structure of the residual difference between two onsets on the same finals.

    Vowel effects shared by both onsets cancel in the difference, so a family
    structure that remains means the two onsets depend on the vowel context
    differently.
    """
    reports: list[TransitionReport] = []
    for left, right in combinations(sorted(samples_by_onset), 2):
        def keyed(rows: list[TransitionSample]) -> dict[tuple[str, str], list[TransitionSample]]:
            grouped: dict[tuple[str, str], list[TransitionSample]] = defaultdict(list)
            for row in rows:
                grouped[(row.oto_set, row.final)].append(row)
            return grouped

        a = keyed(samples_by_onset[left])
        b = keyed(samples_by_onset[right])
        diffs: list[TransitionSample] = []
        for key in sorted(set(a) & set(b)):
            residuals = {
                level: np.mean([r.residuals[level] for r in a[key]], axis=0)
                - np.mean([r.residuals[level] for r in b[key]], axis=0)
                for level in LEVELS
            }
            first = a[key][0]
            diffs.append(
                TransitionSample(first.oto_set, first.family, first.final, first.alias, residuals)
            )
        reports.append(
            analyze_samples(f"{left}-{right}", diffs, Counter(), permutations, seed)
        )
    return reports


def collect_samples(root: Path, base_unit: str) -> tuple[list[TransitionSample], Counter]:
    return _build_samples(root.expanduser().resolve(), base_unit)


def transition_consistency(
    root: Path,
    base_unit: str,
    permutations: int = 4000,
    seed: int = 57011,
) -> TransitionReport:
    samples, skipped = _build_samples(root.expanduser().resolve(), base_unit)
    return analyze_samples(base_unit, samples, skipped, permutations, seed)
