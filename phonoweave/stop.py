from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, read_wav
from .mandarin import MandarinObservation, collect_observations, normalize_alias, structure_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


_SUPPORTED = {
    "b": ("bilabial", False),
    "p": ("bilabial", True),
    "d": ("alveolar", False),
    "t": ("alveolar", True),
    "g": ("velar", False),
    "k": ("velar", True),
}
_PAIRS = (
    ("bilabial", "b", "p"),
    ("alveolar", "d", "t"),
    ("velar", "g", "k"),
)


@dataclass(frozen=True)
class StopCandidate:
    observation: MandarinObservation
    final: str
    role: str
    oto_set: str


@dataclass(frozen=True)
class StopFeatures:
    release_to_vowel_ms: float
    release_strength: float
    vowel_periodicity: float
    burst_centroid_hz: float
    burst_high_ratio: float


@dataclass(frozen=True)
class StopSample:
    base_unit: str
    place: str
    aspirated: bool
    final: str
    role: str
    oto_set: str
    alias: str
    entry: OtoEntry
    features: StopFeatures


@dataclass(frozen=True)
class StopBaseSummary:
    base_unit: str
    samples: int
    median_release_to_vowel_ms: float | None
    median_release_strength: float | None
    median_vowel_periodicity: float | None
    median_burst_centroid_hz: float | None
    median_burst_high_ratio: float | None


@dataclass(frozen=True)
class StopPlaceContrast:
    place: str
    role: str
    matched_cells: int
    median_aspiration_delta_ms: float | None
    positive_cells: int
    oto_set_median_deltas_ms: dict[str, float]


@dataclass(frozen=True)
class StopDiagnostic:
    voicebank: Path
    candidates: int
    samples: int
    skipped: int
    duplicate_observations_removed: int
    ambiguous_segments_removed: int
    ambiguous_observations_removed: int
    bases: tuple[StopBaseSummary, ...]
    contrasts: tuple[StopPlaceContrast, ...]


def _oto_set_name(root: Path, entry: OtoEntry) -> str:
    directory = entry.oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def _segment_key(entry: OtoEntry) -> tuple[Path, float, float]:
    return (
        entry.wav_path,
        round(entry.offset, 3),
        round(entry.offset + entry.preutterance, 3),
    )


def _role(alias: str, affixes: set[tuple[str, str]]) -> str:
    normalized = normalize_alias(alias, affixes).strip()
    return "initial" if normalized.startswith("-") else "internal"


def _resolve_candidates(
    candidates: list[StopCandidate],
) -> tuple[list[StopCandidate], int, int, int]:
    grouped: dict[tuple[Path, float, float], list[StopCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[_segment_key(candidate.observation.entry)].append(candidate)

    resolved: list[StopCandidate] = []
    duplicate_observations_removed = 0
    ambiguous_segments_removed = 0
    ambiguous_observations_removed = 0
    for rows in grouped.values():
        identities = {
            (
                row.observation.base_unit,
                row.final,
                row.role,
            )
            for row in rows
        }
        if len(identities) > 1:
            ambiguous_segments_removed += 1
            ambiguous_observations_removed += len(rows)
            continue
        ordered = sorted(rows, key=lambda row: row.observation.entry.alias)
        resolved.append(ordered[0])
        duplicate_observations_removed += len(ordered) - 1

    return (
        resolved,
        duplicate_observations_removed,
        ambiguous_segments_removed,
        ambiguous_observations_removed,
    )


def _zscore(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values))
    if scale < 1e-9:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / scale


def _frame_positions(length: int, frame: int, hop: int) -> list[int]:
    if length < frame:
        return []
    return list(range(0, length - frame + 1, hop))


def _periodicity(frame: np.ndarray, sample_rate: int) -> float:
    frame = frame.astype(np.float64, copy=False)
    frame = frame - np.mean(frame)
    energy = float(np.dot(frame, frame))
    if energy < 1e-12:
        return 0.0
    min_lag = max(1, int(sample_rate / 500.0))
    max_lag = min(len(frame) - 2, int(sample_rate / 70.0))
    if max_lag <= min_lag:
        return 0.0
    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        left = frame[:-lag]
        right = frame[lag:]
        denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
        if denom > 1e-12:
            best = max(best, float(np.dot(left, right) / denom))
    return max(0.0, min(best, 1.0))


def _spectral(frame: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    frame = frame.astype(np.float64, copy=False)
    frame = frame - np.mean(frame)
    power = np.abs(np.fft.rfft(frame * np.hanning(len(frame)))) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sample_rate)
    valid = (freqs >= 300.0) & (freqs <= min(8000.0, 0.45 * sample_rate))
    if not np.any(valid):
        return 0.0, 0.0, 0.0
    band_power = power[valid]
    band_freqs = freqs[valid]
    total = float(np.sum(band_power))
    centroid = float(np.sum(band_freqs * band_power) / total)
    high = float(np.sum(band_power[band_freqs >= 2000.0]) / total)
    flatness = float(np.exp(np.mean(np.log(band_power))) / np.mean(band_power))
    return centroid, high, flatness


def _detect_release(
    samples: np.ndarray,
    sample_rate: int,
    start_ms: float,
    end_ms: float,
) -> tuple[float, float]:
    start = max(0, int(round(start_ms * sample_rate / 1000.0)))
    end = min(len(samples), int(round(end_ms * sample_rate / 1000.0)))
    region = samples[start:end]
    frame = max(64, int(round(sample_rate * 0.004)))
    hop = max(1, int(round(sample_rate * 0.001)))
    positions = _frame_positions(len(region), frame, hop)
    if len(positions) < 6:
        raise AudioReadError("stop window is too short for release detection")

    rms_db: list[float] = []
    high_ratio: list[float] = []
    centers_ms: list[float] = []
    for position in positions:
        chunk = region[position : position + frame]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) + 1e-12)
        _, high, _ = _spectral(chunk, sample_rate)
        rms_db.append(20.0 * np.log10(rms))
        high_ratio.append(high)
        centers_ms.append(start_ms + 1000.0 * (position + frame / 2.0) / sample_rate)

    rms_array = np.asarray(rms_db, dtype=np.float64)
    high_array = np.asarray(high_ratio, dtype=np.float64)
    drms = np.diff(rms_array, prepend=rms_array[0])
    dhigh = np.diff(high_array, prepend=high_array[0])
    score = _zscore(drms) + 0.45 * _zscore(dhigh) + 0.20 * _zscore(high_array)
    centers = np.asarray(centers_ms, dtype=np.float64)
    candidate = (centers >= start_ms + 2.0) & (centers <= end_ms - 5.0)
    indices = np.flatnonzero(candidate)
    if not len(indices):
        raise AudioReadError("no usable stop release interval")
    best = int(indices[np.argmax(score[indices])])
    return float(centers[best]), float(score[best] - np.median(score[indices]))


def _detect_vowel_onset(
    samples: np.ndarray,
    sample_rate: int,
    prior_ms: float,
    release_ms: float,
) -> tuple[float, float]:
    search_start_ms = max(release_ms + 2.0, prior_ms - 28.0)
    search_end_ms = prior_ms + 32.0
    start = max(0, int(round(search_start_ms * sample_rate / 1000.0)))
    end = min(len(samples), int(round(search_end_ms * sample_rate / 1000.0)))
    region = samples[start:end]
    frame = max(128, int(round(sample_rate * 0.018)))
    hop = max(1, int(round(sample_rate * 0.002)))
    positions = _frame_positions(len(region), frame, hop)
    if not positions:
        return max(release_ms, prior_ms), 0.0

    rows: list[tuple[float, float, float, float]] = []
    rms_values: list[float] = []
    for position in positions:
        chunk = region[position : position + frame]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) + 1e-12)
        _, _, flatness = _spectral(chunk, sample_rate)
        periodicity = _periodicity(chunk, sample_rate)
        center_ms = search_start_ms + 1000.0 * (position + frame / 2.0) / sample_rate
        rows.append((center_ms, periodicity, flatness, rms))
        rms_values.append(rms)

    rms_floor = max(rms_values) * 0.08
    frame_ms = 1000.0 * frame / sample_rate
    for center_ms, periodicity, flatness, rms in rows:
        if center_ms < release_ms + 5.0:
            continue
        if periodicity >= 0.46 and flatness <= 0.48 and rms >= rms_floor:
            onset = max(release_ms, center_ms - 0.25 * frame_ms)
            return float(onset), periodicity

    best = max(rows, key=lambda row: row[1] - 0.25 * row[2])
    onset = max(release_ms, best[0] - 0.25 * frame_ms)
    if abs(best[0] - prior_ms) <= 28.0 and best[1] >= 0.28:
        return float(onset), float(best[1])
    return max(release_ms, prior_ms), float(best[1])


def _extract(entry: OtoEntry) -> StopFeatures:
    if entry.preutterance <= 0:
        raise AudioReadError("preutterance is not positive")
    samples, sample_rate = read_wav(entry.wav_path)
    start_ms = max(0.0, entry.offset)
    prior_ms = entry.offset + entry.preutterance
    if prior_ms - start_ms < 16.0:
        raise AudioReadError("stop window is too short")

    release_ms, release_strength = _detect_release(
        samples,
        sample_rate,
        start_ms,
        prior_ms,
    )
    vowel_onset_ms, vowel_periodicity = _detect_vowel_onset(
        samples,
        sample_rate,
        prior_ms,
        release_ms,
    )

    burst_start_ms = max(start_ms, release_ms - 1.5)
    burst_end_ms = min(prior_ms + 8.0, release_ms + 4.5)
    start = max(0, int(round(burst_start_ms * sample_rate / 1000.0)))
    end = min(len(samples), int(round(burst_end_ms * sample_rate / 1000.0)))
    if end - start < 32:
        raise AudioReadError("burst slice is too short")
    centroid, high_ratio, _ = _spectral(samples[start:end], sample_rate)
    return StopFeatures(
        release_to_vowel_ms=max(0.0, vowel_onset_ms - release_ms),
        release_strength=release_strength,
        vowel_periodicity=vowel_periodicity,
        burst_centroid_hz=centroid,
        burst_high_ratio=high_ratio,
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _base_summary(base: str, samples: list[StopSample]) -> StopBaseSummary:
    return StopBaseSummary(
        base_unit=base,
        samples=len(samples),
        median_release_to_vowel_ms=_median([row.features.release_to_vowel_ms for row in samples]),
        median_release_strength=_median([row.features.release_strength for row in samples]),
        median_vowel_periodicity=_median([row.features.vowel_periodicity for row in samples]),
        median_burst_centroid_hz=_median([row.features.burst_centroid_hz for row in samples]),
        median_burst_high_ratio=_median([row.features.burst_high_ratio for row in samples]),
    )


def _place_contrasts(samples: list[StopSample]) -> tuple[StopPlaceContrast, ...]:
    contrasts: list[StopPlaceContrast] = []
    for place, unaspirated, aspirated in _PAIRS:
        for role in ("initial", "internal"):
            cells: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for row in samples:
                if row.place != place or row.role != role:
                    continue
                cells[(row.oto_set, row.final, role)][row.base_unit].append(
                    row.features.release_to_vowel_ms
                )

            deltas: list[tuple[str, float]] = []
            for (oto_set, _final, _role), by_base in cells.items():
                left = by_base.get(unaspirated, [])
                right = by_base.get(aspirated, [])
                if not left or not right:
                    continue
                delta = float(np.median(right) - np.median(left))
                deltas.append((oto_set, delta))

            by_oto: dict[str, list[float]] = defaultdict(list)
            for oto_set, delta in deltas:
                by_oto[oto_set].append(delta)
            contrasts.append(
                StopPlaceContrast(
                    place=place,
                    role=role,
                    matched_cells=len(deltas),
                    median_aspiration_delta_ms=_median([delta for _, delta in deltas]),
                    positive_cells=sum(delta > 0.0 for _, delta in deltas),
                    oto_set_median_deltas_ms={
                        name: float(np.median(values))
                        for name, values in sorted(by_oto.items())
                    },
                )
            )
    return tuple(contrasts)


def analyze_stop_diagnostic(root: Path) -> StopDiagnostic:
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)

    candidates: list[StopCandidate] = []
    for observation in observations:
        structure = structure_for(observation)
        if structure.onset not in _SUPPORTED:
            continue
        candidates.append(
            StopCandidate(
                observation=observation,
                final=structure.final,
                role=_role(observation.entry.alias, affixes),
                oto_set=_oto_set_name(root, observation.entry),
            )
        )

    resolved, duplicate_removed, ambiguous_segments, ambiguous_observations = _resolve_candidates(candidates)
    samples: list[StopSample] = []
    skipped = 0
    for candidate in resolved:
        base = candidate.observation.base_unit
        place, aspirated = _SUPPORTED[base]
        try:
            features = _extract(candidate.observation.entry)
        except (AudioReadError, ValueError, FloatingPointError):
            skipped += 1
            continue
        samples.append(
            StopSample(
                base_unit=base,
                place=place,
                aspirated=aspirated,
                final=candidate.final,
                role=candidate.role,
                oto_set=candidate.oto_set,
                alias=candidate.observation.entry.alias,
                entry=candidate.observation.entry,
                features=features,
            )
        )

    by_base: dict[str, list[StopSample]] = defaultdict(list)
    for sample in samples:
        by_base[sample.base_unit].append(sample)
    return StopDiagnostic(
        voicebank=root,
        candidates=len(candidates),
        samples=len(samples),
        skipped=skipped,
        duplicate_observations_removed=duplicate_removed,
        ambiguous_segments_removed=ambiguous_segments,
        ambiguous_observations_removed=ambiguous_observations,
        bases=tuple(_base_summary(base, by_base.get(base, [])) for base in _SUPPORTED),
        contrasts=_place_contrasts(samples),
    )
