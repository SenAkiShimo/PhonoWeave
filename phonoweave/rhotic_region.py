from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, AudioSegment, consonant_segment, slice_segment
from .contrast import balanced_accuracy, nearest_centroid_predict
from .mandarin import collect_observations, context_for
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .rhotic import RhoticFeatures, _peak_frequency, _periodicity, _spectrum


_CONTEXTS = ("plain", "front", "rounded")
_WINDOWS = (
    (0.00, 0.20),
    (0.20, 0.40),
    (0.40, 0.60),
    (0.60, 0.80),
    (0.80, 1.00),
)


@dataclass(frozen=True)
class RegionSample:
    subbank: str
    context: str
    features: RhoticFeatures


@dataclass(frozen=True)
class RegionResult:
    start: float
    end: float
    samples: int
    three_way: float | None
    three_way_by_subbank: dict[str, float]
    pairwise: dict[str, float | None]
    pairwise_by_subbank: dict[str, dict[str, float]]
    periodicity_means: dict[str, float]
    centroid_means: dict[str, float]


def _subbank_name(root: Path, oto_path: Path) -> str:
    directory = oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def _features(segment: AudioSegment) -> RhoticFeatures:
    freqs, power = _spectrum(segment.samples, segment.sample_rate)
    mask = freqs >= 300.0
    band_freqs = freqs[mask]
    band_power = power[mask]
    total = float(np.sum(band_power))
    if total <= 1e-16:
        raise AudioReadError("rhotic region has no usable spectral energy")

    weights = band_power / total
    centroid = float(np.sum(band_freqs * weights))
    spread = float(np.sqrt(np.sum(((band_freqs - centroid) ** 2) * weights)))
    flatness = float(np.exp(np.mean(np.log(band_power))) / np.mean(band_power))
    threshold = min(6000.0, segment.sample_rate * 0.22)
    high_ratio = float(np.sum(band_power[band_freqs >= threshold]) / total)
    periodicity = _periodicity(segment.samples, segment.sample_rate)
    f2 = _peak_frequency(freqs, power, 700.0, 2200.0)
    f3 = _peak_frequency(freqs, power, max(1800.0, f2 + 250.0), 3800.0)

    return RhoticFeatures(
        centroid_hz=centroid,
        spread_hz=spread,
        spectral_flatness=flatness,
        high_band_ratio=high_ratio,
        periodicity=periodicity,
        f2_hz=f2,
        f3_hz=f3,
        f3_minus_f2_hz=max(0.0, f3 - f2),
    )


def _matrix(samples: list[RegionSample]) -> np.ndarray:
    return np.vstack([sample.features.vector() for sample in samples])


def _labels(samples: list[RegionSample], contexts: tuple[str, ...]) -> np.ndarray:
    mapping = {context: index for index, context in enumerate(contexts)}
    return np.array([mapping[sample.context] for sample in samples], dtype=np.int8)


def _cross(
    grouped: dict[str, list[RegionSample]],
    contexts: tuple[str, ...],
) -> tuple[float | None, dict[str, float]]:
    names = sorted(grouped)
    scores: dict[str, float] = {}
    for held_out in names:
        train = [
            sample
            for name in names
            if name != held_out
            for sample in grouped[name]
            if sample.context in contexts
        ]
        test = [sample for sample in grouped[held_out] if sample.context in contexts]
        if not train or not test:
            continue
        train_labels = _labels(train, contexts)
        test_labels = _labels(test, contexts)
        if len(np.unique(train_labels)) < len(contexts) or len(np.unique(test_labels)) < len(contexts):
            continue
        predicted = nearest_centroid_predict(_matrix(train), train_labels, _matrix(test))
        scores[held_out] = balanced_accuracy(test_labels, predicted)

    if not scores:
        return None, scores
    return float(np.mean(list(scores.values()))), scores


def _means(samples: list[RegionSample], field: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for context in _CONTEXTS:
        values = [getattr(sample.features, field) for sample in samples if sample.context == context]
        if values:
            output[context] = float(np.mean(values))
    return output


def analyze_rhotic_regions(root: Path) -> list[RegionResult]:
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)

    source = []
    for observation in observations:
        if observation.base_unit != "r":
            continue
        context = context_for("r", observation.final)
        if context not in _CONTEXTS:
            continue
        try:
            whole = consonant_segment(observation.entry, edge_trim=0.04)
        except (AudioReadError, ValueError):
            continue
        source.append((observation, context, whole))

    results: list[RegionResult] = []
    pairs = (("plain", "front"), ("plain", "rounded"), ("front", "rounded"))

    for start, end in _WINDOWS:
        grouped: dict[str, list[RegionSample]] = defaultdict(list)
        all_samples: list[RegionSample] = []
        for observation, context, whole in source:
            try:
                segment = slice_segment(whole, start, end)
                features = _features(segment)
            except (AudioReadError, ValueError):
                continue
            sample = RegionSample(
                subbank=_subbank_name(root, observation.entry.oto_path),
                context=context,
                features=features,
            )
            grouped[sample.subbank].append(sample)
            all_samples.append(sample)

        three_way, three_by = _cross(grouped, _CONTEXTS)
        pairwise: dict[str, float | None] = {}
        pairwise_by: dict[str, dict[str, float]] = {}
        for left, right in pairs:
            key = f"{left}_vs_{right}"
            score, details = _cross(grouped, (left, right))
            pairwise[key] = score
            pairwise_by[key] = details

        results.append(
            RegionResult(
                start=start,
                end=end,
                samples=len(all_samples),
                three_way=three_way,
                three_way_by_subbank=three_by,
                pairwise=pairwise,
                pairwise_by_subbank=pairwise_by,
                periodicity_means=_means(all_samples, "periodicity"),
                centroid_means=_means(all_samples, "centroid_hz"),
            )
        )

    return results


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.rhotic_region")
    parser.add_argument("voicebank", type=Path)
    args = parser.parse_args()

    results = analyze_rhotic_regions(args.voicebank)
    print("Rhotic time-region sweep")
    print("Cross-subbank balanced accuracy by relative OTO consonant region.")
    print("Three-way chance = 0.333; pairwise chance = 0.500.")
    print()

    for result in results:
        print(f"{int(result.start * 100):02d}-{int(result.end * 100):02d}%: samples={result.samples}")
        print(f"  three-way: {_fmt(result.three_way)}")
        for key in ("plain_vs_front", "plain_vs_rounded", "front_vs_rounded"):
            details = result.pairwise_by_subbank[key]
            held = ", ".join(f"{name}={score:.3f}" for name, score in details.items())
            suffix = f" [{held}]" if held else ""
            print(f"  {key}: {_fmt(result.pairwise[key])}{suffix}")
        periodicity = ", ".join(
            f"{context}={result.periodicity_means.get(context, float('nan')):.3f}"
            for context in _CONTEXTS
        )
        centroid = ", ".join(
            f"{context}={result.centroid_means.get(context, float('nan')):.0f}"
            for context in _CONTEXTS
        )
        print(f"  periodicity: {periodicity}")
        print(f"  centroid_hz: {centroid}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
