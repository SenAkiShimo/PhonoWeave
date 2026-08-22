from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError, AudioSegment, consonant_segment, slice_segment
from .mandarin import collect_observations, context_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


@dataclass(frozen=True)
class RhoticSpliceSample:
    subbank: str
    context: str
    alias: str
    final: str
    entry: OtoEntry
    core: AudioSegment
    late: AudioSegment


@dataclass(frozen=True)
class RhoticMergeTarget:
    subbank: str
    context: str
    alias: str
    natural_penalty: float
    same_control: float
    merge_substitution: float
    front_substitution: float
    merge_delta: float
    front_delta: float
    separation_delta: float


@dataclass(frozen=True)
class RhoticFrontTarget:
    subbank: str
    alias: str
    natural_penalty: float
    same_control: float
    merged_substitution: float
    delta: float


def _subbank_name(root: Path, entry: OtoEntry) -> str:
    directory = entry.oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def _periodicity(samples: np.ndarray, sample_rate: int) -> float:
    samples = samples.astype(np.float64, copy=False)
    samples = samples - np.mean(samples)
    min_lag = max(1, int(sample_rate / 500.0))
    max_lag = min(len(samples) - 2, int(sample_rate / 70.0))
    if max_lag <= min_lag:
        return 0.0

    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        left = samples[:-lag]
        right = samples[lag:]
        denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
        if denom > 1e-12:
            best = max(best, float(np.dot(left, right) / denom))
    return max(0.0, min(best, 1.0))


def _edge(segment: AudioSegment, side: str, window_ms: float = 24.0) -> tuple[np.ndarray, np.ndarray, float, float]:
    count = max(64, int(round(segment.sample_rate * window_ms / 1000.0)))
    count = min(count, len(segment.samples))
    if count < 64:
        raise AudioReadError("rhotic edge window is too short")

    samples = segment.samples[:count] if side == "start" else segment.samples[-count:]
    samples = samples.astype(np.float64, copy=False)
    samples = samples - np.mean(samples)
    rms = float(np.sqrt(np.mean(samples ** 2)) + 1e-12)
    periodicity = _periodicity(samples, segment.sample_rate)

    power = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / segment.sample_rate)
    return freqs, power, rms, periodicity


def rhotic_boundary_penalty(left: AudioSegment, right: AudioSegment) -> float:
    left_freqs, left_power, left_rms, left_periodicity = _edge(left, "end")
    right_freqs, right_power, right_rms, right_periodicity = _edge(right, "start")

    upper = min(5000.0, left.sample_rate * 0.45, right.sample_rate * 0.45)
    if upper <= 1200.0:
        raise AudioReadError("sample rate is too low for rhotic boundary analysis")

    grid = np.linspace(300.0, upper, 160)
    left_interp = np.interp(grid, left_freqs, left_power)
    right_interp = np.interp(grid, right_freqs, right_power)
    left_norm = left_interp / (np.sum(left_interp) + 1e-18)
    right_norm = right_interp / (np.sum(right_interp) + 1e-18)
    spectral = float(
        np.sqrt(
            np.mean(
                (np.log10(left_norm + 1e-12) - np.log10(right_norm + 1e-12)) ** 2
            )
        )
    )

    energy_db = abs(20.0 * np.log10(left_rms / right_rms))
    periodicity_jump = abs(left_periodicity - right_periodicity)
    return spectral + 0.02 * float(energy_db) + 0.35 * periodicity_jump


def _paired_p(deltas: list[float], seed: int) -> float:
    values = np.asarray(deltas, dtype=np.float64)
    if len(values) == 0:
        return 1.0
    observed = float(np.mean(values))
    if observed <= 0:
        return 1.0

    rng = np.random.default_rng(seed)
    exceed = 0
    permutations = 10000
    for _ in range(permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(values))
        if float(np.mean(values * signs)) >= observed:
            exceed += 1
    return float((exceed + 1) / (permutations + 1))


def _median_penalty(donors: list[RhoticSpliceSample], target: RhoticSpliceSample) -> float:
    scores = [rhotic_boundary_penalty(donor.core, target.late) for donor in donors]
    return float(np.median(np.asarray(scores, dtype=np.float64)))


def _collect(root: Path) -> dict[str, dict[str, list[RhoticSpliceSample]]]:
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)
    grouped: dict[str, dict[str, list[RhoticSpliceSample]]] = defaultdict(lambda: defaultdict(list))

    for observation in observations:
        if observation.base_unit != "r":
            continue
        context = context_for("r", observation.final)
        if context not in {"plain", "front", "rounded"}:
            continue
        try:
            whole = consonant_segment(observation.entry, edge_trim=0.04)
            core = slice_segment(whole, 0.25, 0.62)
            late = slice_segment(whole, 0.62, 0.94)
        except (AudioReadError, ValueError):
            continue
        sample = RhoticSpliceSample(
            subbank=_subbank_name(root, observation.entry),
            context=context,
            alias=observation.entry.alias,
            final=observation.final,
            entry=observation.entry,
            core=core,
            late=late,
        )
        grouped[sample.subbank][context].append(sample)
    return grouped


def _other(samples: list[RhoticSpliceSample], target: RhoticSpliceSample) -> list[RhoticSpliceSample]:
    return [
        sample
        for sample in samples
        if sample.entry.wav_path != target.entry.wav_path or sample.alias != target.alias
    ]


def analyze_rhotic_splice(root: Path) -> tuple[list[RhoticMergeTarget], list[RhoticFrontTarget]]:
    root = root.expanduser().resolve()
    grouped = _collect(root)
    merge_targets: list[RhoticMergeTarget] = []
    front_targets: list[RhoticFrontTarget] = []

    for subbank in sorted(grouped):
        plain = grouped[subbank].get("plain", [])
        rounded = grouped[subbank].get("rounded", [])
        front = grouped[subbank].get("front", [])
        if len(plain) < 2 or len(rounded) < 2 or len(front) < 2:
            continue

        for context, own, partner in (
            ("plain", plain, rounded),
            ("rounded", rounded, plain),
        ):
            for target in own:
                same_donors = _other(own, target)
                if not same_donors:
                    continue
                try:
                    natural = rhotic_boundary_penalty(target.core, target.late)
                    same = _median_penalty(same_donors, target)
                    merge = _median_penalty(partner, target)
                    front_penalty = _median_penalty(front, target)
                except (AudioReadError, ValueError):
                    continue
                merge_targets.append(
                    RhoticMergeTarget(
                        subbank=subbank,
                        context=context,
                        alias=target.alias,
                        natural_penalty=natural,
                        same_control=same,
                        merge_substitution=merge,
                        front_substitution=front_penalty,
                        merge_delta=merge - same,
                        front_delta=front_penalty - same,
                        separation_delta=front_penalty - merge,
                    )
                )

        merged = plain + rounded
        for target in front:
            same_donors = _other(front, target)
            if not same_donors:
                continue
            try:
                natural = rhotic_boundary_penalty(target.core, target.late)
                same = _median_penalty(same_donors, target)
                substitution = _median_penalty(merged, target)
            except (AudioReadError, ValueError):
                continue
            front_targets.append(
                RhoticFrontTarget(
                    subbank=subbank,
                    alias=target.alias,
                    natural_penalty=natural,
                    same_control=same,
                    merged_substitution=substitution,
                    delta=substitution - same,
                )
            )

    return merge_targets, front_targets


def _mean(rows, name: str) -> float:
    return float(np.mean([getattr(row, name) for row in rows]))


def _print_merge_group(title: str, rows: list[RhoticMergeTarget]) -> None:
    print(title)
    print(f"  targets: {len(rows)}")
    print(f"  natural boundary: {_mean(rows, 'natural_penalty'):.4f}")
    print(f"  same-context control: {_mean(rows, 'same_control'):.4f}")
    print(f"  cross plain/rounded: {_mean(rows, 'merge_substitution'):.4f}")
    print(f"  front substitution: {_mean(rows, 'front_substitution'):.4f}")
    print(f"  same excess over natural: {_mean(rows, 'same_control') - _mean(rows, 'natural_penalty'):+.4f}")
    print(f"  cross excess over natural: {_mean(rows, 'merge_substitution') - _mean(rows, 'natural_penalty'):+.4f}")
    print(f"  merge delta vs same: {_mean(rows, 'merge_delta'):+.4f}")
    print(f"  front delta vs same: {_mean(rows, 'front_delta'):+.4f}")
    print(f"  merge-harm permutation p: {_paired_p([row.merge_delta for row in rows], 8123):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.rhotic_splice")
    parser.add_argument("voicebank", type=Path)
    args = parser.parse_args()

    merge_targets, front_targets = analyze_rhotic_splice(args.voicebank)
    print("Rhotic splice diagnostic:")
    print(f"  merge-context targets: {len(merge_targets)}")
    print(f"  front targets: {len(front_targets)}")
    print()

    for context in ("plain", "rounded"):
        rows = [row for row in merge_targets if row.context == context]
        if rows:
            _print_merge_group(f"{context.capitalize()} targets:", rows)
            print()

    if merge_targets:
        print("Combined plain/rounded:")
        print(f"  natural boundary: {_mean(merge_targets, 'natural_penalty'):.4f}")
        print(f"  same-context control: {_mean(merge_targets, 'same_control'):.4f}")
        print(f"  cross plain/rounded: {_mean(merge_targets, 'merge_substitution'):.4f}")
        print(f"  front substitution: {_mean(merge_targets, 'front_substitution'):.4f}")
        print(f"  merge delta: {_mean(merge_targets, 'merge_delta'):+.4f}")
        print(f"  front-vs-merge separation: {_mean(merge_targets, 'separation_delta'):+.4f}")
        print(f"  merge-harm permutation p: {_paired_p([row.merge_delta for row in merge_targets], 8125):.4f}")
        print(f"  front-vs-merge permutation p: {_paired_p([row.separation_delta for row in merge_targets], 8126):.4f}")
        print()

        for subbank in sorted({row.subbank for row in merge_targets}):
            rows = [row for row in merge_targets if row.subbank == subbank]
            print(
                f"  {subbank}: natural={_mean(rows, 'natural_penalty'):.4f}, "
                f"same={_mean(rows, 'same_control'):.4f}, "
                f"cross={_mean(rows, 'merge_substitution'):.4f}, "
                f"merge_delta={_mean(rows, 'merge_delta'):+.4f}"
            )
        print()

    if front_targets:
        print("Front targets:")
        print(f"  natural boundary: {_mean(front_targets, 'natural_penalty'):.4f}")
        print(f"  same-front control: {_mean(front_targets, 'same_control'):.4f}")
        print(f"  plain/rounded substitution: {_mean(front_targets, 'merged_substitution'):.4f}")
        print(f"  same excess over natural: {_mean(front_targets, 'same_control') - _mean(front_targets, 'natural_penalty'):+.4f}")
        print(f"  cross excess over natural: {_mean(front_targets, 'merged_substitution') - _mean(front_targets, 'natural_penalty'):+.4f}")
        print(f"  delta vs same: {_mean(front_targets, 'delta'):+.4f}")
        print(f"  permutation p: {_paired_p([row.delta for row in front_targets], 9107):.4f}")
        for subbank in sorted({row.subbank for row in front_targets}):
            rows = [row for row in front_targets if row.subbank == subbank]
            print(
                f"  {subbank}: natural={_mean(rows, 'natural_penalty'):.4f}, "
                f"same={_mean(rows, 'same_control'):.4f}, "
                f"cross={_mean(rows, 'merged_substitution'):.4f}, "
                f"delta={_mean(rows, 'delta'):+.4f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
