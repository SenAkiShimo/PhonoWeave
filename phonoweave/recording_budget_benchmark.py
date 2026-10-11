"""Retrospective recording-budget / held-out acoustic coverage benchmark.

Supported first study: Mandarin s/sh/x fricative consonant segments with OTO
timing. Selection operates on candidate *metadata* and acquired acoustic
features only. Evaluation audio is held out by exact-WAV hash group.

The metric measures fixed-feature representativeness of omitted audio, not
perceptual quality, synthesizer performance, phoneme equivalence, or the
quality of an unrecorded candidate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import AudioReadError, consonant_segment, slice_segment
from .features import extract_fricative_features
from .mandarin import collect_observations, context_for
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps

BASES = frozenset({"s", "sh", "x"})
POLICIES = ("fixed_balanced", "stratified_random", "adaptive_observed")
BUDGETS = (0.25, 0.50, 0.75, 1.00)
SPLIT_SEEDS = (1731, 2087, 3251, 4903, 6211)
HOLDOUT_FRACTION = 0.25
MISSING_CELL_PENALTY = 2.0  # normalized acoustic distance, fixed in advance
FEATURE_NAMES = (
    "centroid", "spread", "skewness", "kurtosis", "slope",
    "high_band_ratio", "duration",
)
FEATURE_SCALE = (4000.0, 3000.0, 3.0, 8.0, 15.0, 0.4, 180.0)


@dataclass(frozen=True)
class Example:
    base: str
    context: str
    final: str
    features: tuple[float, ...]

    @property
    def cell(self) -> str:
        return f"{self.base}:{self.context}"


@dataclass(frozen=True)
class RecordingUnit:
    digest: str
    examples: tuple[Example, ...]
    source_count: int = 1

    @property
    def cells(self) -> tuple[str, ...]:
        return tuple(sorted({e.cell for e in self.examples}))

    @property
    def finals(self) -> tuple[str, ...]:
        return tuple(sorted({f"{e.cell}:{e.final}" for e in self.examples}))


@dataclass(frozen=True)
class CandidateMetadata:
    identifier: str
    cells: tuple[str, ...]
    finals: tuple[str, ...]


def normalized_features(feature) -> tuple[float, ...]:
    """Frozen, external-value-independent transforms: no held-out fit."""
    raw = np.asarray(feature.vector(), dtype=np.float64)
    if raw.shape != (len(FEATURE_SCALE),) or not np.all(np.isfinite(raw)):
        raise ValueError("invalid fricative feature vector")
    scale = np.asarray(FEATURE_SCALE, dtype=np.float64)
    return tuple(float(v) for v in np.clip(raw / scale, -3.0, 3.0))


def acoustic_distance(left: Example, right: Example) -> float:
    a = np.asarray(left.features, dtype=np.float64)
    b = np.asarray(right.features, dtype=np.float64)
    if a.shape != b.shape or len(a) != len(FEATURE_NAMES):
        raise ValueError("incompatible acoustic vectors")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_recording_units(root: Path) -> tuple[tuple[RecordingUnit, ...], dict]:
    """One selection unit per unique WAV byte sequence, not per OTO alias.

    Distinct OTO aliases on a single physical WAV are inseparable by budget.
    An exact duplicate WAV under another filename is grouped by SHA-256.
    """
    root = root.expanduser().resolve()
    entries, warnings = load_voicebank(root)
    affixes = affix_pairs(load_prefix_maps(root))
    obs = collect_observations(
        (e for e in entries if e.wav_path.is_file()), affixes
    )
    per_wav: dict[Path, list] = defaultdict(list)
    for o in obs:
        if o.base_unit in BASES and context_for(o.base_unit, o.final) in {"plain", "rounded"}:
            per_wav[o.entry.wav_path.resolve()].append(o)

    by_digest: dict[str, list[Example]] = defaultdict(list)
    unique_wavs = Counter()
    skipped = 0
    ambiguous = 0
    for wav_path, items in sorted(per_wav.items()):
        digest = sha256(wav_path.read_bytes()).hexdigest()
        unique_wavs[digest] += 1
        # Repeated aliases of the same OTO segment count once. Ambiguous
        # incompatible labels for one physical segment are discarded.
        segments: dict[tuple[float, float], list] = defaultdict(list)
        for o in items:
            entry = o.entry
            start = round(entry.offset, 3)
            end = round(entry.offset + entry.preutterance, 3)
            segments[(start, end)].append(o)
        for group in segments.values():
            labels = {
                (o.base_unit, context_for(o.base_unit, o.final), o.final)
                for o in group
            }
            if len(labels) != 1:
                ambiguous += 1
                continue
            base, context, final = next(iter(labels))
            try:
                segment = consonant_segment(group[0].entry)
                core = slice_segment(segment, 0.20, 0.60)
                vector = normalized_features(extract_fricative_features(core))
            except (AudioReadError, ValueError, OSError):
                skipped += 1
                continue
            by_digest[digest].append(Example(base, context, final, vector))

    result = []
    for digest in sorted(by_digest):
        examples = by_digest[digest]
        if not examples:
            continue
        # Preserve distinct segment observations, but deduplicate identical
        # features from WAV aliases or byte-identical source copies.
        seen = set()
        unique = []
        for example in examples:
            signature = (example.cell, example.final, example.features)
            if signature not in seen:
                seen.add(signature)
                unique.append(example)
        result.append(RecordingUnit(digest, tuple(unique), unique_wavs[digest]))
    return tuple(result), {
        "oto_entries": len(entries),
        "oto_warnings": len(warnings),
        "eligible_wav_paths": len(per_wav),
        "unique_hashes_with_features": len(result),
        "exact_duplicate_paths": sum(v - 1 for v in unique_wavs.values()),
        "skipped_segments": skipped,
        "ambiguous_segments": ambiguous,
    }


def split_groups(
    units: tuple[RecordingUnit, ...],
    seed: int,
    fraction: float = HOLDOUT_FRACTION,
) -> tuple[tuple[RecordingUnit, ...], tuple[RecordingUnit, ...]]:
    """Stratified-by-metadata, WAV-grouped holdout without feature inspection."""
    if not 0 < fraction < 0.5 or len(units) < 4:
        raise ValueError("need at least four WAV groups and holdout fraction < 0.5")
    identifiers = [u.digest for u in units]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate physical recording identifiers")
    rng = np.random.default_rng(seed)
    # Stratification uses only phonetic-class/context metadata.
    all_counts = Counter(cell for u in units for cell in u.cells)
    train_counts = all_counts.copy()
    holdout_counts: Counter[str] = Counter()
    shuffled = list(rng.permutation(len(units)))
    target = max(1, min(len(units) - 2, round(len(units) * fraction)))
    selected = set()
    while len(selected) < target:
        feasible = [
            idx for idx in shuffled if idx not in selected
            and all(train_counts[cell] >= 2 for cell in units[idx].cells)
        ]
        if not feasible:
            raise ValueError(
                "cannot form metadata-stratified holdout without removing "
                "the only training recording for a context"
            )
        # Prefer filling currently absent holdout strata, with seeded
        # order determining ties; never inspect WAV feature vectors.
        best = max(feasible, key=lambda i: sum(
            1.0 / (1 + holdout_counts[cell]) for cell in units[i].cells
        ))
        selected.add(best)
        train_counts.subtract(units[best].cells)
        holdout_counts.update(units[best].cells)
    train = tuple(units[i] for i in range(len(units)) if i not in selected)
    held = tuple(units[i] for i in range(len(units)) if i in selected)
    assert {u.digest for u in train}.isdisjoint(u.digest for u in held)
    return train, held


def metadata(units: tuple[RecordingUnit, ...]) -> tuple[CandidateMetadata, ...]:
    return tuple(CandidateMetadata(u.digest, u.cells, u.finals) for u in units)


def select_sequence(
    candidates: tuple[CandidateMetadata, ...],
    acquire: Callable[[str], tuple[Example, ...]],
    *,
    policy: str, seed: int,
) -> tuple[str, ...]:
    """Sequential selection; acquire is called ONLY after a candidate is chosen.

    Unselected WAV features are not passed to the decision rule. All policies
    may use the same acquisition-pool phonetic metadata.
    """
    if policy not in POLICIES:
        raise ValueError("unknown selection policy")
    if len({c.identifier for c in candidates}) != len(candidates):
        raise ValueError("duplicate candidates")
    rng = np.random.default_rng(seed)
    remaining = {c.identifier: c for c in candidates}
    acquired: list[Example] = []
    order = []
    cell_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    while remaining:
        dispersion = {}
        if policy == "adaptive_observed":
            cell_features: dict[str, list[Example]] = defaultdict(list)
            for item in acquired:
                cell_features[item.cell].append(item)
            for cell, samples in cell_features.items():
                if len(samples) >= 2:
                    distances = [
                        acoustic_distance(samples[i], samples[j])
                        for i in range(len(samples)) for j in range(i + 1, len(samples))
                    ]
                    dispersion[cell] = float(np.median(distances))
        def priority(candidate: CandidateMetadata) -> float:
            scores = []
            for cell in candidate.cells:
                score = 1.0 / (1.0 + cell_counts[cell])
                if policy == "adaptive_observed":
                    d = dispersion.get(cell, 0.0)
                    score += 0.4 * d / (0.3 + d) if d > 0 else 0.0
                    score += 0.08 * sum(
                        final_counts[f] == 0 for f in candidate.finals
                        if f.startswith(cell + ":")
                    )
                scores.append(score)
            return max(scores, default=0.0)
        if policy == "stratified_random":
            ordered = sorted(remaining)
            weights = np.array([priority(remaining[k]) for k in ordered])
            if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0:
                raise ValueError("nonpositive selection weights")
            chosen = ordered[int(rng.choice(len(ordered), p=weights / weights.sum()))]
        else:
            chosen = min(remaining, key=lambda k: (-priority(remaining[k]), k))
        candidate = remaining.pop(chosen)
        order.append(chosen)
        # This is the only point at which the selector receives the
        # acoustic values of the chosen candidate.
        newly_acquired = acquire(chosen)
        acquired.extend(newly_acquired)
        cell_counts.update(set(candidate.cells))
        final_counts.update(set(candidate.finals))
    return tuple(order)


def evaluate_heldout(
    selected: tuple[RecordingUnit, ...], heldout: tuple[RecordingUnit, ...]
) -> dict:
    """Separate cell coverage from matched-only and penalized distances."""
    pool = defaultdict(list)
    for unit in selected:
        for e in unit.examples:
            pool[e.cell].append(e)
    unique_heldout_cells = sorted({
        e.cell for unit in heldout for e in unit.examples
    })
    distances = []
    matched = []
    for unit in heldout:
        # Aggregate per test WAV/cell, preventing multiple OTO aliases
        # in one WAV from being treated as independent held-out recordings.
        grouped = defaultdict(list)
        for e in unit.examples:
            grouped[e.cell].append(e)
        for cell, samples in grouped.items():
            if cell not in pool:
                distances.append(MISSING_CELL_PENALTY)
                continue
            d = float(np.mean([
                min(acoustic_distance(e, model) for model in pool[cell])
                for e in samples
            ]))
            distances.append(d)
            matched.append(d)
    return {
        "heldout_cells": len(unique_heldout_cells),
        "covered_cells": sum(cell in pool for cell in unique_heldout_cells),
        "cell_coverage": round(
            sum(cell in pool for cell in unique_heldout_cells)
            / len(unique_heldout_cells), 5
        ) if unique_heldout_cells else None,
        "heldout_wav_cell_pairs": len(distances),
        "matched_wav_cell_pairs": len(matched),
        "matched_only_acoustic_distance": (
            round(float(np.mean(matched)), 5) if matched else None
        ),
        "penalized_acoustic_distance": (
            round(float(np.mean(distances)), 5) if distances else None
        ),
    }


def run_budget_benchmark(
    units: tuple[RecordingUnit, ...],
    *,
    split_seeds: tuple[int, ...] = SPLIT_SEEDS,
    budgets: tuple[float, ...] = BUDGETS,
) -> dict:
    if not units or not split_seeds or any(not 0 < b <= 1 for b in budgets):
        raise ValueError("invalid benchmark input")
    rows = []
    split_diagnostics = []
    for fold, seed in enumerate(split_seeds):
        train, held = split_groups(units, seed)
        train_map = {u.digest: u for u in train}
        candidates = metadata(train)
        split_diagnostics.append({
            "fold": fold, "seed": seed,
            "train_unique_wavs": len(train), "heldout_unique_wavs": len(held),
            "train_cells": sorted({e.cell for u in train for e in u.examples}),
            "heldout_cells": sorted({e.cell for u in held for e in u.examples}),
            "training_sha256": sorted(u.digest for u in train),
            "heldout_sha256": sorted(u.digest for u in held),
        })
        for policy in POLICIES:
            # Caller hides unselected acoustic data behind acquire().
            sequence = select_sequence(
                candidates, lambda digest: train_map[digest].examples,
                policy=policy, seed=seed + 8101,
            )
            for fraction in budgets:
                count = max(1, min(len(train), math.ceil(len(train) * fraction)))
                chosen = tuple(train_map[i] for i in sequence[:count])
                metric = evaluate_heldout(chosen, held)
                rows.append({
                    "fold": fold, "seed": seed, "policy": policy,
                    "budget_fraction": fraction,
                    "budget_wav": count,
                    "train_available_wav": len(train),
                    **metric,
                })
    # Aggregate across held-out folds; folds share recordings and must
    # not be interpreted as independent speakers or sampling CIs.
    aggregated = []
    for policy in POLICIES:
        for fraction in budgets:
            subset = [
                row for row in rows
                if row["policy"] == policy and row["budget_fraction"] == fraction
            ]
            aggregated.append({
                "policy": policy,
                "budget_fraction": fraction,
                "folds": len(subset),
                "mean_cell_coverage": round(float(np.mean([
                    row["cell_coverage"] for row in subset
                    if row["cell_coverage"] is not None
                ])), 5),
                "mean_penalized_acoustic_distance": round(float(np.mean([
                    row["penalized_acoustic_distance"] for row in subset
                    if row["penalized_acoustic_distance"] is not None
                ])), 5),
                "per_fold": [
                    {"seed": row["seed"],
                     "coverage": row["cell_coverage"],
                     "distance": row["penalized_acoustic_distance"]}
                    for row in subset
                ],
            })
    return {
        "experiment": "recording_budget_vs_heldout_acoustic_coverage",
        "version": "0.1-retrospective",
        "grouping": "unique_wav_sha256",
        "scope": "Mandarin s/sh/x fricative OTO consonant core",
        "policies": list(POLICIES), "split_seeds": list(split_seeds),
        "budgets": list(budgets),
        "holdout_fraction": HOLDOUT_FRACTION,
        "distance_missing_cell_penalty": MISSING_CELL_PENALTY,
        "fixed_feature_scale": list(FEATURE_SCALE),
        "splits": split_diagnostics,
        "rows": rows, "aggregate": aggregated,
        "limitations": [
            "Retrospective selection from existing WAV candidates; candidate phonetic metadata is preknown.",
            "Only acquired acoustic vectors influence adaptive selection; held-out WAVs are never candidates.",
            "Acoustic nearest-neighbor coverage is not synthesis quality or perceptual interchangeability.",
            "OTO segmentation, pitch, recording session and alias recognition can bias this exploratory metric.",
            "Group-level SHA-256 avoids exact-WAV leakage but not re-encoded copies or near-duplicate takes.",
            "Repeated folds from a single speaker's voicebank are not independent speaker replicates.",
            "A policy advantage does not establish generalization to unseen singers or languages.",
        ],
    }


def run_voicebank_benchmark(root: Path, output: Path) -> dict:
    units, extraction = load_recording_units(root)
    if len(units) < 4:
        raise ValueError(f"insufficient usable unique WAV groups: {len(units)}")
    report = run_budget_benchmark(units)
    report["extraction"] = extraction
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
