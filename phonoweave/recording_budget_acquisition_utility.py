"""Exploratory, acquired-evidence marginal utility for recording selection.

Candidate information is restricted to planned OTO consonant/context/final
metadata. A WAV contributes at most *one independent centroid per context* to
the evidence state, irrespective of its number of OTO fragments. The next
WAV's spectral features and realized duration are never inspected for ranking.

This is a deterministic proxy for expected reduction in *uncertainty*, not a
calibrated prediction of an unobserved waveform's acoustic quality.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from statistics import median
from typing import Callable

import numpy as np

from .recording_budget_benchmark import CandidateMetadata, Example

PROTOCOL_VERSION = "1.0-acquisition-utility-prototype"
# Theory-driven, frozen exploratory constants; not fitted to a singer.
NOVELTY_REFERENCE = 0.15
NOVELTY_STRENGTH = 0.5
FINAL_BONUS = 0.02
VECTOR_DIMENSION = 7


def _vector_centroids(examples: tuple[Example, ...]) -> dict[str, tuple[float, ...]]:
    by_cell = defaultdict(list)
    for example in examples:
        values = tuple(float(value) for value in example.features)
        if len(values) != VECTOR_DIMENSION or not all(math.isfinite(v) for v in values):
            raise ValueError("expected finite seven-component fricative vectors")
        by_cell[example.cell].append(values)
    return {
        cell: tuple(float(v) for v in np.mean(vectors, axis=0))
        for cell, vectors in by_cell.items()
    }


class AcousticEvidence:
    """Counts independent acquired WAV/context units, not OTO alias counts."""

    def __init__(self) -> None:
        self.centroids: dict[str, list[tuple[float, ...]]] = defaultdict(list)
        self.nearest: dict[str, list[float]] = defaultdict(list)

    def observe(self, examples: tuple[Example, ...]) -> None:
        for cell, centroid in _vector_centroids(examples).items():
            previous = self.centroids[cell]
            closest = self.nearest[cell]
            if previous:
                distances = [
                    math.sqrt(sum((a - b) ** 2 for a, b in zip(centroid, old)) / VECTOR_DIMENSION)
                    for old in previous
                ]
                for i, d in enumerate(distances):
                    closest[i] = min(closest[i], d)
                closest.append(min(distances))
            else:
                closest.append(math.inf)
            previous.append(centroid)

    def observations(self, cell: str) -> int:
        return len(self.centroids[cell])

    def marginal_gain(self, cell: str) -> float:
        n = self.observations(cell)
        # Reduction in n-dependent uncertainty from one additional WAV.
        reduction = (1.0 / math.sqrt(1.0 + n)
                     - 1.0 / math.sqrt(2.0 + n))
        if n < 2:
            return reduction
        finite = [value for value in self.nearest[cell] if math.isfinite(value)]
        residual = max(0.0, float(median(finite))) if finite else 0.0
        normalized_novelty = residual / (NOVELTY_REFERENCE + residual)
        return reduction * (1.0 + NOVELTY_STRENGTH * normalized_novelty)


def select_utility_sequence(
    candidates: tuple[CandidateMetadata, ...],
    acquire: Callable[[str], tuple[Example, ...]] | None = None,
    *,
    use_acoustic_feedback: bool = True,
) -> tuple[str, ...]:
    """Metadata-first coverage, then empirically weighted diminishing returns.

    The acquire callback is called exactly once, only *after* each selection.
    A metadata-only ablation never calls acquire. An unrecorded candidate is
    described exclusively by its declared cell and final labels.
    """
    if use_acoustic_feedback and acquire is None:
        raise ValueError("acoustic-feedback policy requires an acquire callback")
    if len({c.identifier for c in candidates}) != len(candidates):
        raise ValueError("duplicate candidate identifiers")
    if any(not c.cells or not set(c.finals) for c in candidates):
        raise ValueError("missing planned phonetic metadata")
    for c in candidates:
        if any(not any(f.startswith(cell + ":") for f in c.finals) for cell in c.cells):
            raise ValueError("candidate final metadata incompatible with context cells")

    remaining = {c.identifier: c for c in candidates}
    uncovered = set(cell for c in candidates for cell in c.cells)
    selected: list[str] = []
    counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    evidence = AcousticEvidence()

    while remaining:
        def rank(key: str) -> tuple[float, float]:
            meta = remaining[key]
            cells = set(meta.cells)
            gain = len(cells & uncovered)
            benefits = []
            for cell in sorted(cells):
                n = counts[cell]
                reduction = (
                    evidence.marginal_gain(cell)
                    if use_acoustic_feedback else
                    1.0 / math.sqrt(1.0 + n) - 1.0 / math.sqrt(2.0 + n)
                )
                relevant = [f for f in set(meta.finals) if f.startswith(cell + ":")]
                novelty = FINAL_BONUS * sum(1.0 / (1.0 + final_counts[f]) for f in relevant)
                benefits.append(reduction + novelty)
            # Sum information gains across offered contexts; avoid treating
            # repeated OTO aliases as independent candidate opportunities.
            return float(gain), float(sum(benefits))

        chosen = min(remaining, key=lambda key: (
            -rank(key)[0], -rank(key)[1], key
        ))
        candidate = remaining.pop(chosen)
        selected.append(chosen)
        if use_acoustic_feedback:
            # Unacquired WAV vectors are not requested before this point.
            assert acquire is not None
            acquired = tuple(acquire(chosen))
            expected = set(candidate.cells)
            observed = {example.cell for example in acquired}
            if not observed.issubset(expected):
                raise ValueError("acquired acoustic cells disagree with declared metadata")
            evidence.observe(acquired)
        counts.update(set(candidate.cells))
        final_counts.update(set(candidate.finals))
        uncovered.difference_update(candidate.cells)
    return tuple(selected)
