"""Within-context sham acoustic feedback and exact incremental adaptive selector.

Shuffling is applied to complete seven-dimensional acoustic vectors within
each (consonant, context) cell of the TRAINING fold only. Labels, finals,
recording grouping and segment multiplicity are retained.

The sham feedback provider returns values only after its WAV is selected.
The accelerated selector caches pairwise distances of already acquired
vectors; it implements the same priority rule and lexical tie break as
recording_budget_benchmark.select_sequence(adaptive_observed).
"""
from __future__ import annotations

from bisect import insort
from collections import Counter, defaultdict
from dataclasses import replace
from statistics import median

import numpy as np

from .recording_budget_benchmark import CandidateMetadata, Example, RecordingUnit


def permuted_training_feedback(
    training: tuple[RecordingUnit, ...], *, seed: int,
) -> tuple[dict[str, tuple[Example, ...]], dict]:
    """Randomize acoustic vector-to-segment identities within metadata cells.

    Offline permutations use the *training* acoustic vectors to construct a
    retrospective null. Neither the ranking function nor the evaluation
    accesses the held-out recordings or unacquired true WAV features.
    """
    if len({u.digest for u in training}) != len(training):
        raise ValueError("duplicate WAV identifiers")
    slots = defaultdict(list)
    lookup = {u.digest: u for u in training}
    for unit in sorted(training, key=lambda u: u.digest):
        for index, ex in enumerate(unit.examples):
            if not np.all(np.isfinite(np.asarray(ex.features, dtype=float))):
                raise ValueError("non-finite acoustic vector")
            slots[ex.cell].append((unit.digest, index, ex))
    rng = np.random.default_rng(seed)
    assigned = {u.digest: list(u.examples) for u in training}
    changes = 0
    eligible_slots = 0
    permutable_cells = 0
    for cell in sorted(slots):
        values = slots[cell]
        count = len(values)
        eligible_slots += count
        if count > 1:
            permutable_cells += 1
        permutation = rng.permutation(count)
        vectors = [e.features for _, _, e in values]
        for slot_index, (digest, index, original) in enumerate(values):
            vector = vectors[int(permutation[slot_index])]
            if vector != original.features:
                changes += 1
            assigned[digest][index] = replace(original, features=vector)
    return (
        {key: tuple(items) for key, items in assigned.items()},
        {
            "shuffled_training_examples": eligible_slots,
            "changed_feature_assignments": changes,
            "permutable_cells": permutable_cells,
        },
    )


def incremental_adaptive_sequence(
    candidates: tuple[CandidateMetadata, ...],
    acquire,
) -> tuple[str, ...]:
    """Frozen adaptive priority with acquired-only cached pair distances."""
    if len({c.identifier for c in candidates}) != len(candidates):
        raise ValueError("duplicate candidate identifiers")
    remaining = {c.identifier: c for c in candidates}
    cell_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    acquired_vectors = defaultdict(list)
    sorted_pair_distances = defaultdict(list)
    selected = []
    while remaining:
        dispersion = {}
        for cell, d in sorted_pair_distances.items():
            if d:
                length = len(d)
                mid = length // 2
                dispersion[cell] = (
                    float(d[mid]) if length % 2
                    else float((d[mid - 1] + d[mid]) / 2)
                )
        def priority(candidate):
            scores = []
            for cell in candidate.cells:
                score = 1.0 / (1.0 + cell_counts[cell])
                value = dispersion.get(cell, 0.0)
                score += 0.4 * value / (0.3 + value) if value > 0 else 0.0
                score += 0.08 * sum(
                    final_counts[f] == 0 for f in candidate.finals
                    if f.startswith(cell + ":")
                )
                scores.append(score)
            return max(scores, default=0.0)
        chosen = min(
            remaining, key=lambda identifier: (
                -priority(remaining[identifier]), identifier
            )
        )
        meta = remaining.pop(chosen)
        selected.append(chosen)
        # Feedback is requested only for a selected WAV.
        for example in acquire(chosen):
            vector = np.asarray(example.features, dtype=float)
            if vector.ndim != 1 or not np.all(np.isfinite(vector)):
                raise ValueError("invalid acquired acoustic vector")
            old = acquired_vectors[example.cell]
            if old:
                previous = np.asarray(old, dtype=float)
                if previous.shape[1] != vector.shape[0]:
                    raise ValueError("mismatched acoustic vector dimensions")
                distances = np.sqrt(np.mean(
                    (previous - vector[None, :]) ** 2, axis=1
                ))
                for distance in distances:
                    insort(sorted_pair_distances[example.cell], float(distance))
            old.append(vector)
        cell_counts.update(set(meta.cells))
        final_counts.update(set(meta.finals))
    return tuple(selected)
