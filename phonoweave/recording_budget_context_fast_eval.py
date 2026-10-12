"""Vectorized exact-score cache for frozen held-out WAV/context distance.

A trial differs only in *which full WAV groups were acquired*. Distances
between a candidate's acoustic examples and held-out acoustic examples can
therefore be computed once per fold and reused by all seeded acquisition
orders and duration ceilings. This module does not change selection, feature
normalization, sampling, or the original evaluation definition.
"""
from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean

import numpy as np

from .recording_budget_benchmark import (
    MISSING_CELL_PENALTY, RecordingUnit,
)
from .recording_budget_duration_comparison import duration_limited_prefix


class PreparedHeldoutDistances:
    def __init__(
        self,
        candidates: dict[str, RecordingUnit],
        heldout: tuple[RecordingUnit, ...],
    ) -> None:
        groups = []
        target_by_cell = defaultdict(list)
        cursor = 0
        for unit in heldout:
            grouped = defaultdict(list)
            for e in unit.examples:
                grouped[e.cell].append(e.features)
            for cell, values in grouped.items():
                vectors = np.asarray(values, dtype=float)
                start, end = cursor, cursor + len(vectors)
                groups.append((cell, start, end))
                target_by_cell[cell].extend(range(start, end))
                cursor = end
        if not groups:
            raise ValueError("no held-out WAV/context pairs")
        test_features_by_cell = {}
        for cell, indices in target_by_cell.items():
            vectors = []
            for unit in heldout:
                for e in unit.examples:
                    if e.cell == cell:
                        vectors.append(e.features)
            test_features_by_cell[cell] = (
                np.asarray(indices, dtype=int),
                np.asarray(vectors, dtype=np.float64),
            )
        # target_by_cell indices were assigned in WAV/cell group order.
        # For the vectors, the same per-group order must be preserved.
        # Build them in a second pass directly from heldout group slices.
        assembled = defaultdict(list)
        for unit in heldout:
            groups_in_unit = defaultdict(list)
            for e in unit.examples:
                groups_in_unit[e.cell].append(e.features)
            for cell, vectors in groups_in_unit.items():
                assembled[cell].extend(vectors)
        for cell, (indices, _) in test_features_by_cell.items():
            test_features_by_cell[cell] = (
                indices, np.asarray(assembled[cell], dtype=np.float64)
            )
        self.groups = tuple(groups)
        self.unique_cells = frozenset(cell for cell, _, _ in groups)
        self.n_target_examples = cursor
        self.candidate_vectors = {}
        self.candidate_segments = {}
        for key, unit in candidates.items():
            per_cell = defaultdict(list)
            for e in unit.examples:
                per_cell[e.cell].append(e.features)
            mins = np.full(cursor, np.inf, dtype=np.float64)
            for cell, examples in per_cell.items():
                if cell not in test_features_by_cell:
                    continue
                indices, targets = test_features_by_cell[cell]
                refs = np.asarray(examples, dtype=np.float64)
                if refs.ndim != 2 or targets.ndim != 2 or refs.shape[1] != targets.shape[1]:
                    raise ValueError("incompatible acoustic feature vectors")
                differences = targets[:, None, :] - refs[None, :, :]
                mins[indices] = np.min(
                    np.sqrt(np.mean(differences * differences, axis=2)), axis=1
                )
            self.candidate_vectors[key] = mins
            self.candidate_segments[key] = len(unit.examples)

    def score_minima(self, minima: np.ndarray) -> dict:
        matched = []
        missing = 0
        covered_cells = set()
        for cell, start, end in self.groups:
            values = minima[start:end]
            if np.isinf(values).all():
                missing += 1
            else:
                if not np.isfinite(values).all():
                    raise ValueError("inconsistent finite distances in acoustic context")
                matched.append(float(np.mean(values)))
                covered_cells.add(cell)
        denominator = len(self.groups)
        within = sum(matched) / denominator
        penalty = missing * MISSING_CELL_PENALTY / denominator
        return {
            "heldout_wav_cell_pairs": denominator,
            "matched_wav_cell_pairs": len(matched),
            "missing_wav_cell_pairs": missing,
            "missing_penalty_contribution": round(penalty, 6),
            "matched_distance_contribution": round(within, 6),
            "matched_only_distance": round(float(np.mean(matched)), 5) if matched else None,
            "penalized_distance": round((sum(matched) + missing * MISSING_CELL_PENALTY) / denominator, 5),
            "full_context_coverage": len(covered_cells) == len(self.unique_cells),
            "cell_coverage": round(len(covered_cells) / len(self.unique_cells), 5),
        }

    def score_subset(self, identifiers: tuple[str, ...]) -> dict:
        best = np.full(self.n_target_examples, np.inf)
        for key in identifiers:
            best = np.minimum(best, self.candidate_vectors[key])
        return self.score_minima(best)

    def score_duration_budgets(
        self,
        sequence: tuple[str, ...],
        durations: dict[str, float],
        fractions: tuple[float, ...],
        total_available_seconds: float,
        total_available_segments: int,
    ) -> dict[float, dict]:
        if not fractions:
            raise ValueError("no budgets")
        by_fraction = {}
        for fraction in fractions:
            ceiling = total_available_seconds * fraction
            acquired, spent, blocked = duration_limited_prefix(
                sequence, durations, ceiling
            )
            by_fraction[fraction] = (acquired, spent, ceiling)
        sorted_prefixes = sorted(
            set(len(x[0]) for x in by_fraction.values())
        )
        best = np.full(self.n_target_examples, np.inf)
        count = 0
        segments = 0
        scores = {}
        for prefix in sorted_prefixes:
            while count < prefix:
                key = sequence[count]
                best = np.minimum(best, self.candidate_vectors[key])
                segments += self.candidate_segments[key]
                count += 1
            scores[prefix] = (self.score_minima(best), segments)
        result = {}
        for fraction, (acquired, spent, ceiling) in by_fraction.items():
            score, segments = scores[len(acquired)]
            result[fraction] = {
                "acquired_wavs": len(acquired),
                "acquired_eligible_segments": segments,
                "spent_fraction": round(spent / total_available_seconds, 6),
                "unspent_seconds": round(max(0.0, ceiling - spent), 6),
                **score,
            }
        return result
