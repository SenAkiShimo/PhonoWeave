from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .stop_context import analyze_stop_context
from .stop_relevance import stop_relevance_test

if TYPE_CHECKING:
    from .inventory import InventoryDecision


_STOPS = ("b", "p", "d", "t", "g", "k")


def _pair_name(left: str, right: str) -> str:
    return f"{left}<->{right}"


def _stop_decision(root: Path, base_unit: str) -> InventoryDecision:
    from .inventory import InventoryDecision

    acoustic = analyze_stop_context(root, base_unit)
    eligible = [
        pair
        for pair in acoustic.pairwise
        if pair.stratified_permutation_p is not None
    ]
    significant = [
        pair
        for pair in eligible
        if pair.stratified_p_holm is not None
        and pair.stratified_p_holm < 0.05
    ]
    families = tuple(
        family
        for family, count in acoustic.counts.items()
        if count > 0
    )
    notes = [
        "role_scope=internal",
        f"samples={acoustic.samples}",
        f"duplicate_observations_removed={acoustic.duplicate_observations_removed}",
        f"ambiguous_segments_removed={acoustic.ambiguous_segments_removed}",
        f"ambiguous_observations_removed={acoustic.ambiguous_observations_removed}",
        f"context_families={','.join(families)}",
        f"eligible_internal_pairs={len(eligible)}",
        "acoustic_holm_pairs=" + (
            ",".join(
                f"{_pair_name(pair.left, pair.right)}:{pair.stratified_p_holm:.4f}"
                for pair in significant
            )
            if significant
            else "none"
        ),
    ]

    if not significant:
        return InventoryDecision(
            base_unit=base_unit,
            class_name="stop",
            acoustic_evidence="weak_or_inconsistent",
            synthesis_evidence="not_tested",
            decision="unresolved",
            confidence="low",
            notes=tuple(notes),
        )

    acoustic_evidence = (
        "strongly_supported"
        if len(eligible) >= 2 and len(significant) == len(eligible)
        else "supported"
    )
    relevance = stop_relevance_test(root, base_unit)
    split_pairs = [
        pair
        for pair in relevance.pairs
        if pair.both_onset_positive
        and pair.both_onset_holm_significant
        and pair.all_oto_sets_onset_positive
    ]
    notes.extend(
        [
            f"relevance_samples={relevance.samples}",
            "bidirectional_split_pairs=" + (
                ",".join(_pair_name(pair.left, pair.right) for pair in split_pairs)
                if split_pairs
                else "none"
            ),
        ]
    )

    if split_pairs and len(split_pairs) == len(significant):
        return InventoryDecision(
            base_unit=base_unit,
            class_name="stop",
            acoustic_evidence=acoustic_evidence,
            synthesis_evidence="supported_under_proxy",
            decision="split_recommended",
            confidence="moderate",
            notes=tuple(notes),
        )

    return InventoryDecision(
        base_unit=base_unit,
        class_name="stop",
        acoustic_evidence=acoustic_evidence,
        synthesis_evidence="split_not_supported_under_proxy",
        decision="unresolved",
        confidence="moderate",
        notes=tuple(notes),
    )


def stop_decisions(root: Path) -> list[InventoryDecision]:
    from .fh_fricative_inventory import fh_fricative_decisions

    decisions = [_stop_decision(root, base_unit) for base_unit in _STOPS]
    decisions.extend(fh_fricative_decisions(root))
    return decisions
