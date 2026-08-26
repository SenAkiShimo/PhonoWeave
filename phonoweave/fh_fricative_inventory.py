from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .fh_fricative import analyze_fh_fricative
from .h_fricative_relevance import h_fricative_relevance_test

if TYPE_CHECKING:
    from .inventory import InventoryDecision


def _decision(root: Path, base_unit: str) -> "InventoryDecision":
    from .inventory import InventoryDecision

    acoustic = analyze_fh_fricative(root, base_unit)
    p_value = acoustic.stratified_permutation_p
    cross = acoustic.cross_oto_set_balanced_accuracy
    notes = [
        "role_scope=internal",
        "context_families=rounded,other",
        f"samples={acoustic.samples}",
        f"duplicate_observations_removed={acoustic.duplicate_observations_removed}",
        f"ambiguous_segments_removed={acoustic.ambiguous_segments_removed}",
        f"cross_oto_set_ba={cross}",
        f"stratified_distance={acoustic.stratified_distance}",
        f"stratified_p={p_value}",
    ]

    acoustic_supported = p_value is not None and p_value < 0.05
    if not acoustic_supported:
        notes.append("acoustic_gate=False")
        return InventoryDecision(
            base_unit=base_unit,
            class_name="fricative",
            acoustic_evidence="weak_or_inconsistent",
            synthesis_evidence="not_tested",
            decision="unresolved",
            confidence="low",
            notes=tuple(notes),
        )

    strong = (
        p_value is not None
        and p_value < 0.01
        and cross is not None
        and cross >= 0.75
    )
    notes.append("acoustic_gate=True")

    if base_unit == "f":
        return InventoryDecision(
            base_unit="f",
            class_name="fricative",
            acoustic_evidence=("strongly_supported" if strong else "supported"),
            synthesis_evidence="not_tested",
            decision="unresolved",
            confidence="moderate",
            notes=tuple(notes),
        )

    relevance = h_fricative_relevance_test(root)
    pair = relevance.pair
    notes.extend(
        [
            f"relevance_samples={relevance.samples}",
            f"both_boundary_positive={pair.both_boundary_positive if pair else False}",
            f"both_boundary_holm_significant={pair.both_boundary_holm_significant if pair else False}",
            f"all_oto_sets_boundary_positive={pair.all_oto_sets_boundary_positive if pair else False}",
            f"split_supported_under_proxy={pair.split_supported_under_proxy if pair else False}",
        ]
    )

    if pair is not None and pair.split_supported_under_proxy:
        return InventoryDecision(
            base_unit="h",
            class_name="fricative",
            acoustic_evidence=("strongly_supported" if strong else "supported"),
            synthesis_evidence="supported_under_proxy",
            decision="split_recommended",
            confidence="moderate",
            notes=tuple(notes),
        )

    return InventoryDecision(
        base_unit="h",
        class_name="fricative",
        acoustic_evidence=("strongly_supported" if strong else "supported"),
        synthesis_evidence="split_not_supported_under_proxy",
        decision="unresolved",
        confidence="moderate",
        notes=tuple(notes),
    )


def fh_fricative_decisions(root: Path) -> list["InventoryDecision"]:
    root = root.expanduser().resolve()
    return [_decision(root, "f"), _decision(root, "h")]
