from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .affricate import analyze_affricate_contrast
from .analyze import analyze_fricative_contrast
from .rhotic import analyze_rhotic_contrast
from .rhotic_relevance import rhotic_relevance_test
from .splice import splice_relevance_test


_FRICATIVES = ("sh", "s", "x")
_AFFRICATES = ("zh", "ch", "z", "c", "j", "q")


@dataclass(frozen=True)
class InventoryDecision:
    base_unit: str
    class_name: str
    acoustic_evidence: str
    synthesis_evidence: str
    decision: str
    confidence: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class VoicebankInventoryAnalysis:
    voicebank: Path
    decisions: list[InventoryDecision]


def _fricative_acoustic_level(result) -> str:
    if result.cross_core_balanced_accuracy is None or result.mean_core_distance is None:
        return "weak"

    significant = sum(item.core.permutation_p < 0.05 for item in result.subbanks)
    if (
        result.cross_core_balanced_accuracy >= 0.72
        and significant >= 2
        and result.mean_core_distance >= 0.65
    ):
        return "strong"

    if (
        result.cross_core_balanced_accuracy >= 0.68
        and result.mean_core_distance >= 0.60
    ):
        return "candidate"

    return "weak"


def _fricative_decision(root: Path, base_unit: str) -> InventoryDecision:
    result = analyze_fricative_contrast(root, base_unit)
    acoustic_level = _fricative_acoustic_level(result)
    notes = [
        f"core_cross_subbank_ba={result.cross_core_balanced_accuracy}",
        f"core_mean_distance={result.mean_core_distance}",
        f"acoustic_gate={acoustic_level}",
    ]

    if acoustic_level == "weak":
        return InventoryDecision(
            base_unit=base_unit,
            class_name="fricative",
            acoustic_evidence="weak_or_inconsistent",
            synthesis_evidence="not_tested",
            decision="unresolved",
            confidence="low",
            notes=tuple(notes),
        )

    splice = splice_relevance_test(root, base_unit)
    notes.extend(
        [
            f"splice_delta={splice.mean_delta}",
            f"splice_relative_delta={splice.mean_relative_delta}",
            f"splice_p={splice.permutation_p}",
        ]
    )

    acoustic_evidence = "strongly_supported" if acoustic_level == "strong" else "supported"

    if (
        splice.mean_delta is not None
        and splice.mean_delta > 0
        and splice.permutation_p is not None
        and splice.permutation_p < 0.05
    ):
        positive_layers = sum(item.mean_delta > 0 for item in splice.subbanks)
        significant_layers = sum(
            item.mean_delta > 0 and item.permutation_p < 0.05
            for item in splice.subbanks
        )
        notes.append(f"positive_pitch_layers={positive_layers}/{len(splice.subbanks)}")
        notes.append(f"significant_positive_pitch_layers={significant_layers}/{len(splice.subbanks)}")
        confidence = "high" if significant_layers >= 2 else "moderate"
        return InventoryDecision(
            base_unit=base_unit,
            class_name="fricative",
            acoustic_evidence=acoustic_evidence,
            synthesis_evidence="supported_under_proxy",
            decision="split_recommended",
            confidence=confidence,
            notes=tuple(notes),
        )

    return InventoryDecision(
        base_unit=base_unit,
        class_name="fricative",
        acoustic_evidence=acoustic_evidence,
        synthesis_evidence="split_not_supported_under_proxy",
        decision="unresolved",
        confidence="moderate",
        notes=tuple(notes),
    )


def _affricate_decision(root: Path, base_unit: str) -> InventoryDecision:
    result = analyze_affricate_contrast(root, base_unit)
    significant = sum(item.permutation_p < 0.05 for item in result.subbanks)
    notes = (
        f"cross_subbank_ba={result.cross_subbank_balanced_accuracy}",
        f"mean_distance={result.mean_distance}",
        f"significant_pitch_layers={significant}/{len(result.subbanks)}",
    )

    if (
        result.cross_subbank_balanced_accuracy is not None
        and result.cross_subbank_balanced_accuracy >= 0.72
        and significant >= 2
        and result.mean_distance is not None
        and result.mean_distance >= 0.65
    ):
        return InventoryDecision(
            base_unit=base_unit,
            class_name="affricate",
            acoustic_evidence="supported",
            synthesis_evidence="not_tested",
            decision="unresolved",
            confidence="moderate",
            notes=notes,
        )

    return InventoryDecision(
        base_unit=base_unit,
        class_name="affricate",
        acoustic_evidence="weak_or_inconsistent",
        synthesis_evidence="not_tested",
        decision="unresolved",
        confidence="low",
        notes=notes,
    )


def _comparison_key(target: str, substitute: str) -> tuple[str, str]:
    return target, substitute


def _relevance_supported(comparison) -> bool:
    return (
        comparison is not None
        and comparison.targets > 0
        and comparison.mean_body_spectral_delta is not None
        and comparison.mean_body_spectral_delta > 0
        and comparison.body_spectral_p is not None
        and comparison.body_spectral_p < 0.05
        and comparison.mean_boundary_delta is not None
        and comparison.mean_boundary_delta > 0
        and comparison.boundary_p is not None
        and comparison.boundary_p < 0.05
    )


def _rhotic_decision(root: Path) -> InventoryDecision:
    acoustic = analyze_rhotic_contrast(root)
    relevance = rhotic_relevance_test(root)

    pairwise = {
        frozenset((pair.left, pair.right)): pair
        for pair in acoustic.pairwise
    }
    plain_front = pairwise.get(frozenset(("plain", "front")))
    plain_rounded = pairwise.get(frozenset(("plain", "rounded")))
    front_rounded = pairwise.get(frozenset(("front", "rounded")))

    front_acoustic_supported = (
        plain_front is not None
        and front_rounded is not None
        and plain_front.cross_subbank_balanced_accuracy is not None
        and front_rounded.cross_subbank_balanced_accuracy is not None
        and plain_front.cross_subbank_balanced_accuracy >= 0.75
        and front_rounded.cross_subbank_balanced_accuracy >= 0.75
    )

    comparisons = {
        _comparison_key(item.target_context, item.substitution_context): item
        for item in relevance.comparisons
    }
    front_from_plain = comparisons.get(_comparison_key("front", "plain"))
    front_from_rounded = comparisons.get(_comparison_key("front", "rounded"))
    plain_from_rounded = comparisons.get(_comparison_key("plain", "rounded"))
    rounded_from_plain = comparisons.get(_comparison_key("rounded", "plain"))

    plain_rounded_split_supported = (
        _relevance_supported(plain_from_rounded)
        and _relevance_supported(rounded_from_plain)
    )
    front_coverage_complete = (
        front_from_plain is not None
        and front_from_plain.targets > 0
        and front_from_rounded is not None
        and front_from_rounded.targets > 0
    )
    front_synthesis_supported = (
        front_coverage_complete
        and _relevance_supported(front_from_plain)
        and _relevance_supported(front_from_rounded)
    )

    notes = [
        f"three_way_cross_subbank_ba={acoustic.cross_subbank_balanced_accuracy}",
        f"front_acoustic_supported={front_acoustic_supported}",
        f"plain_rounded_pair_ba={plain_rounded.cross_subbank_balanced_accuracy if plain_rounded else None}",
        f"plain_rounded_split_supported={plain_rounded_split_supported}",
        f"front_coverage_complete={front_coverage_complete}",
        f"front_synthesis_supported={front_synthesis_supported}",
    ]

    for name, comparison in (
        ("front_from_plain", front_from_plain),
        ("front_from_rounded", front_from_rounded),
        ("plain_from_rounded", plain_from_rounded),
        ("rounded_from_plain", rounded_from_plain),
    ):
        if comparison is None:
            continue
        notes.extend(
            [
                f"{name}_targets={comparison.targets}",
                f"{name}_body_delta={comparison.mean_body_spectral_delta}",
                f"{name}_body_p={comparison.body_spectral_p}",
                f"{name}_boundary_delta={comparison.mean_boundary_delta}",
                f"{name}_boundary_p={comparison.boundary_p}",
            ]
        )

    if plain_rounded_split_supported and front_acoustic_supported and front_synthesis_supported:
        return InventoryDecision(
            base_unit="r",
            class_name="rhotic",
            acoustic_evidence="front_distinct",
            synthesis_evidence="three_way_split_supported_under_proxy",
            decision="three_realizations_provisional",
            confidence="moderate",
            notes=tuple(notes),
        )

    if plain_rounded_split_supported:
        return InventoryDecision(
            base_unit="r",
            class_name="rhotic",
            acoustic_evidence=(
                "front_distinct_plain_rounded_mixed"
                if front_acoustic_supported
                else "mixed"
            ),
            synthesis_evidence="plain_rounded_split_supported_front_unresolved",
            decision="unresolved",
            confidence="moderate",
            notes=tuple(notes),
        )

    return InventoryDecision(
        base_unit="r",
        class_name="rhotic",
        acoustic_evidence="mixed",
        synthesis_evidence="unresolved",
        decision="unresolved",
        confidence="low",
        notes=tuple(notes),
    )


def analyze_voicebank_inventory(root: Path) -> VoicebankInventoryAnalysis:
    root = root.expanduser().resolve()
    decisions = [
        *(_fricative_decision(root, base) for base in _FRICATIVES),
        *(_affricate_decision(root, base) for base in _AFFRICATES),
        _rhotic_decision(root),
    ]
    return VoicebankInventoryAnalysis(voicebank=root, decisions=decisions)
