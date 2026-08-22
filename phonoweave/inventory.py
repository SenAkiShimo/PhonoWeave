from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .affricate import analyze_affricate_contrast
from .analyze import analyze_fricative_contrast
from .rhotic import analyze_rhotic_contrast
from .rhotic_canonical import analyze_rhotic_canonical
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


def _strong_fricative(result) -> bool:
    if result.cross_core_balanced_accuracy is None:
        return False
    significant = sum(item.core.permutation_p < 0.05 for item in result.subbanks)
    return (
        result.cross_core_balanced_accuracy >= 0.72
        and significant >= 2
        and result.mean_core_distance is not None
        and result.mean_core_distance >= 0.65
    )


def _fricative_decision(root: Path, base_unit: str) -> InventoryDecision:
    result = analyze_fricative_contrast(root, base_unit)
    notes = [
        f"core_cross_subbank_ba={result.cross_core_balanced_accuracy}",
        f"core_mean_distance={result.mean_core_distance}",
    ]

    if not _strong_fricative(result):
        return InventoryDecision(
            base_unit=base_unit,
            class_name="fricative",
            acoustic_evidence="weak_or_inconsistent",
            synthesis_evidence="not_tested",
            decision="no_split_recommended",
            confidence="moderate",
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
            acoustic_evidence="strongly_supported",
            synthesis_evidence="supported_under_proxy",
            decision="split_recommended",
            confidence=confidence,
            notes=tuple(notes),
        )

    return InventoryDecision(
        base_unit=base_unit,
        class_name="fricative",
        acoustic_evidence="strongly_supported",
        synthesis_evidence="not_supported_under_proxy",
        decision="no_split_recommended",
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
        decision="no_split_recommended",
        confidence="moderate",
        notes=notes,
    )


def _rhotic_decision(root: Path) -> InventoryDecision:
    result = analyze_rhotic_contrast(root)
    canonical = analyze_rhotic_canonical(root)

    pairwise = {
        frozenset((pair.left, pair.right)): pair
        for pair in result.pairwise
    }
    plain_front = pairwise.get(frozenset(("plain", "front")))
    plain_rounded = pairwise.get(frozenset(("plain", "rounded")))
    front_rounded = pairwise.get(frozenset(("front", "rounded")))

    front_supported = (
        plain_front is not None
        and front_rounded is not None
        and plain_front.cross_subbank_balanced_accuracy is not None
        and front_rounded.cross_subbank_balanced_accuracy is not None
        and plain_front.cross_subbank_balanced_accuracy >= 0.75
        and front_rounded.cross_subbank_balanced_accuracy >= 0.75
    )

    plain_rounded_weak = (
        plain_rounded is not None
        and plain_rounded.cross_subbank_balanced_accuracy is not None
        and plain_rounded.cross_subbank_balanced_accuracy < 0.72
    )

    plain_to_rounded = canonical.plain_to_rounded
    rounded_to_plain = canonical.rounded_to_plain
    canonical_plain_supported = (
        plain_to_rounded is not None
        and plain_to_rounded.mean_delta <= 0.03
        and (
            plain_to_rounded.permutation_p >= 0.05
            or plain_to_rounded.mean_delta <= 0
        )
    )
    reverse_harm = (
        rounded_to_plain is not None
        and rounded_to_plain.mean_delta > 0
        and rounded_to_plain.permutation_p < 0.05
    )

    notes = [
        f"three_way_cross_subbank_ba={result.cross_subbank_balanced_accuracy}",
        f"front_supported={front_supported}",
        f"plain_rounded_weak={plain_rounded_weak}",
    ]
    if plain_to_rounded is not None:
        notes.append(f"plain_to_rounded_delta={plain_to_rounded.mean_delta}")
        notes.append(f"plain_to_rounded_p={plain_to_rounded.permutation_p}")
    if rounded_to_plain is not None:
        notes.append(f"rounded_to_plain_delta={rounded_to_plain.mean_delta}")
        notes.append(f"rounded_to_plain_p={rounded_to_plain.permutation_p}")

    if front_supported and plain_rounded_weak and canonical_plain_supported:
        notes.append(f"reverse_harm={reverse_harm}")
        return InventoryDecision(
            base_unit="r",
            class_name="rhotic",
            acoustic_evidence="front_distinct_plain_rounded_weak",
            synthesis_evidence="canonical_plain_supported_under_proxy",
            decision="two_realizations_provisional",
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
