from __future__ import annotations

from dataclasses import dataclass

from .inventory import InventoryDecision, VoicebankInventoryAnalysis


@dataclass(frozen=True)
class EvidenceGap:
    base_unit: str
    class_name: str
    gap_type: str
    priority: str
    recommended_action: str
    role_scope: str | None
    context_families: tuple[str, ...]
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceCompletionPlan:
    gaps: tuple[EvidenceGap, ...]

    @property
    def supplemental_recording(self) -> tuple[EvidenceGap, ...]:
        return tuple(gap for gap in self.gaps if gap.recommended_action == "supplemental_recording")

    @property
    def perceptual_validation(self) -> tuple[EvidenceGap, ...]:
        return tuple(gap for gap in self.gaps if gap.recommended_action == "perceptual_validation")


def _note_map(decision: InventoryDecision) -> dict[str, str]:
    values: dict[str, str] = {}
    for note in decision.notes:
        key, separator, value = note.partition("=")
        if separator:
            values[key] = value
    return values


def _families(notes: dict[str, str]) -> tuple[str, ...]:
    raw = notes.get("context_families", "")
    if not raw:
        return ()
    return tuple(item for item in raw.split(",") if item)


def _synthesis_gap(decision: InventoryDecision, notes: dict[str, str]) -> EvidenceGap:
    rationale = (
        "acoustic_contrast_supported",
        f"synthesis_evidence={decision.synthesis_evidence}",
        "additional_recording_is_not_the_primary_missing_evidence",
    )
    return EvidenceGap(
        base_unit=decision.base_unit,
        class_name=decision.class_name,
        gap_type="synthesis_relevance",
        priority="high",
        recommended_action="perceptual_validation",
        role_scope=notes.get("role_scope"),
        context_families=_families(notes),
        rationale=rationale,
    )


def _coverage_gap(decision: InventoryDecision, notes: dict[str, str]) -> EvidenceGap:
    rationale = [
        f"acoustic_evidence={decision.acoustic_evidence}",
        "coverage_is_insufficient_for_a_stable_decision",
    ]
    for key in (
        "eligible_internal_pairs",
        "significant_internal_pairs",
        "samples",
        "front_coverage_complete",
    ):
        if key in notes:
            rationale.append(f"{key}={notes[key]}")
    return EvidenceGap(
        base_unit=decision.base_unit,
        class_name=decision.class_name,
        gap_type="coverage_limited",
        priority="high",
        recommended_action="supplemental_recording",
        role_scope=notes.get("role_scope"),
        context_families=_families(notes),
        rationale=tuple(rationale),
    )


def _acoustic_gap(decision: InventoryDecision, notes: dict[str, str]) -> EvidenceGap:
    rationale = [
        f"acoustic_evidence={decision.acoustic_evidence}",
        "current_acoustic_evidence_does_not_cross_the_frozen_gate",
    ]
    for key in (
        "samples",
        "cross_oto_set_ba",
        "stratified_distance",
        "stratified_p",
        "cross_subbank_ba",
        "mean_distance",
        "core_cross_subbank_ba",
        "core_mean_distance",
    ):
        if key in notes:
            rationale.append(f"{key}={notes[key]}")
    return EvidenceGap(
        base_unit=decision.base_unit,
        class_name=decision.class_name,
        gap_type="acoustic_inconclusive",
        priority="medium",
        recommended_action="supplemental_recording",
        role_scope=notes.get("role_scope"),
        context_families=_families(notes),
        rationale=tuple(rationale),
    )


def diagnose_evidence_gap(decision: InventoryDecision) -> EvidenceGap | None:
    if decision.decision != "unresolved":
        return None

    notes = _note_map(decision)

    if decision.acoustic_evidence == "partial_coverage_limited":
        return _coverage_gap(decision, notes)

    if notes.get("front_coverage_complete") == "False":
        return _coverage_gap(decision, notes)

    if decision.synthesis_evidence in {
        "split_not_supported_under_proxy",
        "plain_rounded_split_supported_front_unresolved",
        "unresolved",
    } and decision.acoustic_evidence not in {
        "weak_or_inconsistent",
        "partial_coverage_limited",
    }:
        return _synthesis_gap(decision, notes)

    return _acoustic_gap(decision, notes)


def build_evidence_completion_plan(
    analysis: VoicebankInventoryAnalysis,
) -> EvidenceCompletionPlan:
    gaps = tuple(
        gap
        for decision in analysis.decisions
        if (gap := diagnose_evidence_gap(decision)) is not None
    )
    return EvidenceCompletionPlan(gaps=gaps)
