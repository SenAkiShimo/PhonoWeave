from pathlib import Path

from phonoweave.evidence_gap import build_evidence_completion_plan, diagnose_evidence_gap
from phonoweave.inventory import InventoryDecision, VoicebankInventoryAnalysis


def _decision(
    base_unit: str,
    acoustic: str,
    synthesis: str,
    decision: str = "unresolved",
    notes: tuple[str, ...] = (),
) -> InventoryDecision:
    return InventoryDecision(
        base_unit=base_unit,
        class_name="fricative",
        acoustic_evidence=acoustic,
        synthesis_evidence=synthesis,
        decision=decision,
        confidence="moderate",
        notes=notes,
    )


def test_resolved_decision_has_no_gap() -> None:
    result = diagnose_evidence_gap(
        _decision(
            "h",
            "strongly_supported",
            "supported_under_proxy",
            decision="split_recommended",
        )
    )
    assert result is None


def test_synthesis_gap_routes_to_perceptual_validation() -> None:
    result = diagnose_evidence_gap(
        _decision(
            "s",
            "strongly_supported",
            "split_not_supported_under_proxy",
        )
    )
    assert result is not None
    assert result.gap_type == "synthesis_relevance"
    assert result.recommended_action == "perceptual_validation"
    assert result.priority == "high"


def test_coverage_gap_routes_to_supplemental_recording() -> None:
    result = diagnose_evidence_gap(
        _decision(
            "m",
            "partial_coverage_limited",
            "not_tested",
            notes=("eligible_internal_pairs=2", "samples=60"),
        )
    )
    assert result is not None
    assert result.gap_type == "coverage_limited"
    assert result.recommended_action == "supplemental_recording"
    assert "eligible_internal_pairs=2" in result.rationale


def test_weak_acoustic_evidence_is_not_merge_evidence() -> None:
    result = diagnose_evidence_gap(
        _decision(
            "f",
            "weak_or_inconsistent",
            "not_tested",
            notes=(
                "role_scope=internal",
                "context_families=rounded,other",
                "stratified_p=0.1152",
            ),
        )
    )
    assert result is not None
    assert result.gap_type == "acoustic_inconclusive"
    assert result.recommended_action == "supplemental_recording"
    assert result.context_families == ("rounded", "other")
    assert result.role_scope == "internal"


def test_completion_plan_only_contains_unresolved_decisions() -> None:
    root = Path("/tmp/voicebank")
    analysis = VoicebankInventoryAnalysis(
        voicebank=root,
        decisions=[
            _decision("f", "weak_or_inconsistent", "not_tested"),
            _decision(
                "h",
                "strongly_supported",
                "supported_under_proxy",
                decision="split_recommended",
            ),
            _decision("s", "strongly_supported", "split_not_supported_under_proxy"),
        ],
    )
    plan = build_evidence_completion_plan(analysis)
    assert [gap.base_unit for gap in plan.gaps] == ["f", "s"]
    assert [gap.base_unit for gap in plan.supplemental_recording] == ["f"]
    assert [gap.base_unit for gap in plan.perceptual_validation] == ["s"]
