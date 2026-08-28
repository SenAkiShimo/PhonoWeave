from phonoweave.evidence_gap import EvidenceCompletionPlan, EvidenceGap
from phonoweave.supplement_plan import build_supplement_plan, request_for_gap


def _gap(
    base_unit: str,
    class_name: str,
    gap_type: str,
    action: str = "supplemental_recording",
    role_scope: str | None = None,
    families: tuple[str, ...] = (),
) -> EvidenceGap:
    return EvidenceGap(
        base_unit=base_unit,
        class_name=class_name,
        gap_type=gap_type,
        priority="high" if gap_type == "coverage_limited" else "medium",
        recommended_action=action,
        role_scope=role_scope,
        context_families=families,
        rationale=(),
    )


def test_acoustic_gap_gets_one_item_per_context() -> None:
    request = request_for_gap(
        _gap(
            "f",
            "fricative",
            "acoustic_inconclusive",
            role_scope="internal",
            families=("rounded", "other"),
        )
    )
    assert request is not None
    assert request.role_scope == "internal"
    assert [(item.context_family, item.diagnostic_items) for item in request.targets] == [
        ("rounded", 1),
        ("other", 1),
    ]
    assert request.diagnostic_items == 2
    assert request.automatic_round_limit == 1


def test_coverage_gap_gets_two_items_per_context() -> None:
    request = request_for_gap(
        _gap(
            "m",
            "nasal",
            "coverage_limited",
            role_scope="internal",
            families=("u_series", "v_series"),
        )
    )
    assert request is not None
    assert request.diagnostic_items == 4
    assert all(item.diagnostic_items == 2 for item in request.targets)


def test_rhotic_coverage_only_targets_front() -> None:
    request = request_for_gap(_gap("r", "rhotic", "coverage_limited"))
    assert request is not None
    assert [(item.context_family, item.diagnostic_items) for item in request.targets] == [
        ("front", 2)
    ]


def test_perceptual_gap_does_not_generate_recording_request() -> None:
    request = request_for_gap(
        _gap(
            "s",
            "fricative",
            "synthesis_relevance",
            action="perceptual_validation",
        )
    )
    assert request is None


def test_plan_counts_diagnostic_items() -> None:
    completion = EvidenceCompletionPlan(
        gaps=(
            _gap("f", "fricative", "acoustic_inconclusive", families=("rounded", "other")),
            _gap("r", "rhotic", "coverage_limited"),
            _gap("s", "fricative", "synthesis_relevance", action="perceptual_validation"),
        )
    )
    plan = build_supplement_plan(completion)
    assert [request.base_unit for request in plan.requests] == ["f", "r"]
    assert plan.diagnostic_items == 4
