from __future__ import annotations

from dataclasses import dataclass

from .evidence_gap import EvidenceCompletionPlan, EvidenceGap


@dataclass(frozen=True)
class SupplementTarget:
    context_family: str
    diagnostic_items: int


@dataclass(frozen=True)
class SupplementRequest:
    base_unit: str
    class_name: str
    gap_type: str
    priority: str
    role_scope: str | None
    targets: tuple[SupplementTarget, ...]
    pitch_policy: str
    automatic_round_limit: int
    stop_rule: str

    @property
    def diagnostic_items(self) -> int:
        return sum(target.diagnostic_items for target in self.targets)


@dataclass(frozen=True)
class SupplementPlan:
    requests: tuple[SupplementRequest, ...]

    @property
    def diagnostic_items(self) -> int:
        return sum(request.diagnostic_items for request in self.requests)


_DEFAULT_FAMILIES = {
    "affricate": ("plain", "rounded"),
    "fricative": ("plain", "rounded"),
    "nasal": ("i_series", "u_series", "v_series", "other"),
    "stop": ("i_series", "u_series", "other"),
}


def _target_families(gap: EvidenceGap) -> tuple[str, ...]:
    if gap.base_unit == "r" and gap.gap_type == "coverage_limited":
        return ("front",)
    if gap.context_families:
        return gap.context_families
    if gap.base_unit == "m":
        return ("i_series", "u_series", "other")
    return _DEFAULT_FAMILIES.get(gap.class_name, ())


def _role_scope(gap: EvidenceGap) -> str | None:
    if gap.role_scope is not None:
        return gap.role_scope
    if gap.class_name in {"nasal", "stop"}:
        return "internal"
    if gap.base_unit in {"f", "h"}:
        return "internal"
    return None


def request_for_gap(gap: EvidenceGap) -> SupplementRequest | None:
    if gap.recommended_action != "supplemental_recording":
        return None

    families = _target_families(gap)
    if not families:
        return None

    items_per_family = 2 if gap.gap_type == "coverage_limited" else 1
    targets = tuple(
        SupplementTarget(context_family=family, diagnostic_items=items_per_family)
        for family in families
    )
    return SupplementRequest(
        base_unit=gap.base_unit,
        class_name=gap.class_name,
        gap_type=gap.gap_type,
        priority=gap.priority,
        role_scope=_role_scope(gap),
        targets=targets,
        pitch_policy="repeat_each_item_across_existing_oto_sets",
        automatic_round_limit=1,
        stop_rule="reanalyze_once_then_freeze_if_still_unresolved",
    )


def build_supplement_plan(
    completion_plan: EvidenceCompletionPlan,
) -> SupplementPlan:
    requests = tuple(
        request
        for gap in completion_plan.gaps
        if (request := request_for_gap(gap)) is not None
    )
    return SupplementPlan(requests=requests)
