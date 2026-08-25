from pathlib import Path

from phonoweave.inventory import InventoryDecision, VoicebankInventoryAnalysis
from phonoweave.profile import build_speaker_profile


def _analysis(decision: InventoryDecision) -> VoicebankInventoryAnalysis:
    return VoicebankInventoryAnalysis(
        voicebank=Path("/tmp/voicebank"),
        decisions=[decision],
    )


def test_stop_split_groups_are_internal_and_family_scoped() -> None:
    decision = InventoryDecision(
        base_unit="d",
        class_name="stop",
        acoustic_evidence="strongly_supported",
        synthesis_evidence="supported_under_proxy",
        decision="split_recommended",
        confidence="moderate",
        notes=(
            "role_scope=internal",
            "context_families=i_series,u_series,other",
        ),
    )
    profile = build_speaker_profile(Path("/tmp/voicebank"), _analysis(decision))
    groups = profile.realizations[0].groups
    assert [group.id for group in groups] == [
        "d_internal_i_series",
        "d_internal_u_series",
        "d_internal_other",
    ]
    assert [group.contexts for group in groups] == [
        ("internal:i_series",),
        ("internal:u_series",),
        ("internal:other",),
    ]


def test_unresolved_stop_keeps_candidate_families_separate() -> None:
    decision = InventoryDecision(
        base_unit="b",
        class_name="stop",
        acoustic_evidence="weak_or_inconsistent",
        synthesis_evidence="not_tested",
        decision="unresolved",
        confidence="low",
        notes=(
            "role_scope=internal",
            "context_families=i_series,u_series,other",
        ),
    )
    profile = build_speaker_profile(Path("/tmp/voicebank"), _analysis(decision))
    groups = profile.realizations[0].groups
    assert len(groups) == 3
    assert all(group.id.endswith("_unresolved") for group in groups)
    assert not any(group.id.endswith("_shared") for group in groups)
