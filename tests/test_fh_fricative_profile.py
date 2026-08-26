from pathlib import Path

from phonoweave.inventory import InventoryDecision, VoicebankInventoryAnalysis
from phonoweave.profile import build_speaker_profile


def _profile(decision: InventoryDecision):
    analysis = VoicebankInventoryAnalysis(
        voicebank=Path("/tmp/voicebank"),
        decisions=[decision],
    )
    return build_speaker_profile(Path("/tmp/voicebank"), analysis)


def test_h_split_groups_are_internal_rounded_and_other() -> None:
    decision = InventoryDecision(
        base_unit="h",
        class_name="fricative",
        acoustic_evidence="strongly_supported",
        synthesis_evidence="supported_under_proxy",
        decision="split_recommended",
        confidence="moderate",
        notes=("role_scope=internal", "context_families=rounded,other"),
    )
    groups = _profile(decision).realizations[0].groups
    assert [group.id for group in groups] == [
        "h_internal_rounded",
        "h_internal_other",
    ]
    assert [group.contexts for group in groups] == [
        ("internal:rounded",),
        ("internal:other",),
    ]


def test_f_unresolved_keeps_internal_families_separate() -> None:
    decision = InventoryDecision(
        base_unit="f",
        class_name="fricative",
        acoustic_evidence="weak_or_inconsistent",
        synthesis_evidence="not_tested",
        decision="unresolved",
        confidence="low",
        notes=("role_scope=internal", "context_families=rounded,other"),
    )
    groups = _profile(decision).realizations[0].groups
    assert [group.id for group in groups] == [
        "f_internal_rounded_unresolved",
        "f_internal_other_unresolved",
    ]
    assert not any(group.id.endswith("_shared") for group in groups)
