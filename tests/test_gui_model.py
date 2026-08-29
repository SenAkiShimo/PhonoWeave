from pathlib import Path

from phonoweave.gui_model import GuiRealizationGroup, _row
from phonoweave.inventory import InventoryDecision, VoicebankInventoryAnalysis
from phonoweave.profile import build_speaker_profile


def test_gui_row_uses_structured_profile_groups() -> None:
    decision = InventoryDecision(
        base_unit="h",
        class_name="fricative",
        acoustic_evidence="strongly_supported",
        synthesis_evidence="supported_under_proxy",
        decision="split_recommended",
        confidence="moderate",
        notes=(
            "role_scope=internal",
            "context_families=rounded,other",
        ),
    )
    analysis = VoicebankInventoryAnalysis(
        voicebank=Path("/tmp/voicebank"),
        decisions=[decision],
    )
    profile = build_speaker_profile(Path("/tmp/voicebank"), analysis)
    row = _row(decision, profile)

    assert row.base_unit == "h"
    assert row.groups == (
        GuiRealizationGroup("h_internal_rounded", ("internal:rounded",)),
        GuiRealizationGroup("h_internal_other", ("internal:other",)),
    )
    assert row.decision == "split_recommended"


def test_gui_group_keeps_multiple_contexts_together() -> None:
    decision = InventoryDecision(
        base_unit="s",
        class_name="fricative",
        acoustic_evidence="strongly_supported",
        synthesis_evidence="split_not_supported_under_proxy",
        decision="unresolved",
        confidence="moderate",
        notes=(),
    )
    analysis = VoicebankInventoryAnalysis(
        voicebank=Path("/tmp/voicebank"),
        decisions=[decision],
    )
    profile = build_speaker_profile(Path("/tmp/voicebank"), analysis)
    row = _row(decision, profile)

    assert all(isinstance(group, GuiRealizationGroup) for group in row.groups)
    assert all(isinstance(group.contexts, tuple) for group in row.groups)
