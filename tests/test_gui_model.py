from pathlib import Path

from phonoweave.gui_model import _row
from phonoweave.inventory import InventoryDecision
from phonoweave.profile import build_speaker_profile
from phonoweave.inventory import VoicebankInventoryAnalysis


def test_gui_row_uses_profile_groups() -> None:
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
    assert row.groups == ("h_internal_rounded", "h_internal_other")
    assert row.contexts == ("internal:rounded", "internal:other")
    assert row.decision == "split_recommended"
