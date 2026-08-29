from pathlib import Path

from phonoweave.coverage import VoicebankCoverage
from phonoweave.gui import _snapshot_payload
from phonoweave.gui_model import (
    GuiAnalysisSnapshot,
    GuiDecisionRow,
    GuiDiagnosticItem,
    GuiEvidenceGap,
    GuiRealizationGroup,
)
from phonoweave.inventory import VoicebankInventoryAnalysis
from phonoweave.profile import SpeakerProfile
from phonoweave.synthesis_inventory import SynthesisInventory


def test_snapshot_payload_is_browser_ready() -> None:
    root = Path("/tmp/voicebank")
    row = GuiDecisionRow(
        base_unit="h",
        class_name="fricative",
        acoustic_evidence="strongly_supported",
        synthesis_evidence="supported_under_proxy",
        decision="split_recommended",
        confidence="moderate",
        groups=(
            GuiRealizationGroup("h_internal_rounded", ("internal:rounded",)),
            GuiRealizationGroup("h_internal_other", ("internal:other",)),
        ),
        notes=("role_scope=internal",),
    )
    snapshot = GuiAnalysisSnapshot(
        root=root,
        analysis=VoicebankInventoryAnalysis(voicebank=root, decisions=[]),
        coverage=VoicebankCoverage(
            voicebank=root,
            observations=0,
            onset_observations=0,
            zero_onset_observations=0,
            items=(),
        ),
        profile=SpeakerProfile(
            speaker_id="voicebank",
            language="mandarin",
            source_voicebank=root,
            realizations=(),
        ),
        synthesis_inventory=SynthesisInventory(
            speaker_id="voicebank",
            language="mandarin",
            source_voicebank=root,
            units=(),
        ),
        rows=(row,),
    )

    payload = _snapshot_payload(snapshot)
    assert payload["voicebank"] == str(root)
    assert payload["summary"]["onsets"] == 1
    assert payload["rows"][0]["base_unit"] == "h"
    assert payload["rows"][0]["groups"] == [
        {"id": "h_internal_rounded", "contexts": ["internal:rounded"]},
        {"id": "h_internal_other", "contexts": ["internal:other"]},
    ]


def test_snapshot_payload_includes_evidence_completion() -> None:
    root = Path("/tmp/voicebank")
    row = GuiDecisionRow(
        base_unit="f",
        class_name="fricative",
        acoustic_evidence="weak_or_inconsistent",
        synthesis_evidence="not_tested",
        decision="unresolved",
        confidence="low",
        groups=(GuiRealizationGroup("f", ("internal:rounded", "internal:other")),),
        notes=(),
        evidence_gap=GuiEvidenceGap(
            gap_type="acoustic_inconclusive",
            priority="medium",
            recommended_action="supplemental_recording",
            role_scope="internal",
            context_families=("rounded", "other"),
            rationale=("current_acoustic_evidence_does_not_cross_the_frozen_gate",),
            diagnostic_items=(
                GuiDiagnosticItem("fo", "rounded", 1, 4, "a_fo_a"),
            ),
        ),
    )
    snapshot = GuiAnalysisSnapshot(
        root=root,
        analysis=VoicebankInventoryAnalysis(voicebank=root, decisions=[]),
        coverage=VoicebankCoverage(
            voicebank=root,
            observations=0,
            onset_observations=0,
            zero_onset_observations=0,
            items=(),
        ),
        profile=SpeakerProfile(
            speaker_id="voicebank",
            language="mandarin",
            source_voicebank=root,
            realizations=(),
        ),
        synthesis_inventory=SynthesisInventory(
            speaker_id="voicebank",
            language="mandarin",
            source_voicebank=root,
            units=(),
        ),
        rows=(row,),
        supplement_reclist="a_fo_a\n",
    )

    payload = _snapshot_payload(snapshot)
    assert payload["summary"]["unresolved"] == 1
    assert payload["summary"]["supplement_items"] == 1
    assert payload["rows"][0]["evidence_gap"]["gap_type"] == "acoustic_inconclusive"
    assert payload["rows"][0]["evidence_gap"]["diagnostic_items"][0]["reclist_line"] == "a_fo_a"
