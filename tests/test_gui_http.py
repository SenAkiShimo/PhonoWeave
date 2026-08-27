from pathlib import Path

from phonoweave.coverage import VoicebankCoverage
from phonoweave.gui import _snapshot_payload
from phonoweave.gui_model import GuiAnalysisSnapshot, GuiDecisionRow
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
        groups=("h_internal_rounded", "h_internal_other"),
        contexts=("internal:rounded", "internal:other"),
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
        "h_internal_rounded",
        "h_internal_other",
    ]
