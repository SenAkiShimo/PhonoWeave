from pathlib import Path

from phonoweave.inventory import InventoryDecision, VoicebankInventoryAnalysis
from phonoweave.profile import build_speaker_profile
from phonoweave.synthesis_inventory import build_synthesis_inventory, synthesis_inventory_yaml


def test_synthesis_inventory_stays_alias_neutral() -> None:
    analysis = VoicebankInventoryAnalysis(
        voicebank=Path("/tmp/voicebank"),
        decisions=[
            InventoryDecision(
                base_unit="x",
                class_name="fricative",
                acoustic_evidence="strongly_supported",
                synthesis_evidence="supported_under_proxy",
                decision="split_recommended",
                confidence="high",
                notes=(),
            ),
        ],
    )
    profile = build_speaker_profile(Path("/tmp/voicebank"), analysis)
    inventory = build_synthesis_inventory(Path("/tmp/voicebank"), profile)
    text = synthesis_inventory_yaml(inventory)

    assert "id: x_front_unrounded" in text
    assert "contexts: [front_unrounded]" in text
    assert "id: x_rounded" in text
    assert "alias_mapping_applied: false" in text
    assert "xw" not in text
