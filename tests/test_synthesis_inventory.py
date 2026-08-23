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

    assert "schema_version: 2" in text
    assert "id: x_front_unrounded" in text
    assert "contexts: [front_unrounded]" in text
    assert "id: x_rounded" in text
    assert "decision: split_recommended" in text
    assert "acoustic: strongly_supported" in text
    assert "synthesis: supported_under_proxy" in text
    assert "alias_mapping_applied: false" in text
    assert "xw" not in text


def test_synthesis_inventory_preserves_unresolved_state() -> None:
    analysis = VoicebankInventoryAnalysis(
        voicebank=Path("/tmp/voicebank"),
        decisions=[
            InventoryDecision(
                base_unit="ch",
                class_name="affricate",
                acoustic_evidence="weak_or_inconsistent",
                synthesis_evidence="not_tested",
                decision="unresolved",
                confidence="low",
                notes=(),
            ),
        ],
    )
    profile = build_speaker_profile(Path("/tmp/voicebank"), analysis)
    inventory = build_synthesis_inventory(Path("/tmp/voicebank"), profile)

    assert len(inventory.units) == 2
    assert all(unit.decision == "unresolved" for unit in inventory.units)
    assert all(unit.acoustic_evidence == "weak_or_inconsistent" for unit in inventory.units)
    assert all(unit.synthesis_evidence == "not_tested" for unit in inventory.units)

    text = synthesis_inventory_yaml(inventory)
    assert "id: ch_plain_unresolved" in text
    assert "id: ch_rounded_unresolved" in text
    assert "decision: unresolved" in text
    assert "acoustic: weak_or_inconsistent" in text
    assert "synthesis: not_tested" in text
