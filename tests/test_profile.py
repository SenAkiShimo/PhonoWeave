from pathlib import Path

from phonoweave.inventory import InventoryDecision, VoicebankInventoryAnalysis
from phonoweave.profile import build_speaker_profile, profile_yaml


def test_profile_uses_neutral_realization_ids() -> None:
    analysis = VoicebankInventoryAnalysis(
        voicebank=Path("/tmp/voicebank"),
        decisions=[
            InventoryDecision(
                base_unit="sh",
                class_name="fricative",
                acoustic_evidence="supported",
                synthesis_evidence="supported_under_proxy",
                decision="split_recommended",
                confidence="high",
                notes=("splice_delta=0.15",),
            ),
            InventoryDecision(
                base_unit="x",
                class_name="fricative",
                acoustic_evidence="strongly_supported",
                synthesis_evidence="supported_under_proxy",
                decision="split_recommended",
                confidence="high",
                notes=("splice_delta=0.35",),
            ),
            InventoryDecision(
                base_unit="r",
                class_name="rhotic",
                acoustic_evidence="front_distinct_plain_rounded_mixed",
                synthesis_evidence="plain_rounded_split_supported_front_unresolved",
                decision="unresolved",
                confidence="moderate",
                notes=("plain_rounded_split_supported=True",),
            ),
        ],
    )

    profile = build_speaker_profile(Path("/tmp/voicebank"), analysis)
    text = profile_yaml(profile)

    assert "id: sh_plain" in text
    assert "id: sh_rounded" in text
    assert "id: x_front_unrounded" in text
    assert "contexts: [front_unrounded]" in text
    assert "id: x_rounded" in text
    assert "id: r_plain" in text
    assert "contexts: [plain]" in text
    assert "id: r_front_unresolved" in text
    assert "contexts: [front]" in text
    assert "id: r_rounded" in text
    assert "contexts: [rounded]" in text
    assert "id: r_plain_rounded" not in text
    assert "alias_mapping_applied: false" in text
    assert "id: shw" not in text
    assert "id: ry" not in text


def test_unresolved_profile_does_not_merge_contexts() -> None:
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
    text = profile_yaml(profile)

    assert "id: ch_plain_unresolved" in text
    assert "id: ch_rounded_unresolved" in text
    assert "id: ch_shared" not in text
