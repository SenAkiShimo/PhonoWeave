from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .profile import SpeakerProfile, build_speaker_profile


@dataclass(frozen=True)
class SynthesisUnit:
    id: str
    base_unit: str
    contexts: tuple[str, ...]
    source_group: str
    canonical_context: str | None
    decision: str
    confidence: str
    acoustic_evidence: str
    synthesis_evidence: str


@dataclass(frozen=True)
class SynthesisInventory:
    speaker_id: str
    language: str
    source_voicebank: Path
    units: tuple[SynthesisUnit, ...]


def build_synthesis_inventory(
    root: Path,
    profile: SpeakerProfile | None = None,
) -> SynthesisInventory:
    root = root.expanduser().resolve()
    if profile is None:
        profile = build_speaker_profile(root)

    units: list[SynthesisUnit] = []
    for entry in profile.realizations:
        for group in entry.groups:
            units.append(
                SynthesisUnit(
                    id=group.id,
                    base_unit=entry.base_unit,
                    contexts=group.contexts,
                    source_group=group.id,
                    canonical_context=group.canonical_context,
                    decision=entry.decision,
                    confidence=entry.confidence,
                    acoustic_evidence=entry.acoustic_evidence,
                    synthesis_evidence=entry.synthesis_evidence,
                )
            )

    return SynthesisInventory(
        speaker_id=profile.speaker_id,
        language=profile.language,
        source_voicebank=profile.source_voicebank,
        units=tuple(units),
    )


def _scalar(value: object) -> str:
    if value is None:
        return "null"
    text = str(value)
    if not text or any(char in text for char in ":#[]{}\n\r\t") or text.strip() != text:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def synthesis_inventory_yaml(inventory: SynthesisInventory) -> str:
    lines = [
        "schema_version: 2",
        "speaker:",
        f"  id: {_scalar(inventory.speaker_id)}",
        f"  language: {_scalar(inventory.language)}",
        "source:",
        f"  voicebank: {_scalar(str(inventory.source_voicebank))}",
        "units:",
    ]

    for unit in inventory.units:
        contexts = ", ".join(_scalar(context) for context in unit.contexts)
        lines.extend(
            [
                f"  - id: {_scalar(unit.id)}",
                f"    base_unit: {_scalar(unit.base_unit)}",
                f"    contexts: [{contexts}]",
                f"    source_group: {_scalar(unit.source_group)}",
                f"    decision: {_scalar(unit.decision)}",
                f"    confidence: {_scalar(unit.confidence)}",
                "    evidence:",
                f"      acoustic: {_scalar(unit.acoustic_evidence)}",
                f"      synthesis: {_scalar(unit.synthesis_evidence)}",
            ]
        )
        if unit.canonical_context is not None:
            lines.append(
                f"    canonical_context: {_scalar(unit.canonical_context)}"
            )

    lines.extend(
        [
            "metadata:",
            "  generator: phonoweave",
            "  alias_mapping_applied: false",
        ]
    )
    return "\n".join(lines) + "\n"


def write_synthesis_inventory(root: Path, output: Path) -> SynthesisInventory:
    inventory = build_synthesis_inventory(root)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(synthesis_inventory_yaml(inventory), encoding="utf-8")
    return inventory
