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


def _unquote(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        body = text[1:-1]
        return body.replace('\\"', '"').replace("\\\\", "\\")
    return text


def _split_list(text: str) -> tuple[str, ...]:
    body = text.strip()
    if not (body.startswith("[") and body.endswith("]")):
        raise ValueError(f"expected a list, got {text!r}")
    body = body[1:-1]
    items: list[str] = []
    current: list[str] = []
    quoted = False
    escaped = False
    for char in body:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\" and quoted:
            current.append(char)
            escaped = True
        elif char == '"':
            current.append(char)
            quoted = not quoted
        elif char == "," and not quoted:
            items.append("".join(current))
            current = []
        else:
            current.append(char)
    if "".join(current).strip():
        items.append("".join(current))
    return tuple(_unquote(item) for item in items)


def load_synthesis_inventory(path: Path) -> SynthesisInventory:
    speaker_id = ""
    language = ""
    voicebank = ""
    units: list[dict[str, object]] = []
    section = ""
    for raw in path.expanduser().read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if indent == 0:
            section = line.partition(":")[0]
            continue
        key, _, value = line.partition(":")
        if section == "speaker" and indent == 2:
            if key == "id":
                speaker_id = _unquote(value)
            elif key == "language":
                language = _unquote(value)
        elif section == "source" and indent == 2 and key == "voicebank":
            voicebank = _unquote(value)
        elif section == "units":
            if line.startswith("- "):
                units.append({})
                key, _, value = line[2:].partition(":")
            if not units:
                continue
            if key in {"acoustic", "synthesis"}:
                units[-1][key] = _unquote(value)
            elif key == "contexts":
                units[-1][key] = _split_list(value)
            elif key in {"id", "base_unit", "source_group", "decision", "confidence", "canonical_context"}:
                units[-1][key] = _unquote(value)

    return SynthesisInventory(
        speaker_id=speaker_id,
        language=language,
        source_voicebank=Path(voicebank),
        units=tuple(
            SynthesisUnit(
                id=str(item["id"]),
                base_unit=str(item["base_unit"]),
                contexts=tuple(item.get("contexts", ())),
                source_group=str(item.get("source_group", item["id"])),
                canonical_context=(
                    None
                    if item.get("canonical_context") in {None, "null"}
                    else str(item["canonical_context"])
                ),
                decision=str(item.get("decision", "")),
                confidence=str(item.get("confidence", "")),
                acoustic_evidence=str(item.get("acoustic", "")),
                synthesis_evidence=str(item.get("synthesis", "")),
            )
            for item in units
        ),
    )
