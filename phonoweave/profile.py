from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .inventory import InventoryDecision, VoicebankInventoryAnalysis, analyze_voicebank_inventory


@dataclass(frozen=True)
class RealizationGroup:
    id: str
    contexts: tuple[str, ...]
    canonical_context: str | None = None


@dataclass(frozen=True)
class ProfileEntry:
    base_unit: str
    class_name: str
    decision: str
    confidence: str
    groups: tuple[RealizationGroup, ...]
    acoustic_evidence: str
    synthesis_evidence: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class SpeakerProfile:
    speaker_id: str
    language: str
    source_voicebank: Path
    realizations: tuple[ProfileEntry, ...]


def _coerce(value: str) -> Any:
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        if "/" not in value:
            return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _metrics(decision: InventoryDecision) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for note in decision.notes:
        key, separator, value = note.partition("=")
        if separator:
            output[key] = _coerce(value)
    return output


def _groups(decision: InventoryDecision) -> tuple[RealizationGroup, ...]:
    base = decision.base_unit

    if base == "r":
        if decision.decision == "three_realizations_provisional":
            return (
                RealizationGroup(id="r_plain", contexts=("plain",)),
                RealizationGroup(id="r_front", contexts=("front",)),
                RealizationGroup(id="r_rounded", contexts=("rounded",)),
            )
        if decision.decision == "unresolved":
            return (
                RealizationGroup(id="r_plain", contexts=("plain",)),
                RealizationGroup(id="r_front_unresolved", contexts=("front",)),
                RealizationGroup(id="r_rounded", contexts=("rounded",)),
            )

    if decision.decision == "split_recommended":
        if base == "x":
            return (
                RealizationGroup(
                    id="x_front_unrounded",
                    contexts=("front_unrounded",),
                ),
                RealizationGroup(id="x_rounded", contexts=("rounded",)),
            )
        return (
            RealizationGroup(id=f"{base}_plain", contexts=("plain",)),
            RealizationGroup(id=f"{base}_rounded", contexts=("rounded",)),
        )

    if decision.decision == "unresolved":
        return (
            RealizationGroup(id=f"{base}_plain_unresolved", contexts=("plain",)),
            RealizationGroup(id=f"{base}_rounded_unresolved", contexts=("rounded",)),
        )

    if decision.decision == "merge_supported":
        return (
            RealizationGroup(
                id=f"{base}_shared",
                contexts=("plain", "rounded"),
            ),
        )

    raise ValueError(
        f"unsupported inventory decision for profile generation: {decision.decision}"
    )


def build_speaker_profile(
    root: Path,
    analysis: VoicebankInventoryAnalysis | None = None,
) -> SpeakerProfile:
    root = root.expanduser().resolve()
    if analysis is None:
        analysis = analyze_voicebank_inventory(root)

    entries = tuple(
        ProfileEntry(
            base_unit=decision.base_unit,
            class_name=decision.class_name,
            decision=decision.decision,
            confidence=decision.confidence,
            groups=_groups(decision),
            acoustic_evidence=decision.acoustic_evidence,
            synthesis_evidence=decision.synthesis_evidence,
            metrics=_metrics(decision),
        )
        for decision in analysis.decisions
    )

    return SpeakerProfile(
        speaker_id=root.name,
        language="mandarin",
        source_voicebank=root,
        realizations=entries,
    )


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text or any(char in text for char in ":#[]{}\n\r\t") or text.strip() != text:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def profile_yaml(profile: SpeakerProfile) -> str:
    lines = [
        "schema_version: 2",
        "speaker:",
        f"  id: {_scalar(profile.speaker_id)}",
        f"  language: {_scalar(profile.language)}",
        "source:",
        f"  voicebank: {_scalar(str(profile.source_voicebank))}",
        "realizations:",
    ]

    for entry in profile.realizations:
        lines.extend(
            [
                f"  - base_unit: {_scalar(entry.base_unit)}",
                f"    class: {_scalar(entry.class_name)}",
                f"    decision: {_scalar(entry.decision)}",
                f"    confidence: {_scalar(entry.confidence)}",
                "    groups:",
            ]
        )
        for group in entry.groups:
            contexts = ", ".join(_scalar(context) for context in group.contexts)
            lines.append(f"      - id: {_scalar(group.id)}")
            lines.append(f"        contexts: [{contexts}]")
            if group.canonical_context is not None:
                lines.append(
                    f"        canonical_context: {_scalar(group.canonical_context)}"
                )
        lines.extend(
            [
                "    evidence:",
                f"      acoustic: {_scalar(entry.acoustic_evidence)}",
                f"      synthesis: {_scalar(entry.synthesis_evidence)}",
                "      metrics:",
            ]
        )
        if entry.metrics:
            for key, value in entry.metrics.items():
                lines.append(f"        {key}: {_scalar(value)}")
        else:
            lines.append("        {}")

    lines.extend(
        [
            "metadata:",
            "  generator: phonoweave",
            "  alias_mapping_applied: false",
            "  perceptual_validation: not_tested",
        ]
    )
    return "\n".join(lines) + "\n"


def write_speaker_profile(root: Path, output: Path) -> SpeakerProfile:
    profile = build_speaker_profile(root)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(profile_yaml(profile), encoding="utf-8")
    return profile
