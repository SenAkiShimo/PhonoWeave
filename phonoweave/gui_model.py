from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .coverage import VoicebankCoverage, analyze_coverage
from .inventory import InventoryDecision, VoicebankInventoryAnalysis, analyze_voicebank_inventory
from .profile import SpeakerProfile, build_speaker_profile, profile_yaml
from .synthesis_inventory import (
    SynthesisInventory,
    build_synthesis_inventory,
    synthesis_inventory_yaml,
)


@dataclass(frozen=True)
class GuiDecisionRow:
    base_unit: str
    class_name: str
    acoustic_evidence: str
    synthesis_evidence: str
    decision: str
    confidence: str
    groups: tuple[str, ...]
    contexts: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class GuiAnalysisSnapshot:
    root: Path
    analysis: VoicebankInventoryAnalysis
    coverage: VoicebankCoverage
    profile: SpeakerProfile
    synthesis_inventory: SynthesisInventory
    rows: tuple[GuiDecisionRow, ...]

    @property
    def analyzed_count(self) -> int:
        return sum(item.status == "analyzed" for item in self.coverage.items)

    @property
    def experimental_count(self) -> int:
        return sum(item.status == "experimental" for item in self.coverage.items)

    @property
    def unsupported_count(self) -> int:
        return sum(item.status == "unsupported" for item in self.coverage.items)


def _row(decision: InventoryDecision, profile: SpeakerProfile) -> GuiDecisionRow:
    entry = next(
        item for item in profile.realizations if item.base_unit == decision.base_unit
    )
    groups = tuple(group.id for group in entry.groups)
    contexts = tuple(
        context
        for group in entry.groups
        for context in group.contexts
    )
    return GuiDecisionRow(
        base_unit=decision.base_unit,
        class_name=decision.class_name,
        acoustic_evidence=decision.acoustic_evidence,
        synthesis_evidence=decision.synthesis_evidence,
        decision=decision.decision,
        confidence=decision.confidence,
        groups=groups,
        contexts=contexts,
        notes=decision.notes,
    )


def analyze_for_gui(root: Path) -> GuiAnalysisSnapshot:
    root = root.expanduser().resolve()
    analysis = analyze_voicebank_inventory(root)
    coverage = analyze_coverage(root)
    profile = build_speaker_profile(root, analysis)
    synthesis = build_synthesis_inventory(root, profile)
    rows = tuple(_row(decision, profile) for decision in analysis.decisions)
    return GuiAnalysisSnapshot(
        root=root,
        analysis=analysis,
        coverage=coverage,
        profile=profile,
        synthesis_inventory=synthesis,
        rows=rows,
    )


def write_cached_profile(snapshot: GuiAnalysisSnapshot, output: Path) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(profile_yaml(snapshot.profile), encoding="utf-8")
    return output


def write_cached_synthesis_inventory(
    snapshot: GuiAnalysisSnapshot,
    output: Path,
) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        synthesis_inventory_yaml(snapshot.synthesis_inventory),
        encoding="utf-8",
    )
    return output
