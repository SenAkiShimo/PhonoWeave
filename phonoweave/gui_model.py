from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .coverage import VoicebankCoverage, analyze_coverage
from .diagnostic_selector import select_diagnostic_items
from .evidence_gap import build_evidence_completion_plan
from .inventory import InventoryDecision, VoicebankInventoryAnalysis, analyze_voicebank_inventory
from .profile import SpeakerProfile, build_speaker_profile, profile_yaml
from .supplement_plan import build_supplement_plan
from .supplement_reclist import build_supplement_reclist
from .synthesis_inventory import (
    SynthesisInventory,
    build_synthesis_inventory,
    synthesis_inventory_yaml,
)


@dataclass(frozen=True)
class GuiRealizationGroup:
    id: str
    contexts: tuple[str, ...]


@dataclass(frozen=True)
class GuiDiagnosticItem:
    syllable: str
    context_family: str
    replicate: int
    existing_observations: int
    reclist_line: str


@dataclass(frozen=True)
class GuiEvidenceGap:
    gap_type: str
    priority: str
    recommended_action: str
    role_scope: str | None
    context_families: tuple[str, ...]
    rationale: tuple[str, ...]
    diagnostic_items: tuple[GuiDiagnosticItem, ...]


@dataclass(frozen=True)
class GuiDecisionRow:
    base_unit: str
    class_name: str
    acoustic_evidence: str
    synthesis_evidence: str
    decision: str
    confidence: str
    groups: tuple[GuiRealizationGroup, ...]
    notes: tuple[str, ...]
    evidence_gap: GuiEvidenceGap | None = None


@dataclass(frozen=True)
class GuiAnalysisSnapshot:
    root: Path
    analysis: VoicebankInventoryAnalysis
    coverage: VoicebankCoverage
    profile: SpeakerProfile
    synthesis_inventory: SynthesisInventory
    rows: tuple[GuiDecisionRow, ...]
    supplement_reclist: str = ""

    @property
    def analyzed_count(self) -> int:
        return sum(item.status == "analyzed" for item in self.coverage.items)

    @property
    def experimental_count(self) -> int:
        return sum(item.status == "experimental" for item in self.coverage.items)

    @property
    def unsupported_count(self) -> int:
        return sum(item.status == "unsupported" for item in self.coverage.items)


def _row(
    decision: InventoryDecision,
    profile: SpeakerProfile,
    evidence_gap: GuiEvidenceGap | None = None,
) -> GuiDecisionRow:
    entry = next(
        item for item in profile.realizations if item.base_unit == decision.base_unit
    )
    groups = tuple(
        GuiRealizationGroup(id=group.id, contexts=tuple(group.contexts))
        for group in entry.groups
    )
    return GuiDecisionRow(
        base_unit=decision.base_unit,
        class_name=decision.class_name,
        acoustic_evidence=decision.acoustic_evidence,
        synthesis_evidence=decision.synthesis_evidence,
        decision=decision.decision,
        confidence=decision.confidence,
        groups=groups,
        notes=decision.notes,
        evidence_gap=evidence_gap,
    )


def analyze_for_gui(root: Path) -> GuiAnalysisSnapshot:
    root = root.expanduser().resolve()
    analysis = analyze_voicebank_inventory(root)
    coverage = analyze_coverage(root)
    profile = build_speaker_profile(root, analysis)
    synthesis = build_synthesis_inventory(root, profile)

    completion = build_evidence_completion_plan(analysis)
    supplement = build_supplement_plan(completion)
    selection = select_diagnostic_items(root, supplement)
    reclist = build_supplement_reclist(selection)

    gap_by_base = {gap.base_unit: gap for gap in completion.gaps}
    items_by_base: dict[str, list[GuiDiagnosticItem]] = {}
    reclist_by_key = {
        (line.base_unit, line.syllable, line.replicate): line.text
        for line in reclist.lines
    }
    for item in selection.items:
        items_by_base.setdefault(item.base_unit, []).append(
            GuiDiagnosticItem(
                syllable=item.syllable,
                context_family=item.context_family,
                replicate=item.replicate,
                existing_observations=item.existing_observations,
                reclist_line=reclist_by_key[
                    (item.base_unit, item.syllable, item.replicate)
                ],
            )
        )

    gui_gaps: dict[str, GuiEvidenceGap] = {}
    for base_unit, gap in gap_by_base.items():
        gui_gaps[base_unit] = GuiEvidenceGap(
            gap_type=gap.gap_type,
            priority=gap.priority,
            recommended_action=gap.recommended_action,
            role_scope=gap.role_scope,
            context_families=gap.context_families,
            rationale=gap.rationale,
            diagnostic_items=tuple(items_by_base.get(base_unit, ())),
        )

    rows = tuple(
        _row(decision, profile, gui_gaps.get(decision.base_unit))
        for decision in analysis.decisions
    )
    return GuiAnalysisSnapshot(
        root=root,
        analysis=analysis,
        coverage=coverage,
        profile=profile,
        synthesis_inventory=synthesis,
        rows=rows,
        supplement_reclist=reclist.text(),
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
