from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .reclist import Syllable, UnitMap
from .synthesis_inventory import SynthesisInventory, SynthesisUnit
from .transition_consistency import (
    TransitionReport,
    analyze_samples,
    collect_samples,
    supported_pairs,
)

BASES = ("m", "n", "l")
_CONTEXTS = {
    "front": ("internal:i_series", "initial:i_series"),
    "rounded": ("rounded",),
    "plain": ("plain",),
}


def collect_reports(
    root: Path,
    bases: tuple[str, ...] = BASES,
    permutations: int = 4000,
) -> dict[str, TransitionReport]:
    reports = {}
    for base in bases:
        samples, skipped = collect_samples(root, base)
        reports[base] = analyze_samples(base, samples, skipped, permutations)
    return reports


def _separated_families(report: TransitionReport) -> set[str]:
    families: set[str] = set()
    for first, second in supported_pairs(report.pairs):
        families.update((first, second))
    return families


def _already_separates(inventory: SynthesisInventory, base: str, families: set[str]) -> bool:
    units = UnitMap(inventory)
    probes = {"i_series": "i", "u_series": "u", "other": "a"}
    resolved = {
        family: units.unit(Syllable(base + final, base, final), "internal")
        for family, final in probes.items()
    }
    return all(
        resolved[first] != resolved[second]
        for first in families
        for second in families
        if first < second
    )


def _units(base: str, families: set[str]) -> list[SynthesisUnit]:
    names = []
    if "i_series" in families:
        names.append("front")
    if "u_series" in families:
        names.append("rounded")
    names.append("plain")
    contexts = dict(_CONTEXTS)
    if "u_series" not in families:
        contexts["plain"] = ("plain", "rounded")
    return [
        SynthesisUnit(
            id=f"{base}_{name}",
            base_unit=base,
            contexts=contexts[name],
            source_group=f"{base}_{name}",
            canonical_context=None,
            decision="split_recommended",
            confidence="moderate",
            acoustic_evidence="transition_supported",
            synthesis_evidence="not_tested",
        )
        for name in names
    ]


def apply_transition_evidence(
    inventory: SynthesisInventory,
    reports: dict[str, TransitionReport],
) -> tuple[SynthesisInventory, list[str]]:
    """Replace the units of m, n and l by the axes the transition analysis supports.

    Only onsets whose transition report finds a consistent difference are touched,
    and only when the current units do not already separate those families. Other
    onsets keep their decision.
    """
    notes: list[str] = []
    result = inventory
    for base, report in reports.items():
        families = _separated_families(report)
        if not families:
            notes.append(f"{base}: {report.verdict}, decision kept")
            continue
        if _already_separates(result, base, families):
            notes.append(f"{base}: transition axis already present")
            continue
        kept = tuple(unit for unit in result.units if unit.base_unit != base)
        replacement = _units(base, families)
        result = replace(result, units=kept + tuple(replacement))
        notes.append(f"{base}: split by transition evidence ({', '.join(unit.id for unit in replacement)})")
    return result, notes
