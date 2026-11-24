from __future__ import annotations

from dataclasses import replace

from .synthesis_inventory import SynthesisInventory

LEVELS = ("strict", "balanced", "loose")
MERGED = "merged_by_tolerance"
COLLAPSIBLE = frozenset({"unresolved", MERGED})


def apply_tolerance(
    inventory: SynthesisInventory,
    level: str,
) -> tuple[SynthesisInventory, tuple[str, ...]]:
    """Merge recommended splits whose evidence is weaker than the chosen level asks for.

    strict keeps every recommended split. balanced merges splits that are not
    high-confidence. loose merges every split whose acoustic evidence is not
    strong. The merge is a tolerance, not evidence that the contexts are equivalent.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown tolerance level: {level}")
    if level == "strict":
        return inventory, ()

    def weak(unit) -> bool:
        if unit.decision != "split_recommended":
            return False
        if level == "balanced":
            return unit.confidence != "high"
        return unit.acoustic_evidence != "strongly_supported"

    bases = sorted({unit.base_unit for unit in inventory.units if weak(unit)})
    units = tuple(
        replace(unit, decision=MERGED) if unit.base_unit in bases else unit
        for unit in inventory.units
    )
    return replace(inventory, units=units), tuple(bases)
