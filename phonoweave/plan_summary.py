from __future__ import annotations

from .bank_concordance import ConcordanceRow
from .bank_supplement import SPLIT_STATUS, BankSupplement
from .synthesis_inventory import SynthesisInventory

_TIERS = (
    ("high", "strong (several layers agree)"),
    ("moderate", "moderate"),
    ("low", "weak (single layer)"),
)
_BASIS = {
    "transition_supported": "transition analysis",
    "single_layer_supported": "within-layer test",
}


def _evidence(inventory: SynthesisInventory, base: str) -> tuple[str, str]:
    for unit in inventory.units:
        if unit.base_unit == base and unit.decision != "unresolved":
            return unit.confidence, unit.acoustic_evidence
    return "", ""


def build_summary(
    rows: list[ConcordanceRow],
    inventory: SynthesisInventory,
    supplement: BankSupplement,
    settings: dict[str, str],
    usable_layers: int | None = None,
) -> str:
    recorded = {label for label in supplement.labels}
    by_tier: dict[str, list[str]] = {tier: [] for tier, _ in _TIERS}
    for row in rows:
        labels = [label for label in row.suggested if label in recorded]
        if not labels or row.status != SPLIT_STATUS:
            continue
        confidence, evidence = _evidence(inventory, row.base)
        basis = _BASIS.get(evidence)
        entry = f"{row.base} ({', '.join(labels)})" + (f", {basis}" if basis else "")
        by_tier.get(confidence, by_tier["moderate"]).append(entry)

    lines = ["PhonoWeave plan summary", ""]
    if supplement.lines:
        lines.append(
            f"Record: {len(supplement.lines)} lines in bank_supplement.txt, once per pitch layer."
        )
        for tier, label in _TIERS:
            lines.append(f"  {label}: {'; '.join(by_tier[tier]) or 'none'}")
    else:
        lines.append("Record: nothing to add for this voicebank at these settings.")

    held_back = [
        row for row in rows
        if row.suggested and not any(label in recorded for label in row.suggested)
    ]
    if held_back:
        lines.append("")
        lines.append("Recommended here but not in the supplement:")
        for row in held_back:
            why = row.status if row.status != SPLIT_STATUS else "left out by --supplement-onsets"
            lines.append(f"  {row.base} ({', '.join(row.suggested)}): {why}")

    no_evidence = [row.base for row in rows if row.status == "voicebank splits, we have no evidence"]
    if no_evidence:
        lines.append("")
        lines.append(
            "The voicebank splits these but there is no evidence either way here: "
            + " ".join(no_evidence)
        )

    unresolved = sorted(
        {unit.base_unit for unit in inventory.units if unit.decision in {"unresolved", "merged_by_tolerance"}}
    )
    if unresolved:
        lines.append("")
        lines.append(
            "Unresolved, treated as one unit per onset (no recording needed): "
            + " ".join(unresolved)
        )

    if usable_layers is not None and usable_layers <= 1:
        covered = settings.get("single-layer evidence") == "on"
        lines.append("")
        lines.append(
            "Only one layer holds usable CV entries. The checks that compare layers "
            + ("(r) " if covered else "(s, sh, x, the affricates and r) ")
            + "cannot be passed, so those onsets stay unresolved for lack of data."
        )

    lines.append("")
    lines.append("Settings: " + ", ".join(f"{key} {value}" for key, value in settings.items()))
    lines.append(
        "Unresolved means the evidence was not enough to call a split, not that the "
        "contexts are equivalent. Thresholds are provisional."
    )
    return "\n".join(lines) + "\n"
