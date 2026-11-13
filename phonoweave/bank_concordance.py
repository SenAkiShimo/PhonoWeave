from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

from .mandarin import _BASES
from .reclist import STOP_ONSETS, Syllable, UnitMap
from .synthesis_inventory import SynthesisInventory


_FAMILY_ORDER = ("i", "u", "o")
_BANK_SUFFIX = {"": "plain", "y": "y", "w": "w"}


@dataclass(frozen=True)
class ConcordanceRow:
    base: str
    ours: str
    bank: str
    status: str
    suggested: tuple[str, ...] = ()
    suggested_vc: int = 0


def bank_variants(audit: dict) -> dict[str, set[str]]:
    """Onset variants the voicebank's own VC labels distinguish: plain, y (before i), w (before u)."""
    found: dict[str, set[str]] = defaultdict(set)
    bases = sorted(_BASES, key=len, reverse=True)
    for row in audit["systematic_labels"]:
        label = row["label"]
        for base in bases:
            suffix = label[len(base):] if label.startswith(base) else None
            if suffix in _BANK_SUFFIX:
                found[base].add(_BANK_SUFFIX[suffix])
                break
    return dict(found)


def _family(base: str, final: str) -> str | None:
    if final.startswith("i"):
        return "i"
    if final.startswith("u"):
        return "u"
    if final.startswith(("v", "ü")):
        return "u" if base in {"j", "q", "x"} else None
    return "o"


def _our_units(
    base: str,
    syllables: tuple[Syllable, ...],
    units: UnitMap,
) -> dict[str, str]:
    votes: dict[str, Counter] = defaultdict(Counter)
    for item in syllables:
        if item.onset != base:
            continue
        family = _family(base, item.final)
        if family is not None:
            votes[family][units.unit(item, "internal")] += 1
    return {family: counter.most_common(1)[0][0] for family, counter in votes.items()}


def _bank_units(variants: set[str], families: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for family in families:
        if family == "i":
            choice = "y" if "y" in variants else "plain" if "plain" in variants else None
        elif family == "u":
            choice = "w" if "w" in variants else "plain" if "plain" in variants else None
        else:
            choice = "plain" if "plain" in variants else None
        if choice is not None:
            out[family] = choice
    return out


def _separated(units: dict[str, str]) -> frozenset[tuple[str, str]]:
    return frozenset(
        (a, b)
        for a, b in combinations(_FAMILY_ORDER, 2)
        if a in units and b in units and units[a] != units[b]
    )


def _groups(units: dict[str, str]) -> str:
    grouped: dict[str, list[str]] = defaultdict(list)
    for family in _FAMILY_ORDER:
        if family in units:
            grouped[units[family]].append(family)
    return " | ".join(f"{name}:{','.join(families)}" for name, families in grouped.items())


def _suggested_labels(base: str, ours: dict[str, str], variants: set[str]) -> tuple[str, ...]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for family, unit in ours.items():
        grouped[unit].add(family)
    labels: list[str] = []
    for families in grouped.values():
        if "o" in families:
            continue
        if "u" in families and "i" not in families:
            variant = "w"
        elif "i" in families and "u" not in families:
            variant = "y"
        else:
            continue
        if variant not in variants:
            labels.append(f"{base}{variant}")
    return tuple(sorted(labels))


def compare_with_bank(
    audit: dict,
    inventory: SynthesisInventory,
    syllables: tuple[Syllable, ...],
) -> list[ConcordanceRow]:
    variants = bank_variants(audit)
    effective = UnitMap(inventory, collapse_unresolved=True)
    unresolved = {unit.base_unit for unit in inventory.units if unit.decision == "unresolved"}
    onsets = sorted({item.onset for item in syllables if item.onset})

    rows: list[ConcordanceRow] = []
    for base in onsets:
        ours = _our_units(base, syllables, effective)
        families = set(ours)
        bank = _bank_units(variants.get(base, set()), families)
        ours_pairs = _separated(ours)
        bank_pairs = _separated(bank)
        ours_split = bool(ours_pairs)
        bank_split = len(set(bank.values())) > 1

        if ours_split and bank_split:
            comparable = _separated({f: u for f, u in ours.items() if f in bank})
            status = "same axis" if comparable == bank_pairs else "different axis"
        elif ours_split:
            status = "we split, voicebank does not"
        elif bank_split:
            status = (
                "voicebank splits, we have no evidence"
                if base in unresolved
                else "voicebank splits, we do not"
            )
        else:
            status = "neither splits" + (" (ours unresolved)" if base in unresolved else "")
        suggested = (
            _suggested_labels(base, ours, variants.get(base, set())) if ours_split else ()
        )
        rows.append(
            ConcordanceRow(
                base=base,
                ours=_groups(ours) if ours_split else ("unresolved" if base in unresolved else "single"),
                bank=_groups(bank) if bank_split else "single",
                status=status,
                suggested=suggested,
                suggested_vc=(
                    0 if base in STOP_ONSETS else len(suggested) * len(audit.get("finals", []))
                ),
            )
        )
    return rows


def format_concordance(rows: list[ConcordanceRow]) -> str:
    counts = Counter(row.status for row in rows)
    lines = ["Onset units: ours vs the voicebank's own VC labels"]
    for row in rows:
        lines.append(f"  {row.base}: ours [{row.ours}]; voicebank [{row.bank}] -> {row.status}")
    lines.append("")
    for status, count in sorted(counts.items()):
        lines.append(f"  {status}: {count}")

    extra = [row for row in rows if row.suggested]
    if extra:
        lines.append("")
        lines.append("Splits recommended here that the voicebank does not have (VC entries per pitch layer):")
        for row in extra:
            note = f"{row.suggested_vc} VC entries" if row.suggested_vc else "no VC entries (stop connections are shared); CV entries already exist"
            lines.append(f"  {row.base}: add {', '.join(row.suggested)} -> {note}")
    return "\n".join(lines)
