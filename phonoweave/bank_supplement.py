from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .bank_concordance import ConcordanceRow, _family
from .reclist import Syllable

SPLIT_STATUS = "we split, voicebank does not"
_FAMILY_OF_VARIANT = {"w": "u", "y": "i"}


@dataclass(frozen=True)
class BankSupplement:
    lines: tuple[tuple[Syllable, ...], ...]
    connections: frozenset[tuple[str, str]]
    labels: tuple[str, ...]

    def text(self) -> str:
        return "".join("_".join(item.text for item in line) + "\n" for line in self.lines)


def _targets(
    rows: list[ConcordanceRow],
    syllables: tuple[Syllable, ...],
) -> dict[str, list[Syllable]]:
    targets: dict[str, list[Syllable]] = {}
    for row in rows:
        if row.status != SPLIT_STATUS or not row.suggested_vc:
            continue
        for label in row.suggested:
            family = _FAMILY_OF_VARIANT.get(label[len(row.base):])
            if family is None:
                continue
            found = [
                item for item in syllables
                if item.onset == row.base and _family(row.base, item.final) == family
            ]
            if found:
                targets[label] = sorted(found, key=lambda item: item.text)
    return targets


def build_bank_supplement(
    rows: list[ConcordanceRow],
    syllables: tuple[Syllable, ...],
    max_syllables: int = 10,
) -> BankSupplement:
    """Lines that record the VC connections a recommended split needs and the voicebank lacks.

    Each missing connection is a preceding final followed by a syllable of the new
    onset variant (for example any final before `zu`). Carriers are ordinary syllables
    with the wanted tail; they are not themselves recording targets.
    """
    targets = _targets(rows, syllables)
    tails = sorted({item.tail for item in syllables})
    remaining = {(tail, label) for label in targets for tail in tails}
    required = frozenset(remaining)

    target_texts = {item.text for items in targets.values() for item in items}
    carriers: dict[str, list[Syllable]] = defaultdict(list)
    for item in sorted(syllables, key=lambda item: item.text):
        if item.text not in target_texts:
            carriers[item.tail].append(item)

    def demand(tail: str) -> int:
        return sum(1 for pair in remaining if pair[0] == tail)

    used: dict[str, int] = defaultdict(int)

    def pick_target(label: str) -> Syllable:
        choice = max(
            targets[label],
            key=lambda item: (demand(item.tail) > 0, -used[item.text], demand(item.tail), item.text),
        )
        used[choice.text] += 1
        return choice

    lines: list[tuple[Syllable, ...]] = []
    while remaining:
        tail = max(sorted({pair[0] for pair in remaining}), key=demand)
        pool = carriers.get(tail)
        if not pool:
            remaining = {pair for pair in remaining if pair[0] != tail}
            required = required - {pair for pair in required if pair[0] == tail}
            continue
        line = [pool[0]]
        current = tail
        while len(line) < max_syllables:
            wanted = sorted(label for t, label in remaining if t == current)
            if wanted:
                label = wanted[0]
                remaining.discard((current, label))
                step = pick_target(label)
                line.append(step)
                current = step.tail
                continue
            if len(line) + 2 > max_syllables or not remaining:
                break
            tail = max(sorted({pair[0] for pair in remaining}), key=demand)
            pool = carriers.get(tail)
            if not pool:
                break
            line.append(pool[0])
            current = tail
        lines.append(tuple(line))

    return BankSupplement(
        lines=tuple(lines),
        connections=required,
        labels=tuple(sorted(targets)),
    )


def uncovered_connections(
    supplement: BankSupplement,
    rows: list[ConcordanceRow],
    syllables: tuple[Syllable, ...],
) -> set[tuple[str, str]]:
    targets = _targets(rows, syllables)
    seen: set[tuple[str, str]] = set()
    for line in supplement.lines:
        for before, after in zip(line, line[1:]):
            for label, items in targets.items():
                if after in items:
                    seen.add((before.tail, label))
    return set(supplement.connections) - seen
