from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from .mandarin import collect_observations, context_for, structure_for
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .synthesis_inventory import SynthesisInventory, SynthesisUnit

ZERO_ONSET = "0"
_ROLES = ("initial", "internal")
_REACH_CAP = 40
_REACH_WEIGHT = 1.0
_JITTER = 0.5
_REPEAT_PENALTY = 3.0


@dataclass(frozen=True)
class Syllable:
    text: str
    onset: str
    final: str


@dataclass(frozen=True)
class Requirements:
    cv: frozenset[tuple[str, str]]
    vc: frozenset[tuple[str, str]]
    start: frozenset[str]
    end: frozenset[str]


@dataclass(frozen=True)
class Reclist:
    lines: tuple[tuple[Syllable, ...], ...]
    requirements: Requirements
    lower_bound: int
    max_syllables: int

    def text(self) -> str:
        return "".join("_".join(item.text for item in line) + "\n" for line in self.lines)


def observed_syllables(root: Path) -> tuple[Syllable, ...]:
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    affixes = affix_pairs(load_prefix_maps(root))
    found: dict[tuple[str, str], Syllable] = {}
    for observation in collect_observations(entries, affixes):
        structure = structure_for(observation)
        onset = structure.onset or ""
        key = (onset, structure.final)
        text = f"{observation.base_unit}{observation.final}"
        known = found.get(key)
        if known is None or text < known.text:
            found[key] = Syllable(text=text, onset=onset, final=structure.final)
    return tuple(sorted(found.values(), key=lambda item: item.text))


def _family(base: str, final: str) -> str:
    if base in {"f", "h"}:
        if final.startswith("u") or final in {"o", "ou", "ong"}:
            return "rounded"
        return "other"
    if final.startswith("i"):
        return "i_series"
    if final.startswith("u"):
        return "u_series"
    if base in {"m", "n", "l"} and (final.startswith("v") or final == "ü"):
        return "v_series"
    return "other"


def _articulatory_context(base: str, final: str) -> str | None:
    context = context_for(base, final)
    if context is not None or base not in {"j", "q", "x"}:
        return context
    if final.startswith(("u", "v", "ü")):
        return "rounded"
    if final.startswith("i"):
        return "plain"
    return None


def _context_keys(base: str, final: str, role: str) -> tuple[str, ...]:
    family = _family(base, final)
    keys = [f"{role}:{family}", f"internal:{family}", f"initial:{family}"]
    context = _articulatory_context(base, final)
    if context is not None:
        keys.append(context)
        if base == "x" and context == "plain":
            keys.append("front_unrounded")
    keys.append("rounded" if family in {"u_series", "v_series", "rounded"} else "plain")
    return tuple(dict.fromkeys(keys))


class UnitMap:
    def __init__(self, inventory: SynthesisInventory | None = None) -> None:
        self._by_base: dict[str, list[SynthesisUnit]] = {}
        if inventory is not None:
            for unit in inventory.units:
                self._by_base.setdefault(unit.base_unit, []).append(unit)
        self._cache: dict[tuple[str, str, str], str] = {}
        self.unmapped: set[tuple[str, str]] = set()

    def unit(self, syllable: Syllable, role: str) -> str:
        key = (syllable.onset, syllable.final, role)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._cache[key] = self._resolve(*key)
        return cached

    def _resolve(self, base: str, final: str, role: str) -> str:
        if not base:
            return ZERO_ONSET
        units = self._by_base.get(base)
        if not units:
            return base
        if len(units) == 1:
            return units[0].id
        for key in _context_keys(base, final, role):
            for unit in units:
                if key in unit.contexts:
                    return unit.id
        self.unmapped.add((base, final))
        return base


def build_requirements(
    syllables: tuple[Syllable, ...],
    units: UnitMap,
    waived_vc: frozenset[tuple[str, str]] = frozenset(),
) -> Requirements:
    internal = {units.unit(item, "internal") for item in syllables}
    initial = {units.unit(item, "initial") for item in syllables}
    finals = {item.final for item in syllables}
    return Requirements(
        cv=frozenset(
            (item.text, units.unit(item, role))
            for item in syllables
            for role in _ROLES
        ),
        vc=frozenset((final, unit) for final in finals for unit in internal) - waived_vc,
        start=frozenset(initial),
        end=frozenset(finals),
    )


def lower_bound(requirements: Requirements, max_syllables: int) -> int:
    return max(
        math.ceil(len(requirements.vc) / (max_syllables - 1)),
        math.ceil(len(requirements.cv) / max_syllables),
        len(requirements.start),
        len(requirements.end),
        1,
    )


def line_coverage(
    line: tuple[Syllable, ...],
    units: UnitMap,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], str, str]:
    cv: set[tuple[str, str]] = set()
    vc: set[tuple[str, str]] = set()
    for index, item in enumerate(line):
        role = "initial" if index == 0 else "internal"
        unit = units.unit(item, role)
        cv.add((item.text, unit))
        if index:
            vc.add((line[index - 1].final, unit))
    return cv, vc, units.unit(line[0], "initial"), line[-1].final


def missing_requirements(
    lines: tuple[tuple[Syllable, ...], ...],
    requirements: Requirements,
    units: UnitMap,
) -> dict[str, list]:
    cv: set[tuple[str, str]] = set()
    vc: set[tuple[str, str]] = set()
    start: set[str] = set()
    end: set[str] = set()
    for line in lines:
        line_cv, line_vc, line_start, line_end = line_coverage(line, units)
        cv |= line_cv
        vc |= line_vc
        start.add(line_start)
        end.add(line_end)
    return {
        "cv": sorted(requirements.cv - cv),
        "vc": sorted(requirements.vc - vc),
        "start": sorted(requirements.start - start),
        "end": sorted(requirements.end - end),
    }


@dataclass
class _State:
    cv: set[tuple[str, str]]
    vc: dict[str, set[str]]
    start: set[str]
    end: set[str]

    def empty(self) -> bool:
        return not (self.cv or self.start or self.end or any(self.vc.values()))


@dataclass
class _Index:
    syllables: tuple[Syllable, ...]
    initial: list[str] = field(default_factory=list)
    internal: list[str] = field(default_factory=list)
    by_internal: dict[str, list[int]] = field(default_factory=dict)
    by_initial: dict[str, list[int]] = field(default_factory=dict)
    by_final: dict[str, list[int]] = field(default_factory=dict)


def _index(syllables: tuple[Syllable, ...], units: UnitMap) -> _Index:
    index = _Index(syllables=syllables)
    for position, item in enumerate(syllables):
        first = units.unit(item, "initial")
        inner = units.unit(item, "internal")
        index.initial.append(first)
        index.internal.append(inner)
        index.by_initial.setdefault(first, []).append(position)
        index.by_internal.setdefault(inner, []).append(position)
        index.by_final.setdefault(item.final, []).append(position)
    return index


def _fresh_state(requirements: Requirements) -> _State:
    vc: dict[str, set[str]] = {}
    for final, unit in requirements.vc:
        vc.setdefault(final, set()).add(unit)
    return _State(
        cv=set(requirements.cv),
        vc=vc,
        start=set(requirements.start),
        end=set(requirements.end),
    )


def _build_line(
    state: _State,
    index: _Index,
    rng: random.Random,
    max_syllables: int,
) -> list[int]:
    syllables = index.syllables
    used: set[str] = set()

    def score(position: int, role_units: list[str], final_bonus: bool) -> float:
        item = syllables[position]
        repeat = -_REPEAT_PENALTY if item.text in used else 0.0
        gain = 2.0 if (item.text, role_units[position]) in state.cv else 0.0
        reach = min(len(state.vc.get(item.final, ())), _REACH_CAP) * _REACH_WEIGHT
        ending = 1.5 if final_bonus and item.final in state.end else 0.0
        return gain + reach + ending + repeat + rng.random() * _JITTER

    if state.start:
        unit = min(
            sorted(state.start),
            key=lambda value: (len(index.by_initial[value]), rng.random()),
        )
        pool = index.by_initial[unit]
    elif any(state.vc.values()):
        final = rng.choice(sorted(value for value, units in state.vc.items() if units))
        pool = index.by_final[final]
    else:
        pool = list(range(len(syllables)))

    first = max(pool, key=lambda position: score(position, index.initial, max_syllables == 1))
    line = [first]
    used.add(syllables[first].text)
    state.start.discard(index.initial[first])
    state.cv.discard((syllables[first].text, index.initial[first]))

    while len(line) < max_syllables:
        previous = syllables[line[-1]].final
        pending = state.vc.get(previous)
        if not pending:
            break
        last_slot = len(line) == max_syllables - 1
        options = [
            position
            for unit in sorted(pending)
            for position in index.by_internal[unit]
        ]
        chosen = max(options, key=lambda position: score(position, index.internal, last_slot))
        unit = index.internal[chosen]
        pending.discard(unit)
        state.cv.discard((syllables[chosen].text, unit))
        line.append(chosen)
        used.add(syllables[chosen].text)

    state.end.discard(syllables[line[-1]].final)
    return line


def _prune(
    lines: list[list[int]],
    requirements: Requirements,
    index: _Index,
    units: UnitMap,
) -> list[list[int]]:
    kept = [line for line in lines]
    changed = True
    while changed:
        changed = False
        for candidate in sorted(kept, key=len):
            rest = tuple(
                tuple(index.syllables[position] for position in line)
                for line in kept
                if line is not candidate
            )
            if rest and not any(missing_requirements(rest, requirements, units).values()):
                kept = [line for line in kept if line is not candidate]
                changed = True
                break
    return kept


def build_reclist(
    syllables: tuple[Syllable, ...],
    units: UnitMap,
    max_syllables: int = 10,
    restarts: int = 40,
    seed: int = 0,
    waived_vc: frozenset[tuple[str, str]] = frozenset(),
) -> Reclist:
    if max_syllables < 2:
        raise ValueError("max_syllables must be at least 2")
    if not syllables:
        raise ValueError("no syllables to cover")

    requirements = build_requirements(syllables, units, waived_vc)
    index = _index(syllables, units)
    bound = lower_bound(requirements, max_syllables)

    best: list[list[int]] | None = None
    for attempt in range(max(restarts, 1)):
        rng = random.Random(seed * 1_000_003 + attempt)
        state = _fresh_state(requirements)
        lines: list[list[int]] = []
        while not state.empty():
            line = _build_line(state, index, rng, max_syllables)
            lines.append(line)
        if best is None or (len(lines), sum(map(len, lines))) < (len(best), sum(map(len, best))):
            best = lines
            if len(best) == bound:
                break

    assert best is not None
    best = _prune(best, requirements, index, units)
    result = tuple(tuple(syllables[position] for position in line) for line in best)
    return Reclist(
        lines=result,
        requirements=requirements,
        lower_bound=bound,
        max_syllables=max_syllables,
    )


def reclist_report(
    reclist: Reclist,
    syllables: tuple[Syllable, ...],
    units: UnitMap,
    inventory: SynthesisInventory | None = None,
    baseline: Reclist | None = None,
) -> dict[str, object]:
    requirements = reclist.requirements
    unit_syllables: dict[str, set[str]] = {}
    for item in syllables:
        for role in _ROLES:
            unit_syllables.setdefault(units.unit(item, role), set()).add(item.text)
    decisions = {}
    if inventory is not None:
        decisions = {unit.id: unit.decision for unit in inventory.units}

    report: dict[str, object] = {
        "lines": len(reclist.lines),
        "max_syllables_per_line": reclist.max_syllables,
        "lower_bound": reclist.lower_bound,
        "syllables": len(syllables),
        "requirements": {
            "cv": len(requirements.cv),
            "vc": len(requirements.vc),
            "start": len(requirements.start),
            "end": len(requirements.end),
        },
        "units": {
            unit: {
                "syllables": len(members),
                **({"decision": decisions[unit]} if unit in decisions else {}),
            }
            for unit, members in sorted(unit_syllables.items())
        },
        "unmapped_contexts": sorted(f"{base}+{final}" for base, final in units.unmapped),
    }
    if baseline is not None:
        report["baseline_unit_per_onset"] = {
            "lines": len(baseline.lines),
            "vc": len(baseline.requirements.vc),
            "lower_bound": baseline.lower_bound,
        }
    return report
