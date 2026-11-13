from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .mandarin import collect_observations, context_for, structure_for
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .synthesis_inventory import SynthesisInventory, SynthesisUnit

ZERO_ONSET = "0"
_ROLES = ("initial", "internal")
_UNBOUNDED = 10**9


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
    def __init__(
        self,
        inventory: SynthesisInventory | None = None,
        collapse_unresolved: bool = False,
    ) -> None:
        self._by_base: dict[str, list[SynthesisUnit]] = {}
        self.collapsed: set[str] = set()
        if inventory is not None:
            for unit in inventory.units:
                if collapse_unresolved and unit.decision == "unresolved":
                    self.collapsed.add(unit.base_unit)
                    continue
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


class _Flow:
    def __init__(self, size: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(size)]

    def add(self, tail: int, head: int, capacity: int, cost: int) -> tuple[int, int]:
        forward = [head, capacity, cost, len(self.graph[head])]
        backward = [tail, 0, -cost, len(self.graph[tail])]
        self.graph[tail].append(forward)
        self.graph[head].append(backward)
        return tail, len(self.graph[tail]) - 1

    def flow_on(self, edge: tuple[int, int]) -> int:
        tail, position = edge
        head, _, _, reverse = self.graph[tail][position]
        return self.graph[head][reverse][1]

    def run(self, source: int, sink: int) -> int:
        size = len(self.graph)
        total = 0
        while True:
            distance = [_UNBOUNDED] * size
            distance[source] = 0
            previous: list[tuple[int, int] | None] = [None] * size
            queue = deque([source])
            queued = [False] * size
            queued[source] = True
            while queue:
                node = queue.popleft()
                queued[node] = False
                for position, (head, capacity, cost, _) in enumerate(self.graph[node]):
                    if capacity > 0 and distance[node] + cost < distance[head]:
                        distance[head] = distance[node] + cost
                        previous[head] = (node, position)
                        if not queued[head]:
                            queued[head] = True
                            queue.append(head)
            if distance[sink] == _UNBOUNDED:
                return total
            push = _UNBOUNDED
            node = sink
            while node != source:
                tail, position = previous[node]
                push = min(push, self.graph[tail][position][1])
                node = tail
            node = sink
            while node != source:
                tail, position = previous[node]
                edge = self.graph[tail][position]
                edge[1] -= push
                self.graph[edge[0]][edge[3]][1] += push
                node = tail
            total += push


@dataclass
class _Plan:
    syllables: tuple[Syllable, ...]
    initial: list[str]
    internal: list[str]
    adjacency: list[list[list[int]]]
    unit_nodes: int


def _plan(
    syllables: tuple[Syllable, ...],
    units: UnitMap,
    requirements: Requirements,
) -> _Plan:
    initial = [units.unit(item, "initial") for item in syllables]
    internal = [units.unit(item, "internal") for item in syllables]
    unit_ids = {name: number for number, name in enumerate(sorted(set(internal)))}
    final_names = sorted({item.final for item in syllables})
    count = len(unit_ids)
    final_ids = {name: count + number for number, name in enumerate(final_names)}
    nodes = count + len(final_names)
    source, sink = nodes, nodes + 1

    out_bound = [0] * nodes
    in_bound = [0] * nodes
    for position, item in enumerate(syllables):
        tail = unit_ids[internal[position]]
        head = final_ids[item.final]
        out_bound[tail] += 1
        in_bound[head] += 1
    for final, unit in requirements.vc:
        tail = final_ids[final]
        head = unit_ids[unit]
        out_bound[tail] += 1
        in_bound[head] += 1

    network = _Flow(nodes + 2)
    syllable_edges = []
    for position, item in enumerate(syllables):
        syllable_edges.append(
            network.add(unit_ids[internal[position]], final_ids[item.final], _UNBOUNDED, 1)
        )
    connection_edges = {}
    for final in final_names:
        for unit in unit_ids:
            connection_edges[(final, unit)] = network.add(
                final_ids[final], unit_ids[unit], _UNBOUNDED, 1
            )
    demand = 0
    for node in range(nodes):
        excess = in_bound[node] - out_bound[node]
        if excess > 0:
            network.add(source, node, excess, 0)
            demand += excess
        elif excess < 0:
            network.add(node, sink, -excess, 0)
    if network.run(source, sink) != demand:
        raise RuntimeError("could not balance the connection graph")

    adjacency: list[list[list[int]]] = [[] for _ in range(nodes)]
    for position, edge in enumerate(syllable_edges):
        item = syllables[position]
        adjacency[unit_ids[internal[position]]].append(
            [final_ids[item.final], 1 + network.flow_on(edge), position]
        )
    for (final, unit), edge in connection_edges.items():
        multiplicity = (1 if (final, unit) in requirements.vc else 0) + network.flow_on(edge)
        if multiplicity:
            adjacency[final_ids[final]].append([unit_ids[unit], multiplicity, -1])
    return _Plan(syllables, initial, internal, adjacency, count)


def _circuits(plan: _Plan, rng: random.Random) -> list[list[int]]:
    adjacency = [[list(edge) for edge in edges] for edges in plan.adjacency]
    for edges in adjacency:
        rng.shuffle(edges)
    pointer = [0] * len(adjacency)
    circuits: list[list[int]] = []
    for origin in rng.sample(range(len(adjacency)), len(adjacency)):
        if not any(edge[1] for edge in adjacency[origin]):
            continue
        stack = [(origin, -1)]
        popped: list[int] = []
        while stack:
            node, via = stack[-1]
            edges = adjacency[node]
            position = pointer[node]
            while position < len(edges) and edges[position][1] == 0:
                position += 1
            pointer[node] = position
            if position < len(edges):
                edge = edges[position]
                edge[1] -= 1
                stack.append((edge[0], edge[2]))
            else:
                popped.append(stack.pop()[1])
        circuit = [value for value in reversed(popped) if value >= 0]
        if circuit:
            circuits.append(circuit)
    return circuits


def _cut(
    circuit: list[int],
    plan: _Plan,
    rng: random.Random,
    max_syllables: int,
    weight: float,
    seen_start: set[str],
    seen_end: set[str],
) -> list[list[int]]:
    offset = rng.randrange(len(circuit))
    ring = circuit[offset:] + circuit[:offset]
    size = len(ring)
    if size + 1 <= max_syllables:
        return [ring + [ring[0]]]

    def gain(position: int) -> int:
        item = plan.syllables[ring[position]]
        return (plan.initial[ring[position]] not in seen_start) + (item.final not in seen_end)

    def mark(position: int) -> None:
        seen_start.add(plan.initial[ring[position]])
        seen_end.add(plan.syllables[ring[position]].final)

    mark(0)
    cuts = [0]
    reach = max_syllables - 1
    while cuts[-1] + reach < size:
        last = cuts[-1]
        choice = max(
            range(last + 1, last + reach + 1),
            key=lambda position: weight * gain(position) + (position - last) + rng.random() * 0.5,
        )
        mark(choice)
        cuts.append(choice)
    cuts.append(size)
    lines = []
    for left, right in zip(cuts, cuts[1:]):
        segment = ring[left:right + 1] if right < size else ring[left:] + [ring[0]]
        lines.append(segment)
    return lines


def _repair(
    lines: list[list[int]],
    plan: _Plan,
    requirements: Requirements,
    units: UnitMap,
) -> list[list[int]]:
    syllables = plan.syllables
    by_text = {item.text: position for position, item in enumerate(syllables)}
    while True:
        missing = missing_requirements(
            tuple(tuple(syllables[position] for position in line) for line in lines),
            requirements,
            units,
        )
        if not any(missing.values()):
            return lines
        need_cv = set(map(tuple, missing["cv"]))
        need_start = set(missing["start"])
        need_end = set(missing["end"])

        def single(position: int) -> int:
            item = syllables[position]
            return (
                ((item.text, plan.initial[position]) in need_cv)
                + (plan.initial[position] in need_start)
                + (item.final in need_end)
            )

        best = max(range(len(syllables)), key=single)
        if single(best) > 0:
            lines.append([best])
        elif missing["vc"]:
            final, unit = missing["vc"][0]
            before = next(
                position for position, item in enumerate(syllables) if item.final == final
            )
            after = [
                position for position in range(len(syllables)) if plan.internal[position] == unit
            ]
            target = max(after, key=lambda position: (syllables[position].text, unit) in need_cv)
            lines.append([before, target])
        else:
            text, unit = missing["cv"][0]
            target = by_text[text]
            lines.append([target] if plan.initial[target] == unit else [0, target])


def _prune(
    lines: list[list[int]],
    requirements: Requirements,
    plan: _Plan,
    units: UnitMap,
) -> list[list[int]]:
    kept = list(lines)
    for candidate in sorted(lines, key=len):
        rest = [line for line in kept if line is not candidate]
        if not rest:
            continue
        coverage = tuple(tuple(plan.syllables[position] for position in line) for line in rest)
        if not any(missing_requirements(coverage, requirements, units).values()):
            kept = rest
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
    plan = _plan(syllables, units, requirements)
    bound = lower_bound(requirements, max_syllables)

    best: list[list[int]] | None = None
    for attempt in range(max(restarts, 1)):
        rng = random.Random(seed * 1_000_003 + attempt)
        weight = (1.0, 2.0, 3.0, 5.0)[attempt % 4]
        seen_start: set[str] = set()
        seen_end: set[str] = set()
        lines: list[list[int]] = []
        for circuit in _circuits(plan, rng):
            lines.extend(_cut(circuit, plan, rng, max_syllables, weight, seen_start, seen_end))
        lines = _repair(lines, plan, requirements, units)
        if best is None or (len(lines), sum(map(len, lines))) < (len(best), sum(map(len, best))):
            best = lines
            if len(best) == bound:
                break

    assert best is not None
    best = _prune(best, requirements, plan, units)
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
    alternative: Reclist | None = None,
    alternative_mode: str | None = None,
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
        "collapsed_unresolved": sorted(units.collapsed),
    }
    if alternative is not None:
        report["alternative"] = {
            "unresolved": alternative_mode,
            "lines": len(alternative.lines),
            "vc": len(alternative.requirements.vc),
            "lower_bound": alternative.lower_bound,
        }
    if baseline is not None:
        report["baseline_unit_per_onset"] = {
            "lines": len(baseline.lines),
            "vc": len(baseline.requirements.vc),
            "lower_bound": baseline.lower_bound,
        }
    return report
