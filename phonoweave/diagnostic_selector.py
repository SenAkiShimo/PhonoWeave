from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .mandarin import collect_observations, context_for, structure_for
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .supplement_plan import SupplementPlan, SupplementRequest


@dataclass(frozen=True)
class DiagnosticItem:
    base_unit: str
    final: str
    syllable: str
    context_family: str
    role_scope: str | None
    existing_observations: int
    replicate: int


@dataclass(frozen=True)
class DiagnosticSelection:
    items: tuple[DiagnosticItem, ...]
    unfilled: tuple[str, ...]


_LEGAL_FINALS: dict[str, tuple[str, ...]] = {
    "b": ("a", "o", "ai", "ei", "ao", "an", "en", "ang", "eng", "i", "ie", "iao", "ian", "in", "ing", "u"),
    "p": ("a", "o", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "i", "ie", "iao", "ian", "in", "ing", "u"),
    "m": ("a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "i", "ie", "iao", "iu", "ian", "in", "ing", "u"),
    "f": ("a", "o", "ei", "ou", "an", "en", "ang", "eng", "u"),
    "t": ("a", "e", "ai", "ao", "ou", "an", "ang", "eng", "i", "ie", "iao", "iu", "ian", "ing", "u", "ui", "uan", "un", "uo"),
    "r": ("a", "e", "ao", "ou", "an", "en", "ang", "eng", "ong", "i", "u", "ua", "uo", "ui", "uan", "un"),
    "zh": ("a", "e", "i", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong", "u", "ua", "uo", "uai", "ui", "uan", "un", "uang"),
    "ch": ("a", "e", "i", "ai", "ao", "ou", "an", "en", "ang", "eng", "ong", "u", "ua", "uo", "uai", "ui", "uan", "un", "uang"),
    "z": ("a", "e", "i", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong", "u", "uo", "ui", "uan", "un"),
    "c": ("a", "e", "i", "ai", "ao", "ou", "an", "en", "ang", "eng", "ong", "u", "uo", "ui", "uan", "un"),
    "j": ("i", "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "u", "ue", "uan", "un"),
    "q": ("i", "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "u", "ue", "uan", "un"),
}


def _family(base_unit: str, final: str) -> str | None:
    direct = context_for(base_unit, final)
    if direct is not None:
        return direct
    if base_unit in {"b", "p", "t", "m"}:
        if final.startswith("i"):
            return "i_series"
        if final.startswith("u"):
            return "u_series"
        if final.startswith("v") or final == "ü":
            return "v_series"
        return "other"
    if base_unit == "f":
        if final.startswith("u") or final in {"o", "ou", "ong"}:
            return "rounded"
        return "other"
    return None


def _observed_counts(root: Path) -> Counter[tuple[str, str]]:
    entries, _ = load_voicebank(root)
    affixes = affix_pairs(load_prefix_maps(root))
    counts: Counter[tuple[str, str]] = Counter()
    for observation in collect_observations(entries, affixes):
        structure = structure_for(observation)
        if structure.onset is None:
            continue
        counts[(structure.onset, structure.final)] += 1
    return counts


def _candidates(request: SupplementRequest, counts: Counter[tuple[str, str]], family: str) -> list[DiagnosticItem]:
    finals = _LEGAL_FINALS.get(request.base_unit, ())
    rows = [
        DiagnosticItem(
            base_unit=request.base_unit,
            final=final,
            syllable=f"{request.base_unit}{final}",
            context_family=family,
            role_scope=request.role_scope,
            existing_observations=counts[(request.base_unit, final)],
            replicate=1,
        )
        for final in finals
        if _family(request.base_unit, final) == family
    ]
    return sorted(rows, key=lambda item: (item.existing_observations, item.final))


def _take_with_replicates(candidates: list[DiagnosticItem], count: int) -> list[DiagnosticItem]:
    if not candidates or count <= 0:
        return []
    selected: list[DiagnosticItem] = []
    replicate_counts: Counter[str] = Counter()
    for index in range(count):
        candidate = candidates[index % len(candidates)]
        replicate_counts[candidate.syllable] += 1
        selected.append(
            DiagnosticItem(
                base_unit=candidate.base_unit,
                final=candidate.final,
                syllable=candidate.syllable,
                context_family=candidate.context_family,
                role_scope=candidate.role_scope,
                existing_observations=candidate.existing_observations,
                replicate=replicate_counts[candidate.syllable],
            )
        )
    return selected


def select_diagnostic_items(root: Path, plan: SupplementPlan) -> DiagnosticSelection:
    root = root.expanduser().resolve()
    counts = _observed_counts(root)
    selected: list[DiagnosticItem] = []
    unfilled: list[str] = []

    for request in plan.requests:
        for target in request.targets:
            candidates = _candidates(request, counts, target.context_family)
            if not candidates:
                unfilled.append(
                    f"{request.base_unit}:{target.context_family}:requested={target.diagnostic_items}:available=0"
                )
                continue
            selected.extend(_take_with_replicates(candidates, target.diagnostic_items))

    return DiagnosticSelection(items=tuple(selected), unfilled=tuple(unfilled))
