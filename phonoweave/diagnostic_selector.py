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


_LEGAL_SYLLABLES: dict[str, tuple[str, ...]] = {
    "b": ("ba", "bai", "ban", "bang", "bao", "bei", "ben", "beng", "bi", "bian", "biao", "bie", "bin", "bing", "bo", "bu"),
    "p": ("pa", "pai", "pan", "pang", "pao", "pei", "pen", "peng", "pi", "pian", "piao", "pie", "pin", "ping", "po", "pou", "pu"),
    "m": ("ma", "mai", "man", "mang", "mao", "me", "mei", "men", "meng", "mi", "mian", "miao", "mie", "min", "ming", "miu", "mo", "mou", "mu"),
    "f": ("fa", "fan", "fang", "fei", "fen", "feng", "fo", "fou", "fu"),
    "t": ("ta", "tai", "tan", "tang", "tao", "te", "teng", "ti", "tian", "tiao", "tie", "ting", "tong", "tou", "tu", "tuan", "tui", "tun", "tuo"),
    "r": ("ran", "rang", "rao", "re", "ren", "reng", "ri", "rong", "rou", "ru", "ruan", "rui", "run", "ruo"),
    "zh": ("zha", "zhai", "zhan", "zhang", "zhao", "zhe", "zhen", "zheng", "zhi", "zhong", "zhou", "zhu", "zhua", "zhuai", "zhuan", "zhuang", "zhui", "zhun", "zhuo"),
    "ch": ("cha", "chai", "chan", "chang", "chao", "che", "chen", "cheng", "chi", "chong", "chou", "chu", "chua", "chuai", "chuan", "chuang", "chui", "chun", "chuo"),
    "z": ("za", "zai", "zan", "zang", "zao", "ze", "zei", "zen", "zeng", "zi", "zong", "zou", "zu", "zuan", "zui", "zun", "zuo"),
    "c": ("ca", "cai", "can", "cang", "cao", "ce", "cen", "ceng", "ci", "cong", "cou", "cu", "cuan", "cui", "cun", "cuo"),
    "j": ("ji", "jia", "jian", "jiang", "jiao", "jie", "jin", "jing", "jiong", "jiu", "ju", "juan", "jue", "jun"),
    "q": ("qi", "qia", "qian", "qiang", "qiao", "qie", "qin", "qing", "qiong", "qiu", "qu", "quan", "que", "qun"),
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


def _candidates(
    request: SupplementRequest,
    counts: Counter[tuple[str, str]],
    family: str,
) -> list[DiagnosticItem]:
    rows: list[DiagnosticItem] = []
    for syllable in _LEGAL_SYLLABLES.get(request.base_unit, ()):
        final = syllable[len(request.base_unit):]
        if _family(request.base_unit, final) != family:
            continue
        rows.append(
            DiagnosticItem(
                base_unit=request.base_unit,
                final=final,
                syllable=syllable,
                context_family=family,
                role_scope=request.role_scope,
                existing_observations=counts[(request.base_unit, final)],
                replicate=1,
            )
        )
    return sorted(rows, key=lambda item: (item.existing_observations, item.syllable))


def _take_with_replicates(
    candidates: list[DiagnosticItem],
    count: int,
) -> list[DiagnosticItem]:
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
