from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .coverage import _ANALYZED
from .mandarin import collect_observations, structure_for
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


@dataclass(frozen=True)
class FinalCoverage:
    final: str
    observations: int
    oto_sets: tuple[str, ...]


@dataclass(frozen=True)
class ContextAuditItem:
    base_unit: str
    status: str
    observations: int
    finals: tuple[FinalCoverage, ...]


@dataclass(frozen=True)
class ContextAudit:
    voicebank: Path
    items: tuple[ContextAuditItem, ...]
    zero_onset_observations: int


def _oto_set_name(root: Path, oto_path: Path) -> str:
    directory = oto_path.parent
    if directory == root:
        return "."
    try:
        return str(directory.relative_to(root))
    except ValueError:
        return str(directory)


def audit_contexts(
    root: Path,
    base_unit: str | None = None,
    unsupported_only: bool = False,
) -> ContextAudit:
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    oto_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    zero_onset = 0

    for observation in observations:
        structure = structure_for(observation)
        base = structure.onset
        if base is None:
            zero_onset += 1
            continue
        if base_unit is not None and base != base_unit:
            continue
        if unsupported_only and base in _ANALYZED:
            continue
        counts[base][structure.final] += 1
        oto_sets[(base, structure.final)].add(
            _oto_set_name(root, observation.entry.oto_path)
        )

    items: list[ContextAuditItem] = []
    for base in sorted(counts):
        final_items = tuple(
            FinalCoverage(
                final=final,
                observations=count,
                oto_sets=tuple(sorted(oto_sets[(base, final)])),
            )
            for final, count in sorted(counts[base].items())
        )
        items.append(
            ContextAuditItem(
                base_unit=base,
                status="analyzed" if base in _ANALYZED else "unsupported",
                observations=sum(counts[base].values()),
                finals=final_items,
            )
        )

    return ContextAudit(
        voicebank=root,
        items=tuple(items),
        zero_onset_observations=zero_onset,
    )
