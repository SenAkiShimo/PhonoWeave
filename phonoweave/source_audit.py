from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .mandarin import collect_observations, normalize_alias
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


@dataclass(frozen=True)
class SourceFinalAudit:
    final: str
    observations: int
    unique_segments: int
    roles: dict[str, int]
    subbanks: tuple[str, ...]


@dataclass(frozen=True)
class SourceBaseAudit:
    base_unit: str
    observations: int
    unique_wavs: int
    unique_segments: int
    roles: dict[str, int]
    finals: tuple[SourceFinalAudit, ...]


@dataclass(frozen=True)
class SourceAudit:
    voicebank: Path
    bases: tuple[SourceBaseAudit, ...]


def _subbank_name(root: Path, entry: OtoEntry) -> str:
    directory = entry.oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def _segment_key(entry: OtoEntry) -> tuple[Path, float, float]:
    return (
        entry.wav_path,
        round(entry.offset, 3),
        round(entry.offset + entry.preutterance, 3),
    )


def _alias_role(alias: str, affixes: set[tuple[str, str]]) -> str:
    normalized = normalize_alias(alias, affixes).strip()
    return "initial" if normalized.startswith("-") else "internal"


def audit_sources(root: Path, base_units: tuple[str, ...]) -> SourceAudit:
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)
    requested = set(base_units)

    grouped: dict[str, list] = defaultdict(list)
    for observation in observations:
        if observation.base_unit in requested:
            grouped[observation.base_unit].append(observation)

    bases: list[SourceBaseAudit] = []
    for base in base_units:
        rows = grouped.get(base, [])
        role_counts = Counter(
            _alias_role(row.entry.alias, affixes)
            for row in rows
        )
        by_final: dict[str, list] = defaultdict(list)
        for row in rows:
            by_final[row.final].append(row)

        finals: list[SourceFinalAudit] = []
        for final in sorted(by_final):
            final_rows = by_final[final]
            finals.append(
                SourceFinalAudit(
                    final=final,
                    observations=len(final_rows),
                    unique_segments=len({_segment_key(row.entry) for row in final_rows}),
                    roles=dict(sorted(Counter(
                        _alias_role(row.entry.alias, affixes)
                        for row in final_rows
                    ).items())),
                    subbanks=tuple(sorted({
                        _subbank_name(root, row.entry)
                        for row in final_rows
                    })),
                )
            )

        bases.append(
            SourceBaseAudit(
                base_unit=base,
                observations=len(rows),
                unique_wavs=len({row.entry.wav_path for row in rows}),
                unique_segments=len({_segment_key(row.entry) for row in rows}),
                roles=dict(sorted(role_counts.items())),
                finals=tuple(finals),
            )
        )

    return SourceAudit(voicebank=root, bases=tuple(bases))
