from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .mandarin import collect_observations, normalize_alias, structure_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


@dataclass(frozen=True)
class SourceFinalAudit:
    final: str
    observations: int
    unique_segments: int
    roles: dict[str, int]
    oto_sets: tuple[str, ...]


@dataclass(frozen=True)
class SourceBaseAudit:
    base_unit: str
    observations: int
    unique_wavs: int
    unique_segments: int
    roles: dict[str, int]
    finals: tuple[SourceFinalAudit, ...]


@dataclass(frozen=True)
class SourceIdentityLabel:
    base_unit: str
    final: str
    role: str
    alias: str
    oto_set: str


@dataclass(frozen=True)
class SourceSharedSegment:
    wav_path: Path
    start_ms: float
    end_ms: float
    observations: int
    status: str
    labels: tuple[SourceIdentityLabel, ...]


@dataclass(frozen=True)
class SourceAudit:
    voicebank: Path
    bases: tuple[SourceBaseAudit, ...]
    shared_segments: tuple[SourceSharedSegment, ...]


def _oto_set_name(root: Path, entry: OtoEntry) -> str:
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


def _identity_status(labels: tuple[SourceIdentityLabel, ...]) -> str:
    identities = {
        (label.base_unit, label.final, label.role)
        for label in labels
    }
    return "duplicate" if len(identities) == 1 else "ambiguous"


def audit_sources(root: Path, base_units: tuple[str, ...]) -> SourceAudit:
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)
    requested = set(base_units)

    grouped: dict[str, list] = defaultdict(list)
    shared: dict[tuple[Path, float, float], list[SourceIdentityLabel]] = defaultdict(list)
    for observation in observations:
        structure = structure_for(observation)
        if structure.onset not in requested:
            continue
        role = _alias_role(observation.entry.alias, affixes)
        grouped[structure.onset].append((observation, structure.final))
        shared[_segment_key(observation.entry)].append(
            SourceIdentityLabel(
                base_unit=structure.onset,
                final=structure.final,
                role=role,
                alias=observation.entry.alias,
                oto_set=_oto_set_name(root, observation.entry),
            )
        )

    bases: list[SourceBaseAudit] = []
    for base in base_units:
        rows = grouped.get(base, [])
        role_counts = Counter(
            _alias_role(observation.entry.alias, affixes)
            for observation, _ in rows
        )
        by_final: dict[str, list] = defaultdict(list)
        for observation, final in rows:
            by_final[final].append(observation)

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
                    oto_sets=tuple(sorted({
                        _oto_set_name(root, row.entry)
                        for row in final_rows
                    })),
                )
            )

        observations_for_base = [observation for observation, _ in rows]
        bases.append(
            SourceBaseAudit(
                base_unit=base,
                observations=len(rows),
                unique_wavs=len({row.entry.wav_path for row in observations_for_base}),
                unique_segments=len({_segment_key(row.entry) for row in observations_for_base}),
                roles=dict(sorted(role_counts.items())),
                finals=tuple(finals),
            )
        )

    shared_segments: list[SourceSharedSegment] = []
    for (wav_path, start_ms, end_ms), labels_list in shared.items():
        if len(labels_list) < 2:
            continue
        labels = tuple(sorted(
            labels_list,
            key=lambda item: (
                item.base_unit,
                item.final,
                item.role,
                item.alias,
                item.oto_set,
            ),
        ))
        shared_segments.append(
            SourceSharedSegment(
                wav_path=wav_path,
                start_ms=start_ms,
                end_ms=end_ms,
                observations=len(labels),
                status=_identity_status(labels),
                labels=labels,
            )
        )

    shared_segments.sort(
        key=lambda item: (str(item.wav_path), item.start_ms, item.end_ms)
    )
    return SourceAudit(
        voicebank=root,
        bases=tuple(bases),
        shared_segments=tuple(shared_segments),
    )
