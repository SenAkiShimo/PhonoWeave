from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .mandarin import collect_observations, context_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import PrefixMapRule, affix_pairs, load_prefix_maps


@dataclass(frozen=True)
class SubbankSummary:
    name: str
    path: Path
    oto_files: int
    entries: int
    valid_entries: int
    missing_wavs: int
    observations: int
    groups: dict[str, dict[str, int]]


@dataclass(frozen=True)
class InspectionResult:
    root: Path
    oto_files: int
    entries: int
    valid_entries: int
    missing_wavs: int
    parse_warnings: int
    observations: int
    groups: dict[str, dict[str, int]]
    subbanks: list[SubbankSummary]
    prefix_rules: list[PrefixMapRule]


def _group_counts(
    entries: list[OtoEntry],
    affixes: tuple[tuple[str, str], ...],
) -> tuple[int, dict[str, dict[str, int]]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    observations = collect_observations(valid_entries, affixes)
    for observation in observations:
        context = context_for(observation.base_unit, observation.final)
        if context is not None:
            grouped[observation.base_unit][context] += 1
    return len(observations), {base: dict(counter) for base, counter in sorted(grouped.items())}


def _subbank_name(root: Path, directory: Path) -> str:
    if directory == root:
        return "."
    return str(directory.relative_to(root))


def inspect_voicebank(root: Path) -> InspectionResult:
    root = root.expanduser().resolve()
    entries, warnings = load_voicebank(root)
    prefix_rules = load_prefix_maps(root)
    affixes = affix_pairs(prefix_rules)

    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    missing_wavs = len(entries) - len(valid_entries)
    oto_paths = {entry.oto_path for entry in entries} | {warning.oto_path for warning in warnings}
    observations, groups = _group_counts(entries, affixes)

    by_directory: dict[Path, list[OtoEntry]] = defaultdict(list)
    for entry in entries:
        by_directory[entry.oto_path.parent].append(entry)

    subbanks: list[SubbankSummary] = []
    for directory in sorted(by_directory):
        bank_entries = by_directory[directory]
        bank_valid_entries = [entry for entry in bank_entries if entry.wav_path.exists()]
        bank_observations, bank_groups = _group_counts(bank_entries, affixes)
        subbanks.append(
            SubbankSummary(
                name=_subbank_name(root, directory),
                path=directory,
                oto_files=len({entry.oto_path for entry in bank_entries}),
                entries=len(bank_entries),
                valid_entries=len(bank_valid_entries),
                missing_wavs=len(bank_entries) - len(bank_valid_entries),
                observations=bank_observations,
                groups=bank_groups,
            )
        )

    return InspectionResult(
        root=root,
        oto_files=len(oto_paths),
        entries=len(entries),
        valid_entries=len(valid_entries),
        missing_wavs=missing_wavs,
        parse_warnings=len(warnings),
        observations=observations,
        groups=groups,
        subbanks=subbanks,
        prefix_rules=prefix_rules,
    )
