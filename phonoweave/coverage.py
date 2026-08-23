from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .mandarin import collect_observations, structure_for
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


_ANALYZED = {
    "sh": "fricative",
    "s": "fricative",
    "x": "fricative",
    "zh": "affricate",
    "ch": "affricate",
    "z": "affricate",
    "c": "affricate",
    "j": "affricate",
    "q": "affricate",
    "r": "rhotic",
}


@dataclass(frozen=True)
class CoverageItem:
    base_unit: str
    observations: int
    status: str
    analyzer: str | None


@dataclass(frozen=True)
class VoicebankCoverage:
    voicebank: Path
    observations: int
    onset_observations: int
    zero_onset_observations: int
    items: tuple[CoverageItem, ...]


def analyze_coverage(root: Path) -> VoicebankCoverage:
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)

    counts: Counter[str] = Counter()
    zero_onset = 0
    for observation in observations:
        structure = structure_for(observation)
        if structure.onset is None:
            zero_onset += 1
            continue
        counts[structure.onset] += 1

    items = tuple(
        CoverageItem(
            base_unit=base,
            observations=count,
            status="analyzed" if base in _ANALYZED else "unsupported",
            analyzer=_ANALYZED.get(base),
        )
        for base, count in sorted(counts.items())
    )
    return VoicebankCoverage(
        voicebank=root,
        observations=len(observations),
        onset_observations=sum(counts.values()),
        zero_onset_observations=zero_onset,
        items=items,
    )
