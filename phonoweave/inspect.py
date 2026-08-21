from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .mandarin import collect_observations, context_for
from .oto import load_voicebank


@dataclass(frozen=True)
class InspectionResult:
    root: Path
    oto_files: int
    entries: int
    missing_wavs: int
    parse_warnings: int
    observations: int
    groups: dict[str, dict[str, int]]


def inspect_voicebank(root: Path) -> InspectionResult:
    root = root.expanduser().resolve()
    entries, warnings = load_voicebank(root)
    missing_wavs = sum(1 for entry in entries if not entry.wav_path.exists())
    oto_files = len({entry.oto_path for entry in entries} | {warning.oto_path for warning in warnings})

    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    observations = collect_observations(entries)
    for observation in observations:
        context = context_for(observation.base_unit, observation.final)
        if context is not None:
            grouped[observation.base_unit][context] += 1

    groups = {base: dict(counter) for base, counter in sorted(grouped.items())}
    return InspectionResult(
        root=root,
        oto_files=oto_files,
        entries=len(entries),
        missing_wavs=missing_wavs,
        parse_warnings=len(warnings),
        observations=len(observations),
        groups=groups,
    )
