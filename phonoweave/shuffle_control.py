from __future__ import annotations

import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .mandarin import classify_alias
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps

SPLIT = "split_recommended"


def build_shuffled_copy(root: Path, target: Path, seed: int) -> int:
    """Write a copy of the voicebank whose CV aliases have their finals permuted
    among entries that share onset, pitch layer and alias pattern. Audio files are
    linked, not copied. Returns the number of aliases that changed."""
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    affixes = affix_pairs(load_prefix_maps(root))
    rng = random.Random(seed)

    groups: dict[tuple, list[tuple[int, str, str]]] = defaultdict(list)
    for index, entry in enumerate(entries):
        observation = classify_alias(entry, affixes)
        if observation is None:
            continue
        syllable = observation.base_unit + observation.final
        position = entry.alias.lower().find(syllable)
        if position < 0:
            continue
        pattern = entry.alias[:position] + "\0" + entry.alias[position + len(syllable):]
        groups[(entry.oto_path, observation.base_unit, pattern)].append(
            (index, observation.base_unit, observation.final)
        )

    renamed: dict[int, str] = {}
    for (_, base, pattern), members in groups.items():
        finals = [final for _, _, final in members]
        rng.shuffle(finals)
        for (index, _, _), final in zip(members, finals):
            renamed[index] = pattern.replace("\0", base + final)

    changed = sum(1 for index, alias in renamed.items() if alias != entries[index].alias)

    by_file: dict[Path, list[str]] = defaultdict(list)
    for index, entry in enumerate(entries):
        alias = renamed.get(index, entry.alias)
        wav = os.path.relpath(entry.wav_path, entry.oto_path.parent)
        by_file[entry.oto_path].append(
            f"{wav}={alias},{entry.offset:g},{entry.consonant:g},{entry.cutoff:g},"
            f"{entry.preutterance:g},{entry.overlap:g}\n"
        )

    for oto_path, lines in by_file.items():
        directory = target / oto_path.parent.relative_to(root)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "oto.ini").write_text("".join(lines), encoding="utf-8")
        for source in oto_path.parent.iterdir():
            if source.is_file() and source.suffix.lower() == ".wav":
                link = directory / source.name
                if not link.exists():
                    link.symlink_to(source.resolve())

    for name in ("prefix.map",):
        if (root / name).is_file():
            shutil.copy2(root / name, target / name)
    prefix_dir = root / "prefix"
    if prefix_dir.is_dir():
        shutil.copytree(prefix_dir, target / "prefix", dirs_exist_ok=True)
    return changed


def split_calls(decisions) -> dict[str, str]:
    return {item.base_unit: item.decision for item in decisions}


def summarize(real: dict[str, str], runs: list[dict[str, str]]) -> list[str]:
    lines = ["Label-shuffle control (finals permuted within onset, layer and alias pattern)"]
    lines.append(f"{'unit':<8}{'real call':<20}shuffled runs calling a split")
    false_calls = 0
    cells = 0
    for unit in sorted(real):
        hits = sum(run.get(unit) == SPLIT for run in runs)
        false_calls += hits
        cells += len(runs)
        lines.append(f"{unit:<8}{real[unit]:<20}{hits}/{len(runs)}")
    rate = false_calls / cells if cells else 0.0
    lines.append("")
    lines.append(f"Splits called on shuffled data: {false_calls}/{cells} ({rate:.0%})")
    return lines
