from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .mandarin import classify_alias
from .oto import OtoEntry, load_voicebank

MIN_GROUP = 20
MIN_SHARE = 0.30
MIN_GAIN = 0.20
MAX_AFFIX = 2


@dataclass(frozen=True)
class InferredAffix:
    directory: str
    prefix: str
    suffix: str
    aliases: int
    readable_before: int
    readable_after: int


def _readable(entries: list[OtoEntry], affixes: tuple[tuple[str, str], ...]) -> int:
    seen = {
        entry.alias
        for entry in entries
        if classify_alias(entry, affixes) is not None
    }
    return len(seen)


def _candidates(aliases: list[str]) -> list[tuple[str, str]]:
    total = len(aliases)
    prefixes: Counter = Counter()
    suffixes: Counter = Counter()
    for alias in aliases:
        for size in range(1, MAX_AFFIX + 1):
            if len(alias) > size:
                prefixes[alias[:size]] += 1
                suffixes[alias[-size:]] += 1
    found = [(p, "") for p, n in prefixes.items() if n >= MIN_SHARE * total]
    found += [("", s) for s, n in suffixes.items() if n >= MIN_SHARE * total]
    return found


def infer_group(
    entries: list[OtoEntry],
    known: tuple[tuple[str, str], ...] = (),
) -> InferredAffix | None:
    aliases = sorted({entry.alias for entry in entries})
    if len(aliases) < MIN_GROUP:
        return None

    base = _readable(entries, known)
    best: tuple[int, float, tuple[str, str]] | None = None
    for pair in _candidates(aliases):
        after = _readable(entries, known + (pair,))
        gain = after - base
        share = sum(
            1 for alias in aliases if alias.startswith(pair[0]) and alias.endswith(pair[1])
        ) / len(aliases)
        key = (gain, share)
        if best is None or key > (best[0], best[1]):
            best = (gain, share, pair)

    if best is None or best[0] < MIN_GAIN * len(aliases):
        return None

    prefix, suffix = best[2]
    return InferredAffix(
        directory="",
        prefix=prefix,
        suffix=suffix,
        aliases=len(aliases),
        readable_before=base,
        readable_after=base + best[0],
    )


def infer_layer_affixes(
    root: Path,
    known: tuple[tuple[str, str], ...] = (),
) -> list[InferredAffix]:
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    groups: dict[Path, list[OtoEntry]] = defaultdict(list)
    for entry in entries:
        groups[entry.oto_path.parent].append(entry)

    found: list[InferredAffix] = []
    for directory in sorted(groups):
        result = infer_group(groups[directory], known)
        if result is None:
            continue
        name = "." if directory == root else str(directory.relative_to(root))
        found.append(
            InferredAffix(
                directory=name,
                prefix=result.prefix,
                suffix=result.suffix,
                aliases=result.aliases,
                readable_before=result.readable_before,
                readable_after=result.readable_after,
            )
        )
    return found
