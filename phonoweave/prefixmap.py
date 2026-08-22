from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re


_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
_NOTE_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class PrefixMapRule:
    color: str
    prefix: str
    suffix: str
    tones: tuple[str, ...]
    tone_ranges: tuple[str, ...]


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def note_to_midi(note: str) -> int | None:
    match = _NOTE_RE.match(note.strip())
    if match is None:
        return None

    name, accidental, octave_text = match.groups()
    semitone = _NOTE_BASE[name.upper()]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1

    octave = int(octave_text)
    return (octave + 1) * 12 + semitone


def midi_to_note(midi: int) -> str:
    octave = midi // 12 - 1
    return f"{_NOTE_NAMES[midi % 12]}{octave}"


def _compress_tones(tones: list[str]) -> tuple[str, ...]:
    midi = sorted({value for tone in tones if (value := note_to_midi(tone)) is not None})
    if not midi:
        return tuple(tones)

    ranges: list[str] = []
    start = midi[0]
    end = midi[0]
    for value in midi[1:]:
        if value == end + 1:
            end = value
            continue
        ranges.append(midi_to_note(start) if start == end else f"{midi_to_note(start)}-{midi_to_note(end)}")
        start = end = value
    ranges.append(midi_to_note(start) if start == end else f"{midi_to_note(start)}-{midi_to_note(end)}")
    return tuple(ranges)


def _load_map(path: Path, color: str) -> list[PrefixMapRule]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for raw_line in _read_text(path).splitlines():
        fields = raw_line.rstrip("\r\n").split("\t")
        if len(fields) != 3:
            continue
        tone, prefix, suffix = fields
        grouped[(prefix, f"{color}{suffix}")].append(tone.strip())

    rules: list[PrefixMapRule] = []
    for (prefix, suffix), tones in grouped.items():
        rules.append(
            PrefixMapRule(
                color=color,
                prefix=prefix,
                suffix=suffix,
                tones=tuple(tones),
                tone_ranges=_compress_tones(tones),
            )
        )
    return rules


def load_prefix_maps(root: Path) -> list[PrefixMapRule]:
    root = root.expanduser().resolve()
    rules: list[PrefixMapRule] = []

    main_map = root / "prefix.map"
    if main_map.is_file():
        rules.extend(_load_map(main_map, ""))

    map_dir = root / "prefix"
    if map_dir.is_dir():
        for path in sorted(map_dir.glob("*.map")):
            rules.extend(_load_map(path, path.stem))

    return rules


def affix_pairs(rules: list[PrefixMapRule]) -> tuple[tuple[str, str], ...]:
    pairs = {(rule.prefix, rule.suffix) for rule in rules}
    return tuple(sorted(pairs, key=lambda pair: len(pair[0]) + len(pair[1]), reverse=True))
