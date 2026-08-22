from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import unicodedata


@dataclass(frozen=True)
class OtoEntry:
    oto_path: Path
    wav_path: Path
    alias: str
    offset: float
    consonant: float
    cutoff: float
    preutterance: float
    overlap: float
    line_number: int


@dataclass(frozen=True)
class OtoWarning:
    oto_path: Path
    line_number: int
    message: str


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def _normalized_name(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def _wav_lookup(directory: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() == ".wav":
            lookup.setdefault(_normalized_name(path.name), path.resolve())
    return lookup


def parse_oto_line(line: str, oto_path: Path, line_number: int) -> OtoEntry:
    wav_name, sep, payload = line.partition("=")
    if not sep:
        raise ValueError("missing '='")

    fields = payload.rsplit(",", 5)
    if len(fields) != 6:
        raise ValueError("expected alias and five timing values")

    alias, *timing = fields
    try:
        offset, consonant, cutoff, preutterance, overlap = map(float, timing)
    except ValueError as exc:
        raise ValueError("invalid timing value") from exc

    wav_path = (oto_path.parent / wav_name.strip()).resolve()
    return OtoEntry(
        oto_path=oto_path,
        wav_path=wav_path,
        alias=alias.strip(),
        offset=offset,
        consonant=consonant,
        cutoff=cutoff,
        preutterance=preutterance,
        overlap=overlap,
        line_number=line_number,
    )


def load_oto(path: Path) -> tuple[list[OtoEntry], list[OtoWarning]]:
    path = path.resolve()
    entries: list[OtoEntry] = []
    warnings: list[OtoWarning] = []
    wavs = _wav_lookup(path.parent)

    for line_number, raw_line in enumerate(_read_text(path).splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        try:
            entry = parse_oto_line(line, path, line_number)
            if not entry.wav_path.exists():
                resolved = wavs.get(_normalized_name(entry.wav_path.name))
                if resolved is not None:
                    entry = replace(entry, wav_path=resolved)
            entries.append(entry)
        except ValueError as exc:
            warnings.append(OtoWarning(path, line_number, str(exc)))

    return entries, warnings


def load_voicebank(root: Path) -> tuple[list[OtoEntry], list[OtoWarning]]:
    root = root.expanduser().resolve()
    oto_files = sorted(path for path in root.rglob("*") if path.is_file() and path.name.lower() == "oto.ini")
    entries: list[OtoEntry] = []
    warnings: list[OtoWarning] = []

    for oto_path in oto_files:
        file_entries, file_warnings = load_oto(oto_path)
        entries.extend(file_entries)
        warnings.extend(file_warnings)

    return entries, warnings
