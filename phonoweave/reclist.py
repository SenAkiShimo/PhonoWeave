"""Language-neutral, non-destructive import of text recording lists.

This module parses *text*, not phonemes. Language packs and recording-scheme
adapters will interpret tokens in later stages. No split/merge decision is
made here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


LineKind = Literal["recording", "comment", "blank"]
SeparatorMode = Literal["whitespace", "underscore"]


@dataclass(frozen=True)
class ReclistLine:
    number: int  # 1-based source line number
    raw: str  # line content, excluding the line ending
    kind: LineKind
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class ReclistDocument:
    source: Path | None
    encoding: str | None
    separator: SeparatorMode
    text: str  # original decoded content, including line endings and BOM
    lines: tuple[ReclistLine, ...]

    @property
    def recording_lines(self) -> tuple[ReclistLine, ...]:
        return tuple(line for line in self.lines if line.kind == "recording")

    def to_text(self) -> str:
        """Return the original text without rewriting or normalizing it."""
        return self.text

    def summary(self) -> dict[str, object]:
        recordings = self.recording_lines
        tokens = [token for line in recordings for token in line.tokens]
        return {
            "source": str(self.source) if self.source is not None else None,
            "encoding": self.encoding,
            "separator": self.separator,
            "total_lines": len(self.lines),
            "recording_lines": len(recordings),
            "comment_lines": sum(line.kind == "comment" for line in self.lines),
            "blank_lines": sum(line.kind == "blank" for line in self.lines),
            "token_occurrences": len(tokens),
            "unique_tokens": len(set(tokens)),
        }


def parse_reclist(
    text: str,
    *,
    separator: SeparatorMode = "whitespace",
    source: Path | None = None,
    encoding: str | None = None,
) -> ReclistDocument:
    """Parse a reclist without assuming its language or audio unit semantics.

    In ``whitespace`` mode, underscores are literal token characters.
    In ``underscore`` mode, whitespace and underscores are token separators.
    The returned document always preserves the source text exactly.
    """
    if separator not in ("whitespace", "underscore"):
        raise ValueError(f"unsupported reclist separator: {separator}")

    lines: list[ReclistLine] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        # The BOM belongs to the original content, not to the first token.
        stripped = raw.lstrip("\ufeff").strip()
        if not stripped:
            kind: LineKind = "blank"
            tokens: tuple[str, ...] = ()
        elif stripped.startswith(("#", ";", "//")):
            kind = "comment"
            tokens = ()
        else:
            kind = "recording"
            if separator == "underscore":
                tokens = tuple(item for item in re.split(r"[\s_]+", stripped) if item)
            else:
                tokens = tuple(stripped.split())
        lines.append(ReclistLine(number=number, raw=raw, kind=kind, tokens=tokens))

    return ReclistDocument(
        source=source,
        encoding=encoding,
        separator=separator,
        text=text,
        lines=tuple(lines),
    )


def load_reclist(
    path: Path,
    *,
    separator: SeparatorMode = "whitespace",
    encoding: str | None = None,
) -> ReclistDocument:
    """Read UTF-8 or CP932 templates; other encodings can be selected explicitly.

    CP932 fallback is useful for historical Japanese UTAU recording lists.
    No transcription, phonemization, or speaker analysis is attempted.
    """
    path = path.expanduser().resolve()
    data = path.read_bytes()
    codecs = (encoding,) if encoding is not None else ("utf-8", "cp932")
    last_error: UnicodeDecodeError | None = None
    for codec in codecs:
        try:
            decoded = data.decode(codec)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        return parse_reclist(decoded, source=path, encoding=codec, separator=separator)
    raise ValueError(
        f"cannot decode reclist {path}; try --encoding with a known character set"
    ) from last_error
