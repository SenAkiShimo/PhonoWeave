from __future__ import annotations

from dataclasses import dataclass

from .diagnostic_selector import DiagnosticSelection


@dataclass(frozen=True)
class ReclistLine:
    base_unit: str
    syllable: str
    replicate: int
    role_scope: str | None
    text: str


@dataclass(frozen=True)
class SupplementReclist:
    lines: tuple[ReclistLine, ...]

    def text(self) -> str:
        return "\n".join(line.text for line in self.lines) + ("\n" if self.lines else "")


_CARRIERS = ("a", "e", "o", "ai")


def _carrier(replicate: int) -> str:
    return _CARRIERS[(replicate - 1) % len(_CARRIERS)]


def _line_for(syllable: str, role_scope: str | None, replicate: int) -> str:
    carrier = _carrier(replicate)
    if role_scope == "internal":
        return f"{carrier}_{syllable}_{carrier}"
    if role_scope == "initial":
        return f"{syllable}_{carrier}"
    return f"{syllable}_{carrier}_{syllable}_{carrier}"


def build_supplement_reclist(selection: DiagnosticSelection) -> SupplementReclist:
    lines = tuple(
        ReclistLine(
            base_unit=item.base_unit,
            syllable=item.syllable,
            replicate=item.replicate,
            role_scope=item.role_scope,
            text=_line_for(item.syllable, item.role_scope, item.replicate),
        )
        for item in selection.items
    )
    return SupplementReclist(lines=lines)
