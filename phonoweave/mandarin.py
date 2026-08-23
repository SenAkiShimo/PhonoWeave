from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .oto import OtoEntry


_FINALS = (
    "uang", "iong", "iang", "uai", "uan", "iao", "ian", "ang", "eng", "ing", "ong",
    "ua", "uo", "ui", "un", "ia", "ie", "iu", "in", "ve", "van", "vn", "ue",
    "ai", "ei", "ao", "ou", "an", "en", "a", "o", "e", "i", "u", "v", "ü",
)

_BASES = (
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "z", "c", "s", "y", "w",
)

_Y_ZERO_FINALS = {
    "a": "ia",
    "an": "ian",
    "ang": "iang",
    "ao": "iao",
    "e": "ie",
    "i": "i",
    "in": "in",
    "ing": "ing",
    "ong": "iong",
    "ou": "iu",
    "u": "v",
    "uan": "van",
    "ue": "ve",
    "un": "vn",
}

_W_ZERO_FINALS = {
    "a": "ua",
    "ai": "uai",
    "an": "uan",
    "ang": "uang",
    "ei": "ui",
    "en": "un",
    "eng": "ueng",
    "o": "uo",
    "u": "u",
}

_PALATAL_U_FINALS = {
    "u": "v",
    "ue": "ve",
    "uan": "van",
    "un": "vn",
}


@dataclass(frozen=True)
class MandarinObservation:
    entry: OtoEntry
    base_unit: str
    final: str
    alias_token: str


@dataclass(frozen=True)
class MandarinSyllableStructure:
    onset: str | None
    final: str
    orthographic_initial: str | None = None


def _tokens(alias: str) -> list[str]:
    cleaned = alias.strip().lower()
    cleaned = cleaned.replace("-", " ").replace("_", " ")
    return [token for token in re.split(r"\s+", cleaned) if token]


def _clean_token(token: str) -> str:
    return re.sub(r"[^a-züv]", "", token.lower())


def _split_syllable(token: str) -> tuple[str, str] | None:
    token = _clean_token(token)
    if not token:
        return None

    for base in _BASES:
        if token.startswith(base):
            final = token[len(base):]
            if final in _FINALS:
                return base, final
    return None


def normalize_alias(alias: str, affixes: Iterable[tuple[str, str]] = ()) -> str:
    candidates = [alias]
    for prefix, suffix in affixes:
        stripped = alias
        if prefix and stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
        if suffix and stripped.endswith(suffix):
            stripped = stripped[:-len(suffix)]
        if stripped != alias:
            candidates.append(stripped)
    return min(candidates, key=len)


def classify_alias(
    entry: OtoEntry,
    affixes: Iterable[tuple[str, str]] = (),
) -> MandarinObservation | None:
    alias = normalize_alias(entry.alias, affixes)
    tokens = _tokens(alias)

    for token in reversed(tokens):
        parsed = _split_syllable(token)
        if parsed is not None:
            base, final = parsed
            return MandarinObservation(entry=entry, base_unit=base, final=final, alias_token=token)

    cleaned = [_clean_token(token) for token in tokens]
    for index in range(len(cleaned) - 1):
        base = cleaned[index]
        final = cleaned[index + 1]
        if base in _BASES and final in _FINALS:
            return MandarinObservation(
                entry=entry,
                base_unit=base,
                final=final,
                alias_token=f"{base} {final}",
            )

    return None


def collect_observations(
    entries: Iterable[OtoEntry],
    affixes: Iterable[tuple[str, str]] = (),
) -> list[MandarinObservation]:
    observations: list[MandarinObservation] = []
    for entry in entries:
        observation = classify_alias(entry, affixes)
        if observation is not None:
            observations.append(observation)
    return observations


def structure_for(observation: MandarinObservation) -> MandarinSyllableStructure:
    base = observation.base_unit
    final = observation.final

    if base == "y" and final in _Y_ZERO_FINALS:
        return MandarinSyllableStructure(
            onset=None,
            final=_Y_ZERO_FINALS[final],
            orthographic_initial="y",
        )

    if base == "w" and final in _W_ZERO_FINALS:
        return MandarinSyllableStructure(
            onset=None,
            final=_W_ZERO_FINALS[final],
            orthographic_initial="w",
        )

    if base in {"j", "q", "x"} and final in _PALATAL_U_FINALS:
        final = _PALATAL_U_FINALS[final]

    if base in {"n", "l"} and final == "ue":
        final = "ve"

    return MandarinSyllableStructure(onset=base, final=final)


def context_for(base_unit: str, final: str) -> str | None:
    if base_unit in {"sh", "zh", "ch", "s", "z", "c"}:
        if final.startswith("u"):
            return "rounded"
        return "plain"

    if base_unit == "r":
        if final == "i":
            return "front"
        if final.startswith("u"):
            return "rounded"
        return "plain"

    if base_unit in {"j", "q", "x"}:
        if final in {"u", "ue", "uan", "un", "v", "ve", "van", "vn", "ü"}:
            return "rounded"
        if final in {"i", "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing"}:
            return "plain"

    return None
