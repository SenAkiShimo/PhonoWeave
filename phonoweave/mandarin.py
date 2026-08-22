from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .oto import OtoEntry


_FINALS = (
    "uang", "iong", "iang", "uai", "uan", "iao", "ian", "ang", "eng", "ing", "ong",
    "ua", "uo", "ui", "un", "ia", "ie", "iu", "in", "ai", "ei", "ao", "ou", "an", "en",
    "a", "o", "e", "i", "u", "v", "ü",
)

_BASES = ("zh", "ch", "sh", "r", "s")


@dataclass(frozen=True)
class MandarinObservation:
    entry: OtoEntry
    base_unit: str
    final: str
    alias_token: str


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


def context_for(base_unit: str, final: str) -> str | None:
    if base_unit in {"sh", "zh", "ch", "s"}:
        if final.startswith("u"):
            return "rounded"
        return "plain"

    if base_unit == "r":
        if final == "i":
            return "front"
        if final.startswith("u"):
            return "rounded"
        return "plain"

    return None
