"""Surface-role recognizer for existing Mandarin UTAU aliases.

It describes source alias grammar without generating new aliases, assigning OTO
timings, inferring consonant realizations, or asserting full OpenUtau support.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .mandarin import _BASES, _FINALS

# Known corpus vowel-only symbols include nonstandard vowel variants.
# A variant label is a surface token, not a phonological equivalence claim.
_VOWELS = frozenset(_FINALS) | {"er", "e0", "en0", "i0", "ir"}
_INITIALS = frozenset(_BASES)
_PITCH_DIRECTORY = re.compile(r"^([A-Ga-g][#b]?-?\d+)(?: EX)?$")
_CV_TOKEN = re.compile(r"^[a-zü]+$")


@dataclass(frozen=True)
class AliasRole:
    surface: str
    pitch_suffix: str
    pitch_suffix_status: str
    role: str
    left_token: str
    right_token: str
    review_status: str


def _syllable(token: str) -> bool:
    if not _CV_TOKEN.fullmatch(token):
        return False
    return any(token.startswith(base) and token[len(base):] in _FINALS
               for base in _BASES)


def _role(surface: str) -> tuple[str, str, str]:
    tokens = surface.split()
    if not tokens:
        return "empty_or_pitch_only", "", ""
    if len(tokens) == 1:
        word = tokens[0]
        if _syllable(word):
            return "standalone_cv", "", word
        if word in _VOWELS:
            return "standalone_vowel", "", word
        if word.startswith("_") and word[1:] in _VOWELS:
            return "prefixed_vowel", "", word[1:]
        return "special_or_unknown", "", word
    if len(tokens) != 2:
        return "special_or_unknown", tokens[0], tokens[-1]
    left, right = tokens
    if left == "-":
        if _syllable(right):
            return "initial_cv", left, right
        if right in _VOWELS:
            return "initial_vowel", left, right
        return "special_or_unknown", left, right
    if left in _VOWELS:
        if right in _INITIALS:
            return "vc_transition", left, right
        if right in _VOWELS:
            return "vv_transition", left, right
        if _CV_TOKEN.fullmatch(right):
            return "nonstandard_transition", left, right
        return "modified_or_special_transition", left, right
    return "special_or_unknown", left, right


def classify_alias_role(alias: str, oto_path: Path) -> AliasRole:
    """Parse only the known surface form, never infer missing CV context.

    Pitch suffix is recognized only if the OTO folder names a note. Mismatched
    suffixes are not removed and must not be silently remapped.
    """
    folder = oto_path.parent.name
    match = _PITCH_DIRECTORY.fullmatch(folder)
    pitch = match.group(1) if match else ""
    name = alias.strip()
    if pitch and name.endswith(pitch):
        surface = name[:-len(pitch)].strip()
        suffix_status = "matched"
    elif pitch:
        surface = name
        suffix_status = "not_matched"
    else:
        surface = name
        suffix_status = "not_applicable"
    role, left, right = _role(surface)
    review_status = (
        "surface_role_only" if role not in
        {"empty_or_pitch_only", "special_or_unknown",
         "modified_or_special_transition", "nonstandard_transition"}
        else "manual_review"
    )
    return AliasRole(
        surface=surface, pitch_suffix=pitch,
        pitch_suffix_status=suffix_status,
        role=role, left_token=left, right_token=right,
        review_status=review_status,
    )
