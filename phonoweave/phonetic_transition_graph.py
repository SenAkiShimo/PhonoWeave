"""Language-neutral, explicit-phone contextual transition inventory.

Separate user-authored reclist aliases from a verified/manual pronunciation
lexicon. No G2P guess, no inference from Latin characters or raw IPA strings,
and no acoustic validation or production alias export.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json

from .reclist import ReclistDocument
from .reclist_budget import plan_literal_budget

LEXICON_SCHEMA = "phonoweave.pronunciation_mapping.v0.1"
SCHEMA = "phonoweave.phonetic_transition_budget.v0.2"
ROLES = ("C", "V")
PHASES = ("within_syllable", "across_syllables")
WEIGHTS_DEFAULT = {"phone": 1, "within_syllable": 2, "across_syllables": 3}


def _hash_json(value: dict) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _check_lexicon(lexicon: dict) -> dict[str, tuple[tuple[str, str, str], ...]]:
    """Require manual mapping; role V also represents non-vowel syllabic nuclei."""
    if not isinstance(lexicon, dict) or lexicon.get("schema") != LEXICON_SCHEMA:
        raise ValueError("unsupported pronunciation lexicon schema")
    if (not isinstance(lexicon.get("language"), str)
            or not lexicon["language"].strip()):
        raise ValueError("lexicon must declare a language identifier")
    entries = lexicon.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("lexicon entries must be nonempty")
    checked = {}
    for token, phones in entries.items():
        if not isinstance(token, str) or not token or token.strip() != token:
            raise ValueError("invalid lexical alias key")
        if (not isinstance(phones, list) or not phones
                or len(phones) > 64):
            raise ValueError(f"invalid or empty pronunciation for {token!r}")
        segments = []
        for segment in phones:
            if not isinstance(segment, dict):
                raise ValueError("pronunciation segments must be objects")
            symbol, role, grapheme = (
                segment.get("symbol"), segment.get("role"), segment.get("grapheme"))
            if (not isinstance(symbol, str) or not symbol.strip()
                    or not isinstance(grapheme, str) or not grapheme.strip()
                    or role not in ROLES):
                raise ValueError(f"invalid pronunciation segment in {token!r}")
            if symbol != symbol.strip() or grapheme != grapheme.strip():
                raise ValueError(f"extraneous whitespace in {token!r}")
            segments.append((symbol, role, grapheme))
        if not any(role == "V" for _, role, _ in segments):
            raise ValueError(f"pronunciation has no syllabic nucleus: {token!r}")
        checked[token] = tuple(segments)
    return checked


def _encode(feature: tuple) -> dict:
    if feature[0] == "phone":
        return {"kind": "phone", "symbol": feature[1], "role": feature[2]}
    return {"kind": "transition", "phase": feature[1],
            "pattern": feature[2], "left": feature[3],
            "right": feature[4]}


def _line_features(tokens: tuple[str, ...],
                   entries: dict[str, tuple]) -> set[tuple]:
    result = set()
    previous = None
    for token in tokens:
        segments = entries[token]
        for symbol, role, _grapheme in segments:
            result.add(("phone", symbol, role))
        for left, right in zip(segments, segments[1:]):
            result.add(("edge", "within_syllable", left[1] + right[1],
                        left[0], right[0]))
        if previous is not None:
            result.add(("edge", "across_syllables",
                        previous[1] + segments[0][1],
                        previous[0], segments[0][0]))
        previous = segments[-1]
    return result


def _inventory_ambiguities(entries: dict[str, tuple]) -> list[dict]:
    spellings = defaultdict(lambda: defaultdict(set))
    for token, segments in entries.items():
        for phone, role, spelling in segments:
            spellings[spelling][(phone, role)].add(token)
    result = []
    for spelling, variants in sorted(spellings.items()):
        if len(variants) > 1:
            result.append({
                "grapheme": spelling,
                "realizations": [
                    {"symbol": phone, "role": role, "lexical_aliases": sorted(tokens)}
                    for (phone, role), tokens in sorted(variants.items())
                ],
            })
    return result


def _metric(features: set[tuple], universe: set[tuple], weights: dict) -> dict:
    transitions = sorted({(f[1], f[2]) for f in universe if f[0] == "edge"})
    return {
        "distinct_phone_types": sum(f[0] == "phone" for f in features),
        "distinct_phone_types_total": sum(f[0] == "phone" for f in universe),
        "transition_types": {
            f"{phase}.{pattern}": {
                "covered": sum(f[0] == "edge" and f[1] == phase
                               and f[2] == pattern for f in features),
                "total": sum(f[0] == "edge" and f[1] == phase
                             and f[2] == pattern for f in universe),
            }
            for phase, pattern in transitions
        },
        "weighted_covered": sum(
            weights["phone" if f[0] == "phone" else f[1]] for f in features),
        "weighted_total": sum(
            weights["phone" if f[0] == "phone" else f[1]] for f in universe),
    }


def compile_transition_budget(document: ReclistDocument, lexicon: dict,
                              budget: int, *, weights: dict | None = None) -> dict:
    if type(budget) is not int or budget < 0:
        raise ValueError("line budget must be a nonnegative integer")
    entries = _check_lexicon(lexicon)
    chosen_weights = WEIGHTS_DEFAULT if weights is None else weights
    if (not isinstance(chosen_weights, dict) or set(chosen_weights) != set(WEIGHTS_DEFAULT)
            or any(type(x) is not int or x <= 0 for x in chosen_weights.values())):
        raise ValueError("weights require positive integers for phone, within_syllable, across_syllables")

    sources = document.recording_lines
    missing = defaultdict(list)
    for line in sources:
        for token in line.tokens:
            if token not in entries:
                missing[token].append(line.number)
    if missing:
        raise ValueError("missing explicit pronunciations: " + json.dumps(
            {k: sorted(set(v)) for k, v in sorted(missing.items())},
            ensure_ascii=False, sort_keys=True))

    candidates = [(line, _line_features(line.tokens, entries)) for line in sources]
    universe = set().union(*(features for _, features in candidates))
    covered = set()
    selected = []
    remaining = list(candidates)

    def cost(f: tuple) -> int:
        return chosen_weights["phone" if f[0] == "phone" else f[1]]

    while remaining and len(selected) < budget:
        line, features = max(
            remaining,
            key=lambda item: (
                sum(cost(f) for f in item[1] - covered),
                -item[0].number,
            ),
        )
        gain = features - covered
        if not gain:
            break
        selected.append({
            "source_line": line.number, "raw": line.raw,
            "aliases": list(line.tokens),
            "marginal_weight": sum(cost(f) for f in gain),
            "new_features": [_encode(f) for f in sorted(gain)],
        })
        covered.update(features)
        remaining.remove((line, features))

    # Separate baseline: original orthographic labels and adjacent aliases.
    literal = plan_literal_budget(document, budget)
    byline = {line.number: features for line, features in candidates}
    common = min(len(selected), literal["selected_count"])
    curve = []
    a_have, b_have = set(), set()
    for k in range(common + 1):
        if k:
            a_have.update(byline[selected[k - 1]["source_line"]])
            b_have.update(byline[literal["selected_lines"][k - 1]["source_line"]])
        curve.append({
            "exact_matched_line_count": k,
            "phonetic_policy": _metric(a_have, universe, chosen_weights),
            "literal_policy": _metric(b_have, universe, chosen_weights),
        })
    return {
        "schema": SCHEMA,
        "status": "symbolic_pronunciation_coverage_only_not_acoustically_validated",
        "language": lexicon["language"],
        "source": {
            "decoded_reclist_sha256": sha256(document.text.encode("utf-8")).hexdigest(),
            "lexicon_sha256": _hash_json(lexicon),
            "separator": document.separator,
            "source_recording_lines": len(sources),
        },
        "settings": {
            "line_budget": budget,
            "weights": dict(chosen_weights),
            "cost_unit": "recording_line_not_measured_duration",
        },
        "spelling_realization_ambiguities": _inventory_ambiguities(entries),
        "phonetic_policy": {
            "selected_lines": selected, "selected_count": len(selected),
            "uncovered_features": [_encode(f) for f in sorted(universe - covered)],
            "coverage": _metric(covered, universe, chosen_weights),
        },
        "literal_policy": {
            "selected_lines": literal["selected_lines"],
            "selected_count": literal["selected_count"],
        },
        "matched_prefix_curve": curve,
        "limits": [
            "All phones, syllabic nuclei and grapheme correspondences are explicitly supplied; no automatic G2P, phonetic discovery or allophone classification.",
            "Phase indicates within-syllable versus cross-syllable adjacent phone boundary, not a validated audio cut position.",
            "VC/CV/CC/VV are surface-role transitions only, not an acoustic claim that they must be independent recording units.",
            "One source recording line is not equivalent to equal capture time or equally many source WAVs.",
            "No phonemic similarity-based merge, synthesis quality gain, perceptual benefit, production alias or OTO timing is inferred.",
        ],
        "production_ready": False,
        "recordings_authorized": 0,
        "production_alias_export": "disabled",
        "oto_timing_changes": "none",
    }
