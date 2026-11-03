"""Deterministic, language-neutral *surface-token* recording budget planner.

No phonemic analysis, phonotactics, alias mapping, or synthesis decision is made.
Every reported unit is a literal token or adjacent token pair from the reclist.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .reclist import ReclistDocument, ReclistLine, load_reclist

SCHEMA = "phonoweave.literal_reclist_budget.v0.1"


def _features(line: ReclistLine, include_pairs: bool) -> set[tuple[str, ...]]:
    items = {("token", value) for value in line.tokens}
    if include_pairs:
        items.update(("pair", a, b) for a, b in zip(line.tokens, line.tokens[1:]))
    return items


def plan_literal_budget(
    document: ReclistDocument,
    budget: int,
    *,
    include_pairs: bool = True,
    token_weight: int = 1,
    pair_weight: int = 2,
) -> dict[str, Any]:
    """Greedily maximize distinct literal features under a line budget.

    Duplicate source lines are kept distinct by 1-based source line number.
    Tie breaks: highest weighted marginal coverage, then lowest source line
    number. Zero-marginal lines are NOT selected; budget is a cap, not quota.
    This is a set-cover heuristic, not a globally optimal or acoustic planner.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    for name, value in (("token_weight", token_weight), ("pair_weight", pair_weight)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(include_pairs, bool):
        raise ValueError("include_pairs must be boolean")
    candidates = [(line, _features(line, include_pairs))
                  for line in document.recording_lines]
    universe: set[tuple[str, ...]] = set().union(*(f for _, f in candidates))
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    covered: set[tuple[str, ...]] = set()

    def weight(feature: tuple[str, ...]) -> int:
        return token_weight if feature[0] == "token" else pair_weight

    while remaining and len(selected) < budget:
        winner = max(
            remaining,
            key=lambda item: (
                sum(weight(x) for x in item[1] - covered),
                -item[0].number,
            ),
        )
        line, features = winner
        gain = features - covered
        score = sum(weight(x) for x in gain)
        if score == 0:
            break
        selected.append({
            "source_line": line.number,
            "raw": line.raw,
            "tokens": list(line.tokens),
            "new_tokens": sorted(x[1] for x in gain if x[0] == "token"),
            "new_pairs": sorted([x[1], x[2]] for x in gain if x[0] == "pair"),
            "marginal_weight": score,
        })
        covered.update(features)
        remaining.remove(winner)

    def encode(features: set[tuple[str, ...]]) -> dict[str, list]:
        return {
            "tokens": sorted(x[1] for x in features if x[0] == "token"),
            "pairs": sorted([x[1], x[2]] for x in features if x[0] == "pair"),
        }

    missing = universe - covered
    source_token_occurrences = Counter(t for line in document.recording_lines
                                       for t in line.tokens)
    return {
        "schema": SCHEMA,
        "status": "literal_coverage_only_not_phonetic_or_synthesis_validated",
        "source": {
            "encoding": document.encoding,
            "separator": document.separator,
            "recording_lines": len(candidates),
            "token_occurrences": sum(source_token_occurrences.values()),
            "unique_literal_tokens": len(source_token_occurrences),
        },
        "settings": {
            "line_budget": budget,
            "include_adjacent_pairs": include_pairs,
            "token_weight": token_weight,
            "pair_weight": pair_weight,
        },
        "selected_lines": selected,
        "selected_count": len(selected),
        "unspent_line_budget": budget - len(selected),
        "covered": encode(covered),
        "uncovered": encode(missing),
        "coverage": {
            "token_types": sum(x[0] == "token" for x in covered),
            "token_types_total": sum(x[0] == "token" for x in universe),
            "adjacent_pair_types": sum(x[0] == "pair" for x in covered),
            "adjacent_pair_types_total": sum(x[0] == "pair" for x in universe),
            "weighted_covered": sum(weight(x) for x in covered),
            "weighted_total": sum(weight(x) for x in universe),
        },
        "production_alias_export": "disabled",
        "oto_timing_changes": "none",
    }


def plan_reclist_file(
    source: Path,
    budget: int,
    *,
    separator: str = "whitespace",
    encoding: str | None = None,
    include_pairs: bool = True,
    token_weight: int = 1,
    pair_weight: int = 2,
) -> dict[str, Any]:
    document = load_reclist(source, separator=separator, encoding=encoding)
    return plan_literal_budget(document, budget, include_pairs=include_pairs,
                               token_weight=token_weight, pair_weight=pair_weight)
