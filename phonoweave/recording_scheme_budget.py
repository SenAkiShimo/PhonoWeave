"""Budget-capped, coverage-audited recording sequence proposals.

This is a deterministic surface-form planning baseline. It *does not* assert
phoneme equivalence, acoustic quality, or a production-safe OpenUtau reclist.
In particular, adjacent orthographic pairs are not verified OTO transitions.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import os

SOURCE_SCHEMA = "phonoweave.reference_reclist_blueprint.v0.1"
SCHEMA = "phonoweave.recording_budget_plan.v0.1"
MODES = ("syllables", "syllables_and_starts", "exact_adjacent_pairs")
_FEATURES = ("syllables", "starts", "exact_adjacent_pairs")


def _features(sequence: str) -> dict[str, set[str]]:
    tokens = sequence.split("_")
    if not tokens or any(not token or not token.isalpha() for token in tokens):
        raise ValueError("invalid syllable sequence")
    return {
        "syllables": set(tokens),
        "starts": {tokens[0]},
        "exact_adjacent_pairs": {
            left + " -> " + right for left, right in zip(tokens, tokens[1:])
        },
    }


def _eligible_source(blueprint: dict):
    if blueprint.get("schema") != SOURCE_SCHEMA:
        raise ValueError("unsupported source blueprint schema")
    pitches = blueprint.get("pitch_groups")
    if (not isinstance(pitches, list) or not pitches
            or len(set(pitches)) != len(pitches)):
        raise ValueError("invalid pitch group list")
    sequences = blueprint.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("missing reference sequences")
    eligible = {}
    excluded = []
    for row in sequences:
        name = row.get("sequence")
        if not isinstance(name, str) or name in eligible:
            raise ValueError("invalid or duplicated source sequence")
        features = _features(name)
        observed = row.get("pitch_groups_observed")
        sources = row.get("source_wavs")
        if not isinstance(observed, list) or not isinstance(sources, list):
            raise ValueError("malformed pitch/source provenance")
        if (row.get("status") == "all_pitch_groups_observed"
                and set(observed) == set(pitches)
                and not row.get("pitch_groups_not_observed")
                and len(sources) == len(pitches)
                and Counter(s.get("pitch_group") for s in sources)
                    == Counter(pitches)
                and all(s.get("source_wav") and s.get("oto_files") for s in sources)):
            eligible[name] = {"features": features, "sources": sources}
        else:
            excluded.append({"sequence": name, "reason": "unverified_pitch_coverage"})
    if (blueprint.get("interpretable_sequences") != len(sequences)
            or blueprint.get("all_pitch_reference_sequences") != len(eligible)):
        raise ValueError("blueprint counts inconsistent with eligible sequences")
    if (set(blueprint.get("reference_reclist", [])) != set(eligible)
            or len(blueprint.get("reference_reclist", [])) != len(eligible)):
        raise ValueError("reference reclist differs from pitch-covered sources")
    return pitches, eligible, excluded


def _selected_names(eligible: dict, mode: str, budget: int) -> list[str]:
    if mode not in MODES:
        raise ValueError("unsupported coverage mode")
    priority = {
        "syllables": ("syllables",),
        "syllables_and_starts": ("syllables", "starts"),
        "exact_adjacent_pairs": _FEATURES,
    }[mode]
    masks = {
        name: frozenset((kind, value)
                        for kind in priority for value in item["features"][kind])
        for name, item in eligible.items()
    }
    uncovered = set().union(*masks.values()) if masks else set()
    selected = []
    for _ in range(min(budget, len(masks))):
        candidates = sorted(set(masks) - set(selected))
        if not candidates:
            break
        # deterministic tie breaking: lexicographically smallest sequence
        best = min(candidates, key=lambda name: (-len(masks[name] & uncovered), name))
        gained = masks[best] & uncovered
        if not gained:
            break
        selected.append(best)
        uncovered.difference_update(gained)
    return selected


def compile_recording_budget(blueprint: dict, budget: int,
                             selected_mode: str = "syllables_and_starts") -> dict:
    """Compare transparent coverage objectives under the same per-pitch cap."""
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if selected_mode not in MODES:
        raise ValueError("unknown selection mode")
    pitches, eligible, excluded = _eligible_source(blueprint)
    universe = {
        key: set().union(*(item["features"][key] for item in eligible.values()))
        if eligible else set()
        for key in _FEATURES
    }
    strategies = {}
    for mode in MODES:
        selected = _selected_names(eligible, mode, budget)
        union = {
            key: set().union(*(eligible[name]["features"][key] for name in selected))
            if selected else set()
            for key in _FEATURES
        }
        strategies[mode] = {
            "selected_lines": selected,
            "selected_line_count": len(selected),
            "referenced_wav_count_at_all_pitches": len(selected) * len(pitches),
            "coverage": {
                key: {
                    "covered": len(union[key]),
                    "total": len(universe[key]),
                    "fraction": (len(union[key]) / len(universe[key])
                                 if universe[key] else 1.0),
                    "uncovered": sorted(universe[key] - union[key]),
                }
                for key in _FEATURES
            },
        }
    return {
        "schema": SCHEMA,
        "source_schema": SOURCE_SCHEMA,
        "pitch_groups": pitches,
        "per_pitch_line_budget_max": budget,
        "selected_mode": selected_mode,
        "eligible_reference_lines": len(eligible),
        "excluded_reference_lines": excluded,
        "excluded_source_wav_groups": blueprint.get("excluded_wav_groups", {}),
        "feature_universe_sizes": {k: len(v) for k, v in universe.items()},
        "strategies": strategies,
        "selected_reclist": strategies[selected_mode]["selected_lines"],
        "selection_is_production_ready": False,
        "new_recordings_authorized": 0,
        "known_limitations": [
            "This is a surface-syllable and ordered-spelling coverage proxy, not acoustic coverage.",
            "Exact adjacent spellings are not equivalent to a validated OTO transition.",
            "Original excluded WAV groups and non-sequence aliases are out of optimization scope.",
            "No split/merge acoustic decisions are automatically adopted; no source file is edited.",
        ],
    }


def write_recording_budget(report: dict, output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".phonoweave-budget-") as stage:
        root = Path(stage)
        (root / "recording_budget_plan.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (root / "pilot_reclist.txt").write_text(
            "".join(line + "\n" for line in report["selected_reclist"]),
            encoding="utf-8",
        )
        (root / "README.txt").write_text(
            "COVERAGE PROXY ONLY — PILOT, NOT PRODUCTION RECLIST.\n"
            "Budget counts recording sequence lines PER pitch subbank.\n"
            "pilot_reclist.txt omits some observed transitions and special sources.\n"
            "Do not replace or delete the original source bank based on this plan.\n"
            "No synthesis quality or speaker-adaptive split benefit is established.\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError("output directory appeared during compilation")
        os.rename(root, output)
    return output
