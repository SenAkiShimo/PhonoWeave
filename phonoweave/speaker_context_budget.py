"""Speaker-context-aware *candidate* recording-sequence budget planner.

Uses explicit profile decisions and traceable sequence/OTO candidate links
under an identical per-pitch line budget. It is a deterministic planning
heuristic; source-profile proxy evidence is not production synthesis evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import tempfile

from .mandarin import _split_syllable, context_for
from .recording_scheme_budget import _eligible_source, _features, _selected_names

SCHEMA = "phonoweave.speaker_context_budget_plan.v0.1"
ELIGIBLE_DECISION = "split_recommended"
ELIGIBLE_SYNTHESIS = "supported_under_proxy"
CONFIDENCE_WEIGHTS = {"high": 2, "moderate": 1}


def _protected_units(unit_rows: list[dict]):
    identified, by_context, omitted = {}, defaultdict(list), []
    for row in unit_rows:
        uid, base = row.get("unit_id"), row.get("base_unit")
        if not isinstance(uid, str) or not uid or uid in identified:
            raise ValueError("missing or duplicate speaker unit ID")
        if not isinstance(base, str) or not base:
            raise ValueError("invalid base consonant")
        identified[uid] = row
        contexts = row.get("contexts", "").split("|")
        if not contexts or any(not c for c in contexts):
            raise ValueError("empty unit contexts")
        if (row.get("decision") == ELIGIBLE_DECISION and
                row.get("synthesis_evidence") == ELIGIBLE_SYNTHESIS and
                row.get("confidence") in CONFIDENCE_WEIGHTS):
            for context in contexts:
                by_context[(base, context)].append(uid)
        else:
            omitted.append(uid)
    # Never select an arbitrary unit when an inventory context is ambiguous.
    conflicts = [key for key, ids in by_context.items() if len(set(ids)) != 1]
    if conflicts:
        raise ValueError("ambiguous protected inventory unit context")
    return identified, {key: ids[0] for key, ids in by_context.items()}, omitted


def _direct_context(base: str, final: str) -> str | None:
    inferred = context_for(base, final)
    if base == "x" and inferred == "plain":
        return "front_unrounded"
    return inferred


def _reference_evidence(eligible: dict, unit_rows: list[dict],
                        link_rows: list[dict], pitches: list[str]):
    units, unit_context, omitted = _protected_units(unit_rows)
    wav_lookup = {}
    for name, item in eligible.items():
        for source in item["sources"]:
            key = (source["pitch_group"], source["source_wav"])
            if key in wav_lookup:
                raise ValueError("duplicate reference WAV provenance")
            wav_lookup[key] = (name, set(source["oto_files"]))

    # Every pitch-covered source line is eligible for *surface* context
    # observations; this is not a claim that OTO contains a valid CV sample.
    features = {name: set() for name in eligible}
    for name in eligible:
        for word in name.split("_"):
            parsed = _split_syllable(word)
            if parsed is None:
                continue
            base, final = parsed
            uid = unit_context.get((base, _direct_context(base, final)))
            if uid is None:
                continue
            for pitch in pitches:
                features[name].add((uid, pitch))

    seen_links = set()
    ignored_outside = 0
    accepted_internal = 0
    for link in link_rows:
        pitch, wav, uid = (link.get("pitch_group"), link.get("source_wav"),
                           link.get("unit_id"))
        key = (pitch, wav)
        if key not in wav_lookup:
            ignored_outside += 1
            continue
        name, oto_files = wav_lookup[key]
        unit = units.get(uid)
        if (unit is None or link.get("base_unit") != unit["base_unit"] or
                link.get("context") not in unit["contexts"].split("|") or
                not link.get("context", "").startswith("internal:") or
                link.get("evidence") != "ordered_reclist_filename_and_oto_boundary" or
                link.get("review_status") != "candidate_link_not_synthesis_validated" or
                link.get("source_oto") not in oto_files):
            raise ValueError("internal link unit or OTO provenance mismatch")
        words = name.split("_")
        try:
            position = int(link["boundary_order"])
        except (TypeError, ValueError, KeyError):
            raise ValueError("invalid internal link boundary order") from None
        if not (0 < position < len(words) and
                words[position] == link.get("syllable") and
                words[position - 1] == link.get("previous_syllable") and
                words[position] == link.get("next_syllable")):
            raise ValueError("internal link sequence/boundary mismatch")
        identity = (link.get("source_oto"), link.get("line_number"),
                    pitch, wav, uid, position)
        if identity in seen_links:
            raise ValueError("duplicate internal link evidence")
        seen_links.add(identity)
        if uid in omitted:
            continue
        # Protect role-scoped units only if the exact *candidate* link
        # is traceable; never derive internal contexts from spelling alone.
        features[name].add((uid, pitch))
        accepted_internal += 1
    return features, units, omitted, ignored_outside, accepted_internal


def _counts(names: list[str], features: dict):
    counter = Counter()
    for name in names:
        counter.update(features[name])
    return counter


def _coverage(names: list[str], eligible: dict, universe: dict):
    have = {key: set().union(*(eligible[n]["features"][key] for n in names))
            if names else set() for key in universe}
    return {
        key: {"covered": len(have[key]), "total": len(universe[key]),
              "fraction": len(have[key]) / len(universe[key])
              if universe[key] else 1.0}
        for key in universe
    }


def compile_speaker_context_budget(
    blueprint: dict, unit_rows: list[dict], link_rows: list[dict],
    budget: int, *, repeat_target: int = 3,
) -> dict:
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if type(repeat_target) is not int or not 1 <= repeat_target <= 10:
        raise ValueError("repeat_target must be 1 through 10")
    pitches, eligible, excluded = _eligible_source(blueprint)
    features, units, omitted, ignored, accepted = _reference_evidence(
        eligible, unit_rows, link_rows, pitches
    )
    universe = {key: set().union(*(row["features"][key] for row in eligible.values()))
                if eligible else set()
                for key in ("syllables", "starts", "exact_adjacent_pairs")}
    complete_counts = _counts(list(eligible), features)
    # Only score source-observable cells: unsupported source contexts remain
    # explicit gaps, rather than impossible optimizer objectives.
    goals = {key: min(repeat_target, count)
             for key, count in complete_counts.items()}
    weights = {key: CONFIDENCE_WEIGHTS[units[key[0]]["confidence"]]
               for key in goals}

    # Hard first-stage policy: cover all observable syllable spellings
    # before maximizing source-context depth. If budget is insufficient,
    # incomplete coverage is reported, never hidden.
    selected = _selected_names(eligible, "syllables", budget)
    first_stage_lines = len(selected)
    observed = _counts(selected, features)
    have_starts = set().union(*(eligible[n]["features"]["starts"] for n in selected)) if selected else set()
    have_pairs = set().union(*(eligible[n]["features"]["exact_adjacent_pairs"] for n in selected)) if selected else set()
    while len(selected) < min(budget, len(eligible)):
        remaining = sorted(set(eligible) - set(selected))
        ranked = []
        for name in remaining:
            gained = sum(weights[cell] for cell in features[name]
                         if observed[cell] < goals[cell])
            starts = len(eligible[name]["features"]["starts"] - have_starts)
            pairs = len(eligible[name]["features"]["exact_adjacent_pairs"] - have_pairs)
            ranked.append((-gained, -starts, -pairs, name))
        if not ranked:
            break
        best = min(ranked)
        if best[:3] == (0, 0, 0):
            break
        name = best[3]
        selected.append(name)
        observed.update(features[name])
        have_starts.update(eligible[name]["features"]["starts"])
        have_pairs.update(eligible[name]["features"]["exact_adjacent_pairs"])

    baseline = _selected_names(eligible, "syllables_and_starts", budget)
    report_rows = []
    for uid, pitch in sorted(goals):
        target = goals[(uid, pitch)]
        report_rows.append({
            "unit_id": uid, "base_unit": units[uid]["base_unit"],
            "context": units[uid]["contexts"],
            "confidence": units[uid]["confidence"],
            "pitch_group": pitch,
            "source_sequence_candidates": complete_counts[(uid, pitch)],
            "desired_distinct_sequence_references": repeat_target,
            "feasible_target_in_reference": target,
            "baseline_selected_sequence_references": _counts(baseline, features)[(uid, pitch)],
            "speaker_selected_sequence_references": observed[(uid, pitch)],
            "speaker_meets_feasible_target": observed[(uid, pitch)] >= target,
            "source_below_requested_target": complete_counts[(uid, pitch)] < repeat_target,
        })
    baseline_counts = _counts(baseline, features)
    weighted = lambda counts: sum(
        weights[cell] * min(goals[cell], counts[cell]) for cell in goals
    )
    full_syllables = universe["syllables"]
    selected_syllables = set().union(*(eligible[n]["features"]["syllables"] for n in selected)) if selected else set()
    return {
        "schema": SCHEMA,
        "pitch_groups": pitches,
        "per_pitch_line_budget_max": budget,
        "repeat_target_per_unit_pitch": repeat_target,
        "reference_lines": len(eligible),
        "source_excluded_sequences": excluded,
        "source_excluded_wav_groups": blueprint.get("excluded_wav_groups", {}),
        "protected_units": sorted({uid for uid, _ in goals}),
        "protected_observable_unit_pitch_cells": len(goals),
        "unresolved_or_not_proxy_supported_units_not_optimized": sorted(omitted),
        "ignored_links_outside_reference": ignored,
        "accepted_internal_candidate_links": accepted,
        "selection": {
            "policy": "syllable_first_then_evidence_context_depth_greedy",
            "syllable_first_stage_lines": first_stage_lines,
            "selected_lines": selected,
            "selected_line_count": len(selected),
            "full_reference_syllables_covered": len(full_syllables - (full_syllables - selected_syllables)),
            "full_reference_syllables_total": len(full_syllables),
            "missing_syllables": sorted(full_syllables - selected_syllables),
            "surface_coverage": _coverage(selected, eligible, universe),
            "weighted_context_depth_score": weighted(observed),
        },
        "same_budget_baseline": {
            "policy": "syllables_and_starts",
            "selected_lines": baseline,
            "selected_line_count": len(baseline),
            "surface_coverage": _coverage(baseline, eligible, universe),
            "weighted_context_depth_score": weighted(baseline_counts),
        },
        "per_unit_pitch_reference_depth": report_rows,
        "production_ready": False,
        "recordings_authorized": 0,
        "limitations": [
            "Protected split recommendations are based on previously reported proxy evidence, not confirmed synthesis gains.",
            "Direct contexts are inferred from source syllable spellings, not verified OTO boundaries.",
            "Internal contexts require sequence/OTO candidate links but those links are not acoustic validation.",
            "Distinct sequence WAV references are not controlled independent repeat takes.",
            "The 522 excluded or other unparsed sources are not accounted for in coverage optimization.",
            "No production-safe merging, deletion, alias or OTO generation is performed.",
        ],
    }


def write_speaker_context_budget(result: dict, output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phonoweave-context-budget-",
                                     dir=output.parent) as stage:
        path = Path(stage)
        (path / "speaker_context_budget_plan.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        (path / "speaker_context_pilot_reclist.txt").write_text(
            "".join(x + "\n" for x in result["selection"]["selected_lines"]),
            encoding="utf-8")
        (path / "README.txt").write_text(
            "CANDIDATE CONTEXT-DEPTH RECORDING PLAN; NOT PRODUCTION READY.\n"
            "Selection uses provisional speaker profile split evidence and\n"
            "recording filename/OTO context links. No WAV, alias or OTO edited.\n"
            "An observed source sequence is not an independent repeat take.\n",
            encoding="utf-8")
        if output.exists():
            raise FileExistsError("output appeared during compilation")
        os.rename(path, output)
    return output
