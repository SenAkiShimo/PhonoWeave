"""Equal-weight, independent-voicebank replication of frozen budget policies.

This is an orchestration and audit layer only. The within-voicebank model,
disjoint five-fold partitions, candidate feature extraction and random
acquisition policy are exactly those used in v0.2/v0.3. No WAV, OTO,
speaker-profile or synthesis-inventory file is modified.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

from .recording_budget_benchmark import load_recording_units
from .recording_budget_disjoint_cv import run_disjoint_benchmark
from .recording_budget_random_sensitivity import evaluate_random_sensitivity

VERSION = "0.5-cross-voicebank-replication"
REQUIRED_BUDGETS = (0.25, 0.50, 0.75, 1.00)


def validate_manifest(data: dict) -> tuple[dict, ...]:
    """Require independently identified speaker groups; never infer identity."""
    if not isinstance(data, dict) or data.get("schema") != "phonoweave.voicebank_replication_manifest.v1":
        raise ValueError("unsupported voicebank replication manifest schema")
    entries = data.get("voicebanks")
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("at least two voicebanks are required")
    ids, groups, paths = set(), set(), set()
    validated = []
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError("invalid voicebank manifest entry")
        bank_id = raw.get("bank_id")
        group = raw.get("speaker_group_id")
        path = raw.get("path")
        if any(not isinstance(value, str) or not value.strip()
               for value in (bank_id, group, path)):
            raise ValueError("bank_id, speaker_group_id and path are required strings")
        if any("/" in value or "\\" in value for value in (bank_id, group)):
            raise ValueError("bank and speaker identifiers must be single tokens")
        if bank_id in ids or group in groups:
            raise ValueError("bank and speaker-group IDs must be unique: use one bank per speaker")
        root = Path(path).expanduser().resolve()
        if root in paths:
            raise ValueError("duplicate voicebank directory")
        ids.add(bank_id)
        groups.add(group)
        paths.add(root)
        validated.append({"bank_id": bank_id, "speaker_group_id": group, "root": root})
    return tuple(validated)


def _equal_or_raise(lhs: float, rhs: float, *, label: str) -> None:
    if not math.isclose(float(lhs), float(rhs), rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"v0.2/v0.3 replay mismatch: {label}")


def _compare_protocols(v02: dict, v03: dict) -> None:
    """Refuse a cross-bank comparison if the frozen seed-zero replay drifts."""
    if v02.get("version") != "0.2-disjoint-wav-cv" or v03.get("version") != "0.3-random-baseline-sensitivity":
        raise ValueError("unexpected research protocol version")
    splits = {row["fold"]: row for row in v02["splits"]}
    rows = {(row["fold"], row["budget_fraction"], row["policy"]): row for row in v02["rows"]}
    if len(splits) != len(v03["fold_results"]):
        raise ValueError("inconsistent fold counts")
    for fold in v03["fold_results"]:
        k = fold["fold"]
        if k not in splits or splits[k]["heldout_sha256"] != fold["heldout_sha256"]:
            raise ValueError("changed test partition between v0.2 and v0.3")
        for item in fold["budgets"]:
            b = item["budget_fraction"]
            for policy, key in (
                ("fixed_balanced", "fixed_distance"),
                ("adaptive_observed", "adaptive_distance"),
                ("stratified_random", "random_seed0_distance"),
            ):
                if (k, b, policy) not in rows:
                    raise ValueError("missing original experiment row")
                reference = rows[(k, b, policy)]
                if item["budget_wav"] != reference["budget_wav"]:
                    raise ValueError("changed recording budget")
                _equal_or_raise(
                    item[key], reference["penalized_acoustic_distance"],
                    label=f"fold={k}, budget={b}, policy={policy}"
                )


def evaluate_replication(manifest: dict, *, repeats: int = 100) -> dict:
    """Independent processing per speaker; never concatenate WAVs across banks.

    Expected data problems are reported per bank rather than concealed.
    A cross-bank conclusion requires at least two successful banks.
    """
    banks = validate_manifest(manifest)
    if type(repeats) is not int or repeats < 2:
        raise ValueError("repeats must be an integer >= 2")
    reports = []
    seen_recording_hashes = {}
    for bank in banks:
        bank_id = bank["bank_id"]
        record = {"bank_id": bank_id, "speaker_group_id": bank["speaker_group_id"]}
        try:
            if not bank["root"].is_dir():
                raise ValueError("voicebank directory does not exist")
            units, extraction = load_recording_units(bank["root"])
            record["eligible_unique_wavs"] = len(units)
            record["extraction"] = {
                "oto_entries": extraction["oto_entries"],
                "oto_warnings": extraction["oto_warnings"],
                "eligible_wav_paths": extraction["eligible_wav_paths"],
                "skipped_segments": extraction["skipped_segments"],
                "ambiguous_segments": extraction["ambiguous_segments"],
            }
            if len(units) < 20:
                raise ValueError("insufficient eligible WAVs for replication (minimum 20)")
            duplicated = sorted({
                seen_recording_hashes[unit.digest]
                for unit in units if unit.digest in seen_recording_hashes
            })
            if duplicated:
                raise ValueError(
                    "identical audio WAV bytes also exist in an earlier bank: "
                    + ", ".join(duplicated)
                )
            v02 = run_disjoint_benchmark(units)
            v03 = evaluate_random_sensitivity(units, n_random=repeats)
            _compare_protocols(v02, v03)
            record["fold_test_sizes"] = v02["partition"]["fold_test_sizes"]
            record["test_reuse_count"] = v02["partition"]["test_reuse_count"]
            record["folds_with_missing_training_cells"] = v02["partition"]["folds_with_missing_training_cells"]
            record["scores"] = [
                {
                    "budget_fraction": item["budget_fraction"],
                    "fixed_distance": round(item["fixed_macro_mean"], 6),
                    "adaptive_distance": round(item["adaptive_macro_mean"], 6),
                    "random_mean": round(item["random_macro_mean"], 6),
                    "random_p05": round(item["random_macro_p05"], 6),
                    "random_p95": round(item["random_macro_p95"], 6),
                    "random_draws_better_than_adaptive": item["random_repeats_better_than_adaptive"],
                    "random_draws_equal_to_adaptive": item["random_repeats_equal_to_adaptive"],
                    "random_draws_worse_than_adaptive": item["random_repeats_worse_than_adaptive"],
                    "adaptive_minus_random_mean": round(
                        item["adaptive_macro_mean"] - item["random_macro_mean"], 6
                    ),
                    "adaptive_minus_fixed": round(
                        item["adaptive_macro_mean"] - item["fixed_macro_mean"], 6
                    ),
                    "per_fold": [
                        {
                            "fold": row["fold"],
                            "adaptive_distance": round(row["adaptive"], 6),
                            "random_mean": round(row["random_mean"], 6),
                        } for row in item["per_fold"]
                    ],
                } for item in v03["summary"]
            ]
            record["status"] = "evaluated"
            for unit in units:
                seen_recording_hashes[unit.digest] = bank_id
        except (ValueError, OSError, UnicodeError, KeyError, IndexError) as error:
            record["status"] = "excluded"
            record["exclusion_reason"] = f"{type(error).__name__}: {error}"
        reports.append(record)

    successful = [r for r in reports if r["status"] == "evaluated"]
    summary = []
    for budget in REQUIRED_BUDGETS:
        vals = [next(row for row in bank["scores"] if row["budget_fraction"] == budget)
                for bank in successful]
        summary.append({
            "budget_fraction": budget,
            "n_independent_speaker_groups": len(vals),
            "banks_adaptive_better_than_random_mean": sum(
                row["adaptive_minus_random_mean"] < -1e-8 for row in vals
            ),
            "banks_adaptive_worse_than_random_mean": sum(
                row["adaptive_minus_random_mean"] > 1e-8 for row in vals
            ),
            "banks_adaptive_tied_to_random_mean": sum(
                abs(row["adaptive_minus_random_mean"]) <= 1e-8 for row in vals
            ),
            "unweighted_mean_adaptive_minus_random_mean": (
                round(sum(row["adaptive_minus_random_mean"] for row in vals) / len(vals), 6)
                if vals else None
            ),
            "unweighted_mean_adaptive_minus_fixed": (
                round(sum(row["adaptive_minus_fixed"] for row in vals) / len(vals), 6)
                if vals else None
            ),
        })
    return {
        "experiment": "independent_voicebank_recording_budget_replication",
        "version": VERSION,
        "manifest_speaker_groups_self_reported": True,
        "source_paths_included_in_report": False,
        "protocols": ["0.2-disjoint-wav-cv", "0.3-random-baseline-sensitivity"],
        "random_repeats_per_fold": repeats,
        "requested_banks": len(banks),
        "evaluated_banks": len(successful),
        "excluded_banks": len(reports) - len(successful),
        "cross_bank_replication_adequate": len(successful) >= 2,
        "banks": reports,
        "summary_by_budget": summary,
        "limitations": [
            "Speaker-group independence is declared in the manifest, not acoustically verified.",
            "Each voicebank is evaluated separately; no WAV from another bank enters its acquisition pool.",
            "At least two successful independent speaker groups are required before reporting cross-bank replication.",
            "Reporting means across banks weights each speaker equally; feature-distance magnitude comparability is exploratory.",
            "Random selection percentiles are not confidence intervals across singers.",
            "The original fixed and adaptive policies, disjoint folds and normalized evaluation distances are unchanged.",
            "Mandarin s/sh/x fricative OTO acoustic-feature coverage is not synthesized-singing quality.",
            "Recording count is not duration or effort; near-duplicate audio cannot be detected by WAV SHA-256 alone.",
            "This was designed after examining a single-voicebank pilot and is exploratory.",
        ],
    }


def run_replication_manifest(path: Path, output: Path, *, repeats: int = 100) -> dict:
    raw = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    report = evaluate_replication(raw, repeats=repeats)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8"
    )
    return report
