"""v2.8: source-template-disjoint check of EXISTING author-native realizations.

This is an OTO-derived, model-rendered, supervised *diagnostic*, not unknown
phone discovery, an OpenUtau synthesis comparison, or a recording generator.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from .native_unit_budget_v25 import _read_archive

SCHEMA = "phonoweave.native_source_disjoint_context_review.v2.8"


def _triplets(rows: list[dict[str, Any]], labels: list[str]) -> tuple[list[dict], dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("source-linked acoustic rows are required")
    if (not isinstance(labels, list) or not 2 <= len(labels) <= 5
            or any(not isinstance(x, str) or not x.strip() for x in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("choose 2–5 distinct, existing original labels")
    grouped: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    pcm_by_wav: dict[tuple[str, str], str] = {}
    wav_by_pcm: dict[tuple[str, str], str] = {}
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError("invalid source feature row")
        if not all(isinstance(r.get(k), str) and r[k] for k in (
                "bank", "context", "symbol", "source_pcm_sha256", "source_wav")):
            raise ValueError("source-linked bank/context/WAV identity missing")
        digest = r["source_pcm_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid decoded PCM SHA-256")
        bank, wav = r["bank"], r["source_wav"]
        old_hash = pcm_by_wav.setdefault((bank, wav), digest)
        if old_hash != digest:
            raise ValueError("same bank/WAV has conflicting PCM source fingerprints")
        old_wav = wav_by_pcm.setdefault((bank, digest), wav)
        if old_wav != wav:
            raise ValueError("different WAV paths share decoded PCM within a bank: do not count as separate sources")
        if r["symbol"] not in labels:
            continue
        v = np.asarray(r.get("features"), dtype=float)
        if v.shape != (12,) or not np.all(np.isfinite(v)):
            raise ValueError("expected finite 12-dimensional OTO-derived feature")
        grouped[(bank, r["context"], digest)][r["symbol"]].append({
            "wav": wav, "vector": v,
        })
    if not any(label in per for per in grouped.values() for label in labels):
        raise ValueError("none of the author-supplied labels exists in archive")
    triplets, ambiguous = [], 0
    for (bank, ctx, digest), per in sorted(grouped.items()):
        if not all(label in per for label in labels):
            continue
        if any(len(per[label]) != 1 for label in labels):
            ambiguous += 1
            continue
        records = [per[label][0] for label in labels]
        wavs = {item["wav"] for item in records}
        if len(wavs) != 1:
            ambiguous += 1
            continue
        vectors = np.stack([item["vector"] for item in records])
        triplets.append({
            "bank": bank, "following_context": ctx, "pcm": digest,
            "source_wav": next(iter(wavs)), "raw": vectors,
            # This demeaning intentionally uses all known, co-recorded labels
            # at evaluation: a paired diagnostic, NOT blind text-to-audio inference.
            "paired": vectors - vectors.mean(axis=0, keepdims=True),
        })
    return triplets, {
        "ambiguous_or_duplicate_paired_groups": ambiguous,
        "original_author_labels": labels[:],
    }


def _centroid(train: list[dict], templates: list[str], key: str,
              label_count: int, permutations: dict[str, np.ndarray] | None = None) -> np.ndarray:
    values = []
    for j in range(label_count):
        by_template = []
        for wav in templates:
            members = [g for g in train if g["source_wav"] == wav]
            if not members:
                raise AssertionError("empty training template")
            index = int(permutations[wav][j]) if permutations is not None else j
            by_template.append(np.stack([g[key][index] for g in members]).mean(axis=0))
        values.append(np.stack(by_template).mean(axis=0))
    return np.stack(values)


def _score(groups: list[dict], centers: np.ndarray, key: str) -> np.ndarray:
    confusion = np.zeros((centers.shape[0], centers.shape[0]), dtype=int)
    for g in groups:
        d = ((g[key][:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        predictions = d.argmin(axis=1)
        for actual, predicted in enumerate(predictions):
            confusion[actual, int(predicted)] += 1
    return confusion


def evaluate_existing_labels(
    rows: list[dict[str, Any]], labels: list[str], *,
    min_train_source_templates: int = 2,
    null_repetitions: int = 32,
    random_seed: int = 20261111,
    archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Leave whole source WAV templates out, within one pitch/style bank.

    Compare direct 12D signatures with paired co-recorded contextual
    differences; use permuted training label identities as a negative control.
    Every subbank is the SAME singer's model-rendered voice, not a new speaker.
    """
    if (type(min_train_source_templates) is not int
            or not 2 <= min_train_source_templates <= 20):
        raise ValueError("min_train_source_templates must be between 2 and 20")
    if (type(null_repetitions) is not int
            or not 0 <= null_repetitions <= 128):
        raise ValueError("null_repetitions must be in [0,128]")
    if type(random_seed) is not int or not 0 <= random_seed < 2**32:
        raise ValueError("random_seed must fit an unsigned 32-bit integer")
    if archive_sha256 is not None:
        if not isinstance(archive_sha256, str) or len(archive_sha256) != 64:
            raise ValueError("invalid archive SHA-256")
    cases, diagnostics = _triplets(rows, labels)
    by_bank = defaultdict(list)
    for case in cases:
        by_bank[case["bank"]].append(case)
    raw_total = np.zeros((len(labels), len(labels)), dtype=int)
    paired_total = np.zeros_like(raw_total)
    bank_rows, stored_folds = [], []
    for bank in sorted(by_bank):
        bank_cases = by_bank[bank]
        templates = sorted({case["source_wav"] for case in bank_cases})
        bank_raw = np.zeros_like(raw_total)
        bank_paired = np.zeros_like(raw_total)
        source_summaries = []
        for held_wav in templates:
            held = [g for g in bank_cases if g["source_wav"] == held_wav]
            held_hashes = {g["pcm"] for g in held}
            train = [g for g in bank_cases
                     if g["source_wav"] != held_wav and g["pcm"] not in held_hashes]
            train_templates = sorted({g["source_wav"] for g in train})
            if len(train_templates) < min_train_source_templates:
                continue
            raw_centers = _centroid(train, train_templates, "raw", len(labels))
            paired_centers = _centroid(train, train_templates, "paired", len(labels))
            raw_conf = _score(held, raw_centers, "raw")
            paired_conf = _score(held, paired_centers, "paired")
            bank_raw += raw_conf
            bank_paired += paired_conf
            source_summaries.append({
                "heldout_source_wav": held_wav,
                "heldout_source_pcm_count": len(held_hashes),
                "heldout_context_group_count": len(held),
                "training_distinct_wav_templates": len(train_templates),
                "raw_correct": int(np.trace(raw_conf)),
                "paired_correct": int(np.trace(paired_conf)),
                "label_decisions": int(raw_conf.sum()),
            })
            stored_folds.append((held, train, train_templates))
        raw_total += bank_raw
        paired_total += bank_paired
        if source_summaries:
            bank_rows.append({
                "bank": bank,
                "co_recorded_complete_context_groups": len(bank_cases),
                "distinct_wav_templates": len(templates),
                "evaluated_wav_template_folds": len(source_summaries),
                "raw_accuracy": round(float(np.trace(bank_raw) / bank_raw.sum()), 6),
                "paired_accuracy": round(float(np.trace(bank_paired) / bank_paired.sum()), 6),
                "heldout_source_fold_details": source_summaries,
            })
    n_decisions = int(raw_total.sum())
    null = []
    if stored_folds and null_repetitions:
        rng = np.random.default_rng(random_seed)
        for _ in range(null_repetitions):
            correct, total = 0, 0
            for held, train, templates in stored_folds:
                permutations = {wav: rng.permutation(len(labels)) for wav in templates}
                centers = _centroid(train, templates, "paired", len(labels), permutations)
                cm = _score(held, centers, "paired")
                correct += int(np.trace(cm))
                total += int(cm.sum())
            null.append(correct / total)
    # Avoid a statistical p-value: folds/banks are highly dependent on one singer.
    return {
        "schema": SCHEMA,
        "original_author_labels": labels[:],
        "source_archive_sha256": archive_sha256,
        "input_oto_relative_feature_rows": len(rows),
        "complete_paired_source_context_groups": len(cases),
        **diagnostics,
        "eligible_label_decisions": n_decisions,
        "heldout_template_folds": len(stored_folds),
        "per_bank": bank_rows,
        "raw_correct": int(np.trace(raw_total)),
        "paired_context_centered_correct": int(np.trace(paired_total)),
        "raw_accuracy": round(float(np.trace(raw_total) / n_decisions), 6)
                        if n_decisions else None,
        "paired_context_centered_accuracy":
            round(float(np.trace(paired_total) / n_decisions), 6)
            if n_decisions else None,
        "raw_confusion_actual_rows_predicted_columns": raw_total.tolist(),
        "paired_confusion_actual_rows_predicted_columns": paired_total.tolist(),
        "training_label_permutation_negative_control": {
            "repetitions": len(null),
            "mean_accuracy": round(float(np.mean(null)), 6) if null else None,
            "max_accuracy": round(float(max(null)), 6) if null else None,
            "not_a_calibrated_significance_test": True,
        },
        "review_status": (
            "exploratory_existing_label_source_disjoint_diagnostic"
            if n_decisions else "insufficient_distinct_wav_template_folds"
        ),
        "author_confirmed_label_distinctions_preserved": True,
        "automatic_new_recordings": [],
        "automatic_unit_merges_or_splits": [],
        "limitations": [
            "Known author labels and OTO-relative 12D signatures; not new phone discovery.",
            "The paired diagnostic centers heldout samples using ALL known labels; not blind inference.",
            "All seven pitch/style subbanks are synthetic/model-rendered voices of ONE singer.",
            "Distinct source WAV templates do not establish independent human performances.",
            "The 420 correlated label decisions are not 420 independent singer trials.",
            "No rendered-song, subjective quality or matched recording-minute evidence.",
        ],
    }


def run_archive(archive: Path, labels: list[str], *,
                null_repetitions: int = 32) -> dict:
    rows, digest = _read_archive(archive)
    return evaluate_existing_labels(
        rows, labels, null_repetitions=null_repetitions,
        archive_sha256=digest,
    )


def attach_context_review(report: dict, archive: Path, labels: list[str],
                          *, null_repetitions: int = 32) -> dict:
    if not isinstance(report, dict) or not isinstance(report.get("recording_plan"), dict):
        raise ValueError("existing recording plan report required")
    if "existing_native_context_stability" in report:
        raise ValueError("existing context review must not be overwritten")
    author = report.get("native_author_feedback_review")
    if isinstance(author, dict):
        if author.get("author_confirmation", {}).get(
                "desire_to_keep_existing_realization_labels_distinct") != "yes":
            raise ValueError("author has not confirmed these original distinctions")
        if labels != author.get("original_user_labels"):
            raise ValueError("source labels disagree with author's confirmed labels")
    review = run_archive(archive, labels, null_repetitions=null_repetitions)
    return {**report, "existing_native_context_stability": review}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Read-only template-disjoint diagnostic of existing user-native units"
    )
    p.add_argument("--source-archive", required=True, type=Path)
    p.add_argument("--label", required=True, action="append")
    p.add_argument("--null-repetitions", type=int, default=32)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    target = args.output.expanduser().resolve()
    if target.exists():
        p.error("output already exists")
    if not target.parent.is_dir():
        p.error("output parent directory must exist")
    result = run_archive(args.source_archive, args.label,
                         null_repetitions=args.null_repetitions)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": result["review_status"],
        "eligible_label_decisions": result["eligible_label_decisions"],
        "automatic_new_recordings": result["automatic_new_recordings"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
