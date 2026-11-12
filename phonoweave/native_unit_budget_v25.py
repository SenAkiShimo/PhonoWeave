"""Source-disjoint, equal-donor-budget proxy for EXISTING user-native unit pairs.

This is a narrow feasibility gate, not a renderer or a new phone recognizer.
No recording recommendations, phone split/merge, or original WAV/OTO edits.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

SCHEMA = "phonoweave.native_unit_budget_feasibility.v2.5"
MIN_DONORS_PER_LABEL = 1


def _members(archive: zipfile.ZipFile) -> tuple[str, str]:
    names = archive.namelist()
    indexes = [name for name in names if name.endswith("/source_index.csv")]
    features = [name for name in names if name.endswith("/feature_vectors.json")]
    if len(indexes) != 1 or len(features) != 1:
        raise ValueError("archive needs one source_index.csv and feature_vectors.json")
    if indexes[0].rsplit("/", 1)[0] != features[0].rsplit("/", 1)[0]:
        raise ValueError("source index and feature vectors are from different runs")
    for name in indexes + features:
        if archive.getinfo(name).file_size > 64 * 1024 * 1024:
            raise ValueError("source evidence file exceeds safe size")
    return indexes[0], features[0]


def _read_archive(path: Path) -> tuple[list[dict], str]:
    path = Path(path).expanduser().resolve(strict=True)
    raw_archive = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as archive:
        index_member, vector_member = _members(archive)
        index = list(csv.DictReader(io.StringIO(archive.read(index_member).decode("utf-8-sig"))))
        vectors = json.loads(archive.read(vector_member))
    if not isinstance(vectors, list):
        raise ValueError("feature_vectors.json must contain a list")
    if not index or len(index) != len(vectors):
        raise ValueError("source-index and feature-vector row count mismatch")
    required = {"bank", "symbol", "context", "source_wav", "source_pcm_sha256", "oto_line"}
    if not required.issubset(index[0]):
        raise ValueError("source-index missing required columns")
    positions = {}
    for v in vectors:
        if not isinstance(v, dict):
            raise ValueError("invalid acoustic feature row")
        key = (v.get("bank"), v.get("source_wav"), str(v.get("oto_line")))
        if key in positions:
            raise ValueError("duplicate acoustic source/OTO identity")
        x = v.get("features")
        if not isinstance(x, list) or len(x) != 12 or any(
            type(z) not in (float, int) or not math.isfinite(z) for z in x
        ):
            raise ValueError("expected finite 12-dimensional source-linked features")
        positions[key] = tuple(float(z) for z in x)
    merged = []
    seen = set()
    source_hashes = {}
    for row in index:
        key = tuple(row[k] for k in ("bank", "source_wav", "oto_line"))
        if key in seen or key not in positions:
            raise ValueError("source-index/feature-vector alignment mismatch")
        seen.add(key)
        pcm = row["source_pcm_sha256"]
        if not isinstance(pcm, str) or len(pcm) != 64 or any(c not in "0123456789abcdef" for c in pcm):
            raise ValueError("bad decoded-source PCM hash")
        wav = (row["bank"], row["source_wav"])
        if wav in source_hashes and source_hashes[wav] != pcm:
            raise ValueError("same bank WAV has conflicting PCM identities")
        source_hashes[wav] = pcm
        if not all(isinstance(row[k], str) and row[k] for k in ("bank", "symbol", "context", "source_wav")):
            raise ValueError("source identity missing")
        merged.append({
            "bank": row["bank"], "symbol": row["symbol"], "context": row["context"],
            "source_pcm_sha256": pcm, "features": positions[key],
            "source_wav": row["source_wav"],
            "source_alias": row.get("source_alias", ""),
            "style": row.get("style", ""),
        })
    if len(seen) != len(positions):
        raise ValueError("acoustic features lack corresponding source-index rows")
    return merged, sha256(raw_archive).hexdigest()


def _observations(rows: list[dict]) -> dict:
    """Collapse repeated OTO events within ONE PCM source before scoring."""
    grouped = defaultdict(list)
    for row in rows:
        key = (row["bank"], row["context"], row["symbol"], row["source_pcm_sha256"])
        grouped[key].append(np.asarray(row["features"], dtype=float))
    result = defaultdict(dict)
    for (bank, ctx, label, source), vectors in grouped.items():
        result[(bank, ctx, label)][source] = np.stack(vectors).mean(axis=0)
    return result


def compare_unit_pairs(
    rows: list[dict], pairs: list[tuple[str, str]], *,
    budget_per_label: int = MIN_DONORS_PER_LABEL,
    archive_sha256: str | None = None,
) -> dict:
    """Compare held-out SAME-CONTEXT features using the SAME donor instances.

    Two-label model: separate label-specific donor means.
    Shared-label model: pooled mean of the EXACT SAME donor observations.
    This tests a reconstruction proxy with equal donor material, not an equal
    verified recording-time cost, resampled audio or singing quality.
    """
    if (type(budget_per_label) is not int or budget_per_label < 1 or
            budget_per_label > 8):
        raise ValueError("donor budget must be an integer in [1,8]")
    if not isinstance(rows, list) or not rows:
        raise ValueError("at least one acoustic region required")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("at least one author-selected existing label pair required")
    if len(pairs) > 20:
        raise ValueError("at most 20 comparisons")
    labels = {r["symbol"] for r in rows}
    selected = set()
    for pair in pairs:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or not all(isinstance(x, str) and x for x in pair)
                or pair[0] == pair[1] or any(x not in labels for x in pair)):
            raise ValueError("pairs must name two distinct existing source-index labels")
        if frozenset(pair) in selected:
            raise ValueError("duplicate label pair")
        selected.add(frozenset(pair))
    indexed = _observations(rows)
    output = []
    for a, b in pairs:
        cohorts = sorted({(bank, ctx) for bank, ctx, sym in indexed if sym in (a,b)})
        both = 0
        scored = []
        source_shortages = []
        for bank, ctx in cohorts:
            A = indexed.get((bank, ctx, a), {})
            B = indexed.get((bank, ctx, b), {})
            if not A or not B:
                continue
            both += 1
            common_targets = sorted(set(A) & set(B))
            eligible = 0
            for target in common_targets:
                other_a = sorted(set(A) - {target})
                other_b = sorted(set(B) - {target})
                if len(other_a) < budget_per_label or len(other_b) < budget_per_label:
                    continue
                # Source selection is fixed by stable identifiers before
                # looking at target acoustic features.
                donor_a = [A[key] for key in other_a[:budget_per_label]]
                donor_b = [B[key] for key in other_b[:budget_per_label]]
                shared = np.mean(np.stack(donor_a + donor_b), axis=0)
                distinct_a = np.mean(np.stack(donor_a), axis=0)
                distinct_b = np.mean(np.stack(donor_b), axis=0)
                targets = (A[target], B[target])
                loss_separate = float(np.mean(
                    [np.mean((targets[0]-distinct_a)**2),
                     np.mean((targets[1]-distinct_b)**2)]))
                loss_shared = float(np.mean(
                    [np.mean((targets[0]-shared)**2),
                     np.mean((targets[1]-shared)**2)]))
                eligible += 1
                scored.append((bank, ctx, target, loss_separate, loss_shared))
            if eligible == 0:
                source_shortages.append({
                    "bank": bank, "context": ctx,
                    "distinct_pcm_sources_for_first_label": len(A),
                    "distinct_pcm_sources_for_second_label": len(B),
                    "co_recorded_target_pcm_sources": len(common_targets),
                    "reason": "insufficient_independent_same_context_source_donors",
                })
        # Do not treat 20 OTO entries from the same WAV as 20 independent trials.
        # Average target-source contrasts within bank, then bank summaries.
        bank_deltas = defaultdict(list)
        for bank, ctx, target, separate, shared in scored:
            bank_deltas[bank].append(shared - separate)
        bank_means = {bank: round(float(np.mean(ds)), 6)
                      for bank, ds in sorted(bank_deltas.items())}
        avg_delta = (round(float(np.mean(list(bank_means.values()))), 6)
                     if bank_means else None)
        output.append({
            "labels": [a,b],
            "same_context_cohorts_with_both_labels": both,
            "eligible_independent_target_sources": len(scored),
            "eligible_subbanks": sorted(bank_means),
            "blocked_cohort_count": len(source_shortages),
            "blocked_cohort_examples": source_shortages[:8],
            "status": "exploratory_equal_donor_proxy_only" if scored else
                      "insufficient_independent_source_repetitions",
            "pooled_minus_separate_mse_per_subbank": bank_means,
            "mean_pooled_minus_separate_mse": avg_delta,
            "interpretation": ("positive favors preserving existing labels under THIS spectral proxy"
                               if scored else "no fair comparison possible; do not decide split/merge"),
            "new_recording_lines_authorized": [],
            "unit_merge_or_split_authorized": False,
        })
    return {
        "schema": SCHEMA,
        "budget_per_label": budget_per_label,
        "matched_material": "exact_same_training_acoustic_sources_for_separate_and_shared_models",
        "equal_heldout_pcm_source_exclusion": True,
        "source_archive_sha256": archive_sha256,
        "input_rows": len(rows),
        "compared_existing_label_pairs": output,
        "automatic_recording_lines": [],
        "automatic_unit_changes": [],
        "claims_not_supported": [
            "proxy feature reconstruction is not rendered singing or listening quality",
            "two vs one unit IDs does not measure recording minutes saved",
            "distinct PCM WAVs do not establish independent human takes",
            "OTO-relative features do not establish true phoneme boundaries",
            "one source reused for many OTO entries never counts as independent repeats",
        ],
    }


def run_archive(archive: Path, pairs: list[tuple[str,str]], *,
                budget_per_label: int = 1) -> dict:
    rows, digest = _read_archive(archive)
    return compare_unit_pairs(rows, pairs, budget_per_label=budget_per_label,
                              archive_sha256=digest)


def attach_unit_budget_feasibility(report: dict, archive: Path,
                                   pairs: list[tuple[str,str]], *,
                                   budget_per_label: int = 1) -> dict:
    if not isinstance(report, dict) or not isinstance(report.get("recording_plan"), dict):
        raise ValueError("recording plan report required")
    audit = run_archive(archive, pairs, budget_per_label=budget_per_label)
    reviewed = report.get("existing_sound_unit_review")
    if isinstance(reviewed, dict):
        names = set(reviewed.get("original_user_labels", []))
        if any(set(p) - names for p in pairs):
            raise ValueError("pair not in existing sound unit review")
    return {**report, "existing_unit_budget_feasibility": audit}
