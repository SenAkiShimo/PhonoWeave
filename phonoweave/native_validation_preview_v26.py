"""Minimal author-check preview for EXISTING user-native realization labels.

Find a few source-provenance-linked comparisons in the frozen index. These
are validation examples, NOT new recording instructions or invented phonemes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .native_unit_budget_v25 import _read_archive

SCHEMA = "phonoweave.minimal_native_validation_preview.v2.6"


def build_preview(rows: list[dict[str, Any]], labels: list[str], *,
                  max_cases: int = 3, archive_sha256: str | None = None) -> dict:
    if not isinstance(rows, list) or not rows:
        raise ValueError("source-linked acoustic rows are required")
    if not isinstance(labels, list) or len(labels) < 2 or len(labels) > 5:
        raise ValueError("use 2-5 existing author-defined labels")
    if any(not isinstance(x, str) or not x.strip() for x in labels) or len(set(labels)) != len(labels):
        raise ValueError("labels must be distinct nonempty original strings")
    if type(max_cases) is not int or not 1 <= max_cases <= 8:
        raise ValueError("max_cases must be an integer in [1, 8]")
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not all(isinstance(r.get(k), str) and r[k] for k in
                   ("bank", "context", "symbol", "source_pcm_sha256", "source_wav", "source_alias")):
            raise ValueError("missing source/alias provenance in frozen index")
        digest = r["source_pcm_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid source PCM SHA-256")
        if r["symbol"] in labels:
            groups[(r["bank"], r["context"], digest)][r["symbol"]].append(r)
    candidates = []
    for (bank, context, pcm), per_label in sorted(groups.items()):
        if not all(label in per_label for label in labels):
            continue
        items = []
        for label in labels:
            distinct = {(x["source_wav"], x["source_alias"]) for x in per_label[label]}
            if len(distinct) != 1:
                items = []
                break
            items.append({"original_label": label,
                          "original_alias": next(iter(distinct))[1],
                          "source_wav": next(iter(distinct))[0]})
        if not items:
            continue
        wavs = {item["source_wav"] for item in items}
        if len(wavs) != 1:
            # Each proposed case must correspond to one historical physical
            # WAV, rather than pretending multiple files were one performance.
            continue
        candidates.append({
            "bank": bank, "style": str(per_label[labels[0]][0].get("style", bank.split("_")[0])),
            "following_context": context,
            "source_wav": next(iter(wavs)),
            "source_pcm_sha256": pcm,
            "examples": items,
        })
    selected = []
    seen_wavs, seen_contexts, seen_banks, seen_styles, seen_pcm = set(), set(), set(), set(), set()
    remaining = sorted(candidates, key=lambda x: (
        x["following_context"], x["bank"], x["source_wav"], x["source_pcm_sha256"]))
    while remaining and len(selected) < max_cases:
        def priority(x):
            return (int(x["following_context"] not in seen_contexts),
                    int(x["source_wav"] not in seen_wavs),
                    int(x["style"] not in seen_styles),
                    int(x["bank"] not in seen_banks),
                    int(x["source_pcm_sha256"] not in seen_pcm))
        chosen = max(remaining, key=priority)
        remaining.remove(chosen)
        selected.append({
            "case_number": len(selected) + 1,
            **chosen,
            "review_question": "Are these existing labels meaningful distinct realizations in this context?",
            "recording_check": "If a new independent take is needed later, can the author naturally record each example?",
            "new_recording_required_now": False,
            "verified_phone_boundary": False,
            "verified_independent_singing_take": False,
        })
        seen_contexts.add(chosen["following_context"])
        seen_wavs.add(chosen["source_wav"])
        seen_banks.add(chosen["bank"])
        seen_styles.add(chosen["style"])
        seen_pcm.add(chosen["source_pcm_sha256"])
        remaining = [x for x in remaining
                     if x["following_context"] not in seen_contexts
                     and x["source_pcm_sha256"] not in seen_pcm]
    return {
        "schema": SCHEMA, "original_user_labels": labels,
        "source_archive_sha256": archive_sha256,
        "source_linked_candidate_cases": len(candidates),
        "selected_review_cases": selected,
        "selected_count": len(selected),
        "status": "author_preview_only" if selected else "insufficient_co_recorded_original_label_examples",
        "review_decisions": ["label_and_alias_mapping_correct",
                             "distinctions_meaningful_to_author",
                             "suggested_context_naturally_recordable"],
        "new_recording_lines": [],
        "automatic_unit_merges_or_splits": [],
        "limitations": [
            "These original aliases describe OTO-derived windows, not validated phones or full recording scripts.",
            "Co-occurrence in one WAV does not mean distinct independent singing takes.",
            "Review selection is for human comprehension, not measured recording efficiency.",
            "No input WAV/OTO was modified; no new IPA or user symbols are generated.",
        ],
    }


def run_archive(path: Path, labels: list[str], max_cases: int = 3) -> dict:
    rows, digest = _read_archive(path)
    return build_preview(rows, labels, max_cases=max_cases, archive_sha256=digest)


def attach_validation_preview(report: dict, archive: Path, labels: list[str],
                              *, max_cases: int = 3) -> dict:
    if not isinstance(report, dict) or not isinstance(report.get("recording_plan"), dict):
        raise ValueError("existing recording plan report is required")
    result = run_archive(archive, labels, max_cases)
    previous = report.get("existing_sound_unit_review")
    if isinstance(previous, dict) and any(x not in previous.get("original_user_labels", [])
                                           for x in labels):
        raise ValueError("preview labels disagree with earlier original-label audit")
    return {**report, "minimal_unit_validation_preview": result}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Read-only minimal source-linked author validation preview")
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--label", action="append", required=True)
    p.add_argument("--max-cases", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        p.error("output already exists; refusing overwrite")
    result = run_archive(args.source_archive, args.label, args.max_cases)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"status": result["status"], "selected_count": result["selected_count"],
                      "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
