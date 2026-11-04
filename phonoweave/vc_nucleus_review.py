"""Human-confirmed nucleus-label overlay for a provisional OTO VC inventory.

This read-only analysis deliberately separates (a) author-confirmed alias
semantics, (b) provisional OTO-relative spectral proxies, (c) IPA/phone mapping
and (d) true phonetic boundary and synthesis validation. Only (a) is confirmed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import tempfile

SCHEMA = "phonoweave.oto_vc_nucleus_review.v0.1"
PROFILE_SCHEMA = "phonoweave.author_confirmed_vc_nucleus_labels.v0.1"
SOURCE_SCHEMA = "phonoweave.oto_surface_vc_probe.v0.1"
MEASURABLE = "provisional_oto_relative_only"


def _source_candidates(data: dict) -> list[dict]:
    if data.get("schema") != SOURCE_SCHEMA:
        raise ValueError("unsupported OTO VC evidence schema")
    candidates = data.get("candidates", data.get("provisional_candidates"))
    if not isinstance(candidates, list):
        raise ValueError("missing VC candidate array")
    return candidates


def _validate_profile(profile: dict, data: dict) -> dict[str, dict]:
    if not isinstance(profile, dict) or profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError("invalid author-confirmed nucleus profile")
    if profile.get("review_scope") != "nucleus_label_semantics_only":
        raise ValueError("review must explicitly be limited to nucleus label semantics")
    source = data.get("source", {})
    fingerprint = profile.get("source_oto_sha256")
    actual = source.get("oto_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or fingerprint != actual:
        raise ValueError("confirmed profile does not match source OTO hash")
    entries = profile.get("labels")
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("expected at least two confirmed nucleus labels")
    result = {}
    identities = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid label entry")
        alias = entry.get("alias_left_surface")
        nucleus = entry.get("nucleus_class_id")
        examples = entry.get("speaker_examples")
        if (not isinstance(alias, str) or not alias or alias.strip() != alias
                or alias in result or not isinstance(nucleus, str)
                or not nucleus or nucleus in identities
                or not isinstance(examples, list) or not examples
                or any(not isinstance(x, str) or not x for x in examples)
                or entry.get("author_confirmed") is not True):
            raise ValueError("ambiguous, duplicate or unconfirmed nucleus identity")
        result[alias] = entry
        identities.add(nucleus)
    return result


def compile_confirmed_nucleus_review(evidence: dict, profile: dict) -> dict:
    """Enumerate candidate triples and comparability, without inferring IPA."""
    candidates = _source_candidates(evidence)
    labels = _validate_profile(profile, evidence)
    sorted_labels = sorted(labels)
    expected_prefix = profile.get("alias_prefix")
    if not isinstance(expected_prefix, str) or not expected_prefix:
        raise ValueError("alias prefix must be explicit")
    indexed = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        if not isinstance(row, dict):
            raise ValueError("non-object candidate")
        alias = row.get("alias")
        left, right = row.get("left_surface"), row.get("right_surface")
        if (not isinstance(alias, str) or not isinstance(left, str)
                or not isinstance(right, str) or not right):
            raise ValueError("invalid OTO candidate alias surface")
        if left not in labels:
            continue
        if alias != f"{expected_prefix}{left} {right}":
            raise ValueError("OTO alias is inconsistent with confirmed surface labels")
        if not isinstance(row.get("oto_line"), int) or row["oto_line"] < 1:
            raise ValueError("missing original OTO line number")
        if not isinstance(row.get("wav"), str) or not row["wav"]:
            raise ValueError("missing WAV source")
        indexed[right][left].append(row)
    groups = []
    for right in sorted(indexed):
        family = indexed[right]
        all_present = all(family.get(label) for label in sorted_labels)
        measurable = {
            label: [r for r in family.get(label, []) if
                    r.get("spectral_status", r.get("spectrum_status")) == MEASURABLE]
            for label in sorted_labels
        }
        complete_measurable = all(measurable[label] for label in sorted_labels)
        common_wavs = (set.intersection(*(
            {r["wav"] for r in family[label]} for label in sorted_labels
        )) if all_present else set())
        common_measurable = (set.intersection(*(
            {r["wav"] for r in measurable[label]} for label in sorted_labels
        )) if complete_measurable else set())
        if common_measurable:
            category = "first_review_same_wav_three_measurable"
        elif complete_measurable:
            category = "second_review_three_measurable_different_wavs"
        elif all_present:
            category = "needs_better_window_or_manual_boundaries"
        else:
            category = "incomplete_nucleus_family"
        groups.append({
            "right_surface": right,
            "right_phone_identity": "unverified",
            "right_segment_role": "unverified_C_or_V",
            "same_wav_all_labels": sorted(common_wavs),
            "same_wav_all_measurable": sorted(common_measurable),
            "priority": category,
            "all_confirmed_labels_present": bool(all_present),
            "all_have_measurable_proxy": bool(complete_measurable),
            "nucleus_candidates": [{
                "left_surface": label,
                "author_confirmed_nucleus_class_id": labels[label]["nucleus_class_id"],
                "example_syllables": labels[label]["speaker_examples"],
                "oto_lines": sorted(r["oto_line"] for r in family.get(label, [])),
                "wavs": sorted({r["wav"] for r in family.get(label, [])}),
                "candidate_windows": len(family.get(label, [])),
                "measurable_provisional_windows": len(measurable[label]),
                "provisional_spectral_centroids_hz": [
                    r.get("spectral_centroid_hz", r.get("centroid_hz"))
                    for r in sorted(measurable[label], key=lambda x: x["oto_line"])
                ],
            } for label in sorted_labels],
        })
    categories = Counter(g["priority"] for g in groups)
    groups.sort(key=lambda g: (
        ["first_review_same_wav_three_measurable",
         "second_review_three_measurable_different_wavs",
         "needs_better_window_or_manual_boundaries",
         "incomplete_nucleus_family"].index(g["priority"]),
        g["right_surface"],
    ))
    return {
        "schema": SCHEMA,
        "status": "confirmed_nucleus_class_labels_unverified_phone_boundaries",
        "source_oto_sha256": profile["source_oto_sha256"],
        "source_archive_sha256": evidence.get("source", {}).get("archive_sha256"),
        "author_confirmed_labels": [
            {"surface": label,
             "nucleus_class_id": labels[label]["nucleus_class_id"],
             "speaker_examples": labels[label]["speaker_examples"],
             "review_scope": "nucleus_label_semantics_only"}
            for label in sorted_labels
        ],
        "summary": {
            "tracked_nucleus_labels": len(sorted_labels),
            "unique_right_surface_labels": len(groups),
            "complete_surface_groups": sum(g["all_confirmed_labels_present"] for g in groups),
            "three_measurable_proxy_groups": sum(g["all_have_measurable_proxy"] for g in groups),
            "same_wav_complete_groups": sum(bool(g["same_wav_all_labels"]) for g in groups),
            "same_wav_three_measurable_groups": sum(bool(g["same_wav_all_measurable"]) for g in groups),
            "review_priority_counts": dict(sorted(categories.items())),
            "total_candidates_with_confirmed_left_label": sum(
                len(rows) for family in indexed.values() for rows in family.values()),
        },
        "groups": groups,
        "limitations": [
            "The user confirmed pronunciation class labels, not a precise IPA realization or shared phoneme across syllable contexts.",
            "Right-side alias identity and role remain unreviewed, so CC/CV/VC/VV edge type is not automatically assigned.",
            "The OTO-relative window is not a human-reviewed acoustic VC transition boundary.",
            "Same-WAV candidates can share session/environment confounds and do not represent independent phonations.",
            "No automatically inferred split/share, audible improvement, or production alias/OTO edit is authorized.",
        ],
        "production_ready": False,
        "production_alias_export": "disabled",
    }


def write_confirmed_nucleus_review(report: dict, output: Path) -> Path:
    """Atomically write new JSON/CSV reviews without overwriting evidence."""
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("review output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pw-vc-nuclei-", dir=output.parent) as temporary:
        staging = Path(temporary) / "review"
        staging.mkdir()
        (staging / "confirmed_nucleus_groups.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (staging / "priority_comparison_queue.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            cols = ["right_surface", "priority", "same_wav_all_measurable",
                    "left_surface", "nucleus_class_id", "oto_lines", "wavs",
                    "measurable_windows", "right_phone_review", "right_role_review",
                    "boundary_reviewed", "synthesis_comparison_reviewed"]
            writer = csv.DictWriter(stream, fieldnames=cols)
            writer.writeheader()
            for group in report["groups"]:
                for left in group["nucleus_candidates"]:
                    writer.writerow({
                        "right_surface": group["right_surface"],
                        "priority": group["priority"],
                        "same_wav_all_measurable": "|".join(group["same_wav_all_measurable"]),
                        "left_surface": left["left_surface"],
                        "nucleus_class_id": left["author_confirmed_nucleus_class_id"],
                        "oto_lines": "|".join(map(str, left["oto_lines"])),
                        "wavs": "|".join(left["wavs"]),
                        "measurable_windows": left["measurable_provisional_windows"],
                        "right_phone_review": "", "right_role_review": "",
                        "boundary_reviewed": "false",
                        "synthesis_comparison_reviewed": "false",
                    })
        if output.exists():
            raise FileExistsError("review output appeared while publishing")
        os.rename(staging, output)
    return output


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="phonoweave review-verified-vc-nuclei")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--confirmed-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        evidence = json.loads(args.evidence.expanduser().read_text(encoding="utf-8"))
        profile = json.loads(args.confirmed_labels.expanduser().read_text(encoding="utf-8"))
        result = compile_confirmed_nucleus_review(evidence, profile)
        dest = args.output.expanduser().resolve()
        if dest in (args.evidence.expanduser().resolve(), args.confirmed_labels.expanduser().resolve()):
            raise ValueError("cannot overwrite a source evidence or label file")
        write_confirmed_nucleus_review(result, dest)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    summary = result["summary"]
    print("Complete surface families:", summary["complete_surface_groups"])
    print("All 3 provisional windows measurable:", summary["three_measurable_proxy_groups"])
    print("Same WAV, all 3 measurable:", summary["same_wav_three_measurable_groups"])
    print("Author-confirmed label classes only; phonetic boundaries remain unresolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
