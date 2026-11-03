"""Read-only, blinded vote intake with exact trial-set and media integrity checks."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from .synthesis_relevance_rendered_intake import (
    BLIND_SCHEMA, CHOICES, KEY_SCHEMA, _canonical, _digest,
)

SCORE_SCHEMA = "phonoweave.synthesis_relevance.blind_ab_score.v1"
RESPONSE_SCHEMA = "phonoweave.synthesis_relevance.blind_ab_response.v1"


def compile_responses(manifest: dict, key: dict, responses: list[dict],
                      participant_root: Path) -> dict:
    """Unblind only after verifying media matches its private receipt-derived key.

    Votes are descriptive only: no p-values, effective-listener inflation or
    independence inference from clustered repeated targets is claimed.
    """
    root = participant_root.expanduser().resolve(strict=True)
    if manifest.get("schema") != BLIND_SCHEMA or key.get("schema") != KEY_SCHEMA:
        raise ValueError("unsupported blind A/B schema")
    fp = _canonical(manifest)
    if fp != key.get("participant_manifest_sha256"):
        raise ValueError("private key and public participant manifest mismatch")
    trials = manifest.get("trials")
    known = key.get("trials")
    if not isinstance(trials, list) or not isinstance(known, list):
        raise ValueError("invalid blind trial collection")
    visible = {row["trial_id"]: row for row in trials}
    mapping = {row["trial_id"]: row for row in known}
    if (len(visible) != len(trials) or len(mapping) != len(known)
            or not visible or set(visible) != set(mapping)):
        raise ValueError("duplicate or mismatched trial ID")
    used = set()
    for tid, row in visible.items():
        private = mapping[tid]
        if {private.get("A_condition"), private.get("B_condition")} != {"pooled", "split"}:
            raise ValueError("invalid private condition map")
        for arm in ("A", "B"):
            rel = row.get(arm)
            if (not isinstance(rel, str) or rel != f"media/{tid}_{arm}.wav"
                    or rel in used):
                raise ValueError("invalid participant media path")
            used.add(rel)
            path = (root / rel).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("participant media escapes root")
            if _digest(path) != private.get(f"{arm}_sha256"):
                raise ValueError("participant WAV changed since blinding")
    seen_listeners = set()
    votes = Counter()
    by_trial = {t: Counter() for t in visible}
    for response in responses:
        if (response.get("schema") != RESPONSE_SCHEMA
                or response.get("participant_manifest_sha256") != fp):
            raise ValueError("response is for a different blind session")
        listener = response.get("listener_id")
        if not isinstance(listener, str) or not listener.strip() or listener in seen_listeners:
            raise ValueError("empty or repeated listener ID")
        seen_listeners.add(listener)
        rows = response.get("votes")
        if not isinstance(rows, list) or len(rows) != len(trials):
            raise ValueError("incomplete or duplicate participant vote list")
        per = {}
        for row in rows:
            tid, choice = row.get("trial_id"), row.get("choice")
            if tid not in visible or tid in per or choice not in CHOICES:
                raise ValueError("invalid or duplicate trial response")
            per[tid] = choice
        if set(per) != set(visible):
            raise ValueError("response trial set mismatch")
        for tid, choice in per.items():
            if choice == "unrated":
                continue
            label = mapping[tid].get(f"{choice}_condition") if choice in ("A", "B") else choice
            if label not in {"pooled", "split", "tie", "unusable"}:
                raise ValueError("invalid scoring condition")
            votes[label] += 1
            by_trial[tid][label] += 1
    return {
        "schema": SCORE_SCHEMA,
        "status": "descriptive_only_no_perceptual_significance_claim",
        "participant_manifest_sha256": fp,
        "listeners": len(responses),
        "trials": len(trials),
        "votes": {kind: votes[kind] for kind in ("pooled", "split", "tie", "unusable")},
        "unrated_or_missing_votes": len(responses) * len(trials) - sum(votes.values()),
        "per_trial_votes": [
            {"trial_id": tid, "target_take_id": mapping[tid]["target_take_id"],
             **{k: by_trial[tid][k] for k in ("pooled", "split", "tie", "unusable")}}
            for tid in sorted(visible)
        ],
        "limitations": [
            "No power analysis, sampling design, confidence interval or significance test.",
            "Repeated stimuli/targets within the same speaker are clustered, not independent singers.",
            "External render donor-route statements cannot be independently verified by this compiler.",
            "Scoring does not certify actual singing synthesis or acoustic parity.",
        ],
    }


def score_directory(blind_root: Path, private_key: Path,
                    responses: list[Path], output: Path) -> dict:
    blind_root = blind_root.expanduser().resolve(strict=True)
    private_key = private_key.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if (output.exists() or output == blind_root or blind_root in output.parents
            or output == private_key):
        raise ValueError("output exists or is inside the blind media root/private key")
    manifest = json.loads((blind_root / "manifest.json").read_text(encoding="utf-8"))
    key = json.loads(private_key.read_text(encoding="utf-8"))
    data = [
        json.loads(path.expanduser().read_text(encoding="utf-8"))
        for path in responses
    ]
    report = compile_responses(manifest, key, data, blind_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report
