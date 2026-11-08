"""Provisional fricative onset proposals and human-review gate for v0.5 VC clips.

This module NEVER infers a verified boundary or authorizes unit split/merge.
It checks the exact 600-ms WAV excerpts against the original review manifest,
provides explicitly *unverified* /sh/ onset brackets, and validates a
separate human-edited annotation CSV. Nasal /m/ is intentionally not detected
using the fricative rule.
"""
from __future__ import annotations

from collections import Counter
import csv
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
import wave

import numpy as np

SOURCE_SCHEMA = "phonoweave.vc_boundary_feasibility_gate.v0.1"
PROPOSAL_SCHEMA = "phonoweave.vc_manual_boundary_proposals.v0.1"
REVIEW_SCHEMA = "phonoweave.vc_boundary_human_review.v0.1"
CSV_COLUMNS = (
    "right_surface", "left_surface", "original_alias", "source_wav",
    "oto_line", "boundary_onset_ms_in_excerpt", "boundary_offset_ms_in_excerpt",
    "target_sound_present", "listening_comments", "reviewer",
)
LEFT_ORDER = ("i", "ir", "i0")
RIGHT_ALLOWLIST = ("m", "sh")
FRAME_MS = 16
STEP_MS = 2


def _sha_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _audio(path: Path, expected_hash: str) -> tuple[np.ndarray, int]:
    if not path.is_file() or _sha_bytes(path.read_bytes()) != expected_hash:
        raise ValueError(f"excerpt source SHA-256 mismatch: {path.name}")
    try:
        with wave.open(str(path), "rb") as wav:
            if (wav.getnchannels() != 1 or wav.getsampwidth() != 2
                    or wav.getcomptype() != "NONE"
                    or not 8000 <= wav.getframerate() <= 96000):
                raise ValueError("excerpt must be uncompressed mono PCM16")
            rate = wav.getframerate()
            frames = wav.getnframes()
            raw = wav.readframes(frames)
    except (wave.Error, EOFError, OSError) as exc:
        raise ValueError("invalid WAV excerpt") from exc
    if len(raw) != frames * 2 or not 0.58 <= frames / rate <= 0.62:
        raise ValueError("excerpt duration must be approximately 600 ms")
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768, rate


def _load_review_pack(root: Path) -> tuple[dict, bytes, list[dict]]:
    root = Path(root).expanduser().resolve(strict=True)
    raw = (root / "review_manifest.json").read_bytes()
    data = json.loads(raw)
    if data.get("schema") != SOURCE_SCHEMA:
        raise ValueError("unrecognized boundary-feasibility source schema")
    if data.get("status") != "auditory_boundary_review_required_no_split_share_evidence":
        raise ValueError("source is not an unreviewed boundary feasibility pack")
    groups = data.get("review_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("review pack is missing its groups")
    records = []
    keys = set()
    for group in groups:
        right = group.get("right_surface")
        if right not in RIGHT_ALLOWLIST:
            raise ValueError("unexpected right-hand alias in frozen pilot")
        clips = group.get("clip_records")
        if not isinstance(clips, list):
            raise ValueError("missing clip records")
        for row in clips:
            left = row.get("left_surface")
            if left not in LEFT_ORDER or (right, left) in keys:
                raise ValueError("unexpected or duplicate left/right excerpt")
            keys.add((right, left))
            if row.get("excerpt_filename") != f"right_{right}/{left}.wav":
                raise ValueError("noncanonical excerpt filename")
            if row.get("excerpt_oto_anchor_ms") != 240:
                raise ValueError("unexpected OTO reference point")
            if row.get("boundary_verified") is not False:
                raise ValueError("source pack is not an unreviewed boundary package")
            p = (root / row["excerpt_filename"]).resolve(strict=True)
            if not p.is_relative_to(root):
                raise ValueError("excerpt symlink escapes the reviewed folder")
            audio, rate = _audio(p, row["excerpt_sha256"])
            records.append({
                "right": right, "left": left,
                "original_alias": row["original_alias"],
                "source_wav": row["source_wav"], "oto_line": row["oto_line"],
                "relative_audio": row["excerpt_filename"], "audio_sha256": row["excerpt_sha256"],
                "oto_anchor_ms": 240, "sample_rate": rate,
                "source_wav_sha256": row["source_wav_sha256"],
                "_samples": audio,
            })
    if keys != set((right, left) for right in ("m", "sh") for left in LEFT_ORDER):
        raise ValueError("frozen v0.6 pilot requires the exact six verified excerpts")
    return data, raw, sorted(records, key=lambda x: (x["right"], LEFT_ORDER.index(x["left"])))


def _frame(audio: np.ndarray, rate: int, center_ms: float) -> tuple[float, float, float] | None:
    length = round(rate * FRAME_MS / 1000)
    start = round(center_ms * rate / 1000) - length // 2
    if start < 0 or start + length > len(audio):
        return None
    x = audio[start:start + length]
    rms_dbfs = 20 * math.log10(max(float(np.sqrt(np.mean(x * x))), 1e-12))
    spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(length))) ** 2
    freq = np.fft.rfftfreq(length, 1 / rate)
    usable = (freq >= 150) & (freq <= min(8000, rate / 2))
    total = float(np.sum(spec[usable]))
    if total < 1e-20:
        return rms_dbfs, 0.0, 0.0
    high = float(np.sum(spec[(freq >= 2200) & usable]))
    centroid = float(np.sum(spec[usable] * freq[usable]) / total)
    return rms_dbfs, high / total, centroid


def suggest_sh_noise_bracket(samples: np.ndarray, rate: int, *,
                             anchor_ms: float = 240) -> dict:
    """High-frequency frication *cue*, not a perceptually verified onset.

    Require low pre-anchor HF baseline, actual signal energy, and persistent
    broadband noise. Return lower/high threshold crossing interval, not a
    single alleged phoneme boundary. Thresholds are exploratory and frozen
    in the report, not trained/tuned against held-out speech.
    """
    if samples.ndim != 1 or len(samples) < round(rate * .55):
        raise ValueError("invalid excerpt audio for frication screening")
    times = range(round(anchor_ms - 80), round(anchor_ms + 170), STEP_MS)
    frames = [(t, *_frame(samples, rate, t)) for t in times]
    before = [f[2] for f in frames if anchor_ms-80 <= f[0] <= anchor_ms-22]
    if len(before) < 10 or np.median(before) >= .10:
        return {"status": "abstain_nonquiet_preanchor_baseline", "bracket_ms": None}
    stable_strong = []
    for i in range(len(frames) - 8):
        sample = frames[i:i+9]
        # Persistence 9 frames * 2ms, with overlapping 16ms windows.
        if all(rms > -53 and ratio >= .72 and centroid > 2400
               for _, rms, ratio, centroid in sample):
            stable_strong.append(frames[i][0])
            break
    if not stable_strong:
        return {"status": "abstain_no_sustained_high_frequency_frication", "bracket_ms": None}
    strong = stable_strong[0]
    lows = [
        t for t, rms, ratio, centroid in frames
        if (anchor_ms - 25 <= t <= strong and rms > -53 and ratio >= .20)
    ]
    if not lows:
        return {"status": "abstain_no_gradual_onset_bracket", "bracket_ms": None}
    first = min(lows)
    if strong - first > 45:
        return {"status": "abstain_ambiguous_onset", "bracket_ms": None}
    return {
        "status": "unverified_noise_onset_bracket",
        "bracket_ms": [first, strong],
        "offset_from_oto_anchor_ms": [round(first-anchor_ms, 1), round(strong-anchor_ms, 1)],
        "evidence_type": "sustained_2p2_to_8khz_fraction_not_phone_boundary",
    }


def _envelope(samples: np.ndarray, rate: int, n_bins: int = 300) -> list[list[float]]:
    # Downsampled waveform min/max for the local offline annotation UI.
    bins = np.array_split(samples, n_bins)
    return [
        [round(float(np.min(segment)), 4), round(float(np.max(segment)), 4)]
        for segment in bins
    ]


def prepare_manual_boundary_review(pack: Path) -> dict:
    """Validate source WAV hashes, then construct a repeatable human-review UI model."""
    manifest, raw, source = _load_review_pack(pack)
    rows = []
    for record in source:
        right = record["right"]
        if right == "sh":
            cue = suggest_sh_noise_bracket(record["_samples"], record["sample_rate"])
        else:
            cue = {"status": "manual_nasal_transition_review_only", "bracket_ms": None}
        rows.append({
            **{k: v for k, v in record.items() if k != "_samples"},
            "screening": cue,
            "waveform_envelope": _envelope(record["_samples"], record["sample_rate"]),
        })
    return {
        "schema": PROPOSAL_SCHEMA,
        "status": "proposals_only_human_listening_required",
        "original_review_manifest_sha256": _sha_bytes(raw),
        "original_source_archive_sha256": manifest["source"]["archive_sha256"],
        "original_source_oto_sha256": manifest["source"]["oto_sha256"],
        "frame_ms": FRAME_MS, "hop_ms": STEP_MS,
        "screening_rules": {
            "nasals": "abstain",
            "sh": "HF fraction ≥0.20 to sustained fraction ≥0.72 and centroid >2400Hz",
            "hf_range_hz": [2200, 8000],
            "passive_rms_floor_dbfs": -53,
            "note": "pre-specified diagnostic thresholds; not validated against gold annotations",
        },
        "clips": rows,
        "claimed_true_boundaries": 0,
        "approved_split_or_share_decisions": 0,
    }


def validate_human_review(pack: Path, csv_path: Path) -> dict:
    """Report review completeness, without mistaking annotations for synthesis evidence."""
    model = prepare_manual_boundary_review(pack)
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or set(reader.fieldnames) != set(CSV_COLUMNS):
            raise ValueError("review CSV header does not match frozen schema")
        data = list(reader)
    expected = {
        (r["right"], r["left"]): r for r in model["clips"]
    }
    seen = set()
    checks = []
    for row in data:
        key = (row["right_surface"], row["left_surface"])
        if key not in expected or key in seen:
            raise ValueError("review CSV changed or duplicated a frozen target")
        seen.add(key)
        record = expected[key]
        if (row["original_alias"] != record["original_alias"]
                or row["source_wav"] != record["source_wav"]
                or row["oto_line"] != str(record["oto_line"])):
            raise ValueError("review source identity is altered")
        answer = row["target_sound_present"].strip().lower()
        if answer not in ("yes", "no", "uncertain", "unreviewed"):
            raise ValueError("invalid target_sound_present value")
        onset_str = row["boundary_onset_ms_in_excerpt"].strip()
        offset_str = row["boundary_offset_ms_in_excerpt"].strip()
        reviewer = row["reviewer"].strip()
        note = row["listening_comments"].strip()
        verified = False
        interval = None
        if answer == "yes":
            if not reviewer or not onset_str or not offset_str:
                verified = False
            else:
                try:
                    a, b = float(onset_str), float(offset_str)
                except ValueError as exc:
                    raise ValueError("invalid boundary time") from exc
                if not (math.isfinite(a) and math.isfinite(b) and 0 <= a <= b <= 600):
                    raise ValueError("boundary interval must fit the excerpt")
                verified = True
                interval = [a, b]
        elif answer in ("no", "uncertain"):
            # Negative/ambiguous judgments require justification and reviewer.
            verified = bool(reviewer and note and not onset_str and not offset_str)
        elif answer == "unreviewed" and (onset_str or offset_str):
            raise ValueError("unreviewed rows cannot declare a boundary")
        checks.append({
            "right": key[0], "left": key[1], "source_oto_line": record["oto_line"],
            "answer": answer, "review_complete": verified,
            "human_boundary_interval_ms": interval,
            "reviewer": reviewer or None,
            "note": note,
            "automated_screening_status": record["screening"]["status"],
            "source_audio_sha256": record["audio_sha256"],
        })
    if set(expected) != seen:
        raise ValueError("review CSV omitted target rows")
    all_done = all(row["review_complete"] for row in checks)
    any_uncertain = any(row["answer"] != "yes" for row in checks)
    return {
        "schema": REVIEW_SCHEMA,
        "source_review_manifest_sha256": model["original_review_manifest_sha256"],
        "source_oto_sha256": model["original_source_oto_sha256"],
        "status": (
            "manual_boundary_review_complete_but_not_synthesis_validated"
            if all_done and not any_uncertain else
            "human_review_incomplete_or_uncertain_no_split_share"
        ),
        "reviewed_rows": sum(row["review_complete"] for row in checks),
        "target_rows": len(checks),
        "all_six_boundaries_marked": all_done and not any_uncertain,
        "rows": checks,
        "production_ready": False,
        "split_share_authorized": False,
        "independent_recording_takes_confirmed": False,
        "heldout_synthesis_gain_confirmed": False,
    }


def write_validation_report(pack: Path, annotations: Path, output: Path) -> dict:
    """Non-overwriting report, outside the original review pack."""
    report = validate_human_review(pack, annotations)
    root = Path(pack).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError("cannot modify source review pack")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return report
