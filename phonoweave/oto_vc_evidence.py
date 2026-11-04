"""Read-only UTAU OTO VC surface evidence from an archived voicebank.

An OTO interval is a *provisional alignment region*, not a confirmed phone
boundary. This module deliberately does not infer IPA from a singer's aliases.
Original WAV/OTO/aliases are never edited and cannot be exported for production.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import wave
import zipfile

import numpy as np

SCHEMA = "phonoweave.oto_surface_vc_probe.v0.1"
REVIEW_SCHEMA = "phonoweave.oto_vc_manual_identity_review.v0.1"
LINK_SCHEMA = "phonoweave.oto_to_phonetic_graph_links.v0.1"
GRAPH_SCHEMA = "phonoweave.phonetic_transition_budget.v0.2"
FREQUENCY_EDGES = np.geomspace(150.0, 8000.0, 13)


def _digest_bytes(content: bytes) -> str:
    return sha256(content).hexdigest()


def _digest_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _wav_samples(content: bytes) -> tuple[np.ndarray, int, float]:
    try:
        with wave.open(io.BytesIO(content), "rb") as stream:
            if (stream.getcomptype() != "NONE"
                    or stream.getnchannels() != 1
                    or stream.getsampwidth() != 2
                    or stream.getframerate() < 8000):
                raise ValueError("voicebank probe needs mono PCM16 WAV")
            rate = stream.getframerate()
            frames = stream.getnframes()
            raw = stream.readframes(frames)
    except (wave.Error, EOFError, OSError) as exc:
        raise ValueError("invalid WAV in voicebank archive") from exc
    if len(raw) != frames * 2:
        raise ValueError("truncated WAV data")
    return np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0, rate, frames * 1000.0 / rate


def _spectral_window(samples: np.ndarray, rate: int,
                     start_ms: float, end_ms: float) -> dict | None:
    """Fixed, 16ms centered OTO-relative fingerprint, never a phone estimate."""
    if end_ms - start_ms < 16:
        return None
    n = round(rate * 0.016)
    at = round((start_ms + end_ms) * rate / 2000) - n // 2
    if at < 0 or at + n > len(samples):
        return None
    x = samples[at:at + n].astype(float)
    rms = float(np.sqrt(np.mean(x * x)))
    x = (x - np.mean(x)) * np.hanning(n)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / rate)
    audible = (freqs >= 150) & (freqs <= 8000)
    denominator = float(np.sum(power[audible]))
    centroid = float(np.sum(freqs[audible] * power[audible])
                     / max(denominator, 1e-25))
    bands = []
    for low, high in zip(FREQUENCY_EDGES[:-1], FREQUENCY_EDGES[1:]):
        subset = power[(freqs >= low) & (freqs < high)]
        bands.append(float(np.log10(np.mean(subset) + 1e-10)))
    mean = float(np.mean(bands))
    return {
        "rms": round(rms, 7),
        "spectral_centroid_hz": round(centroid, 1),
        "normalized_12_log_power": [round(v - mean, 5) for v in bands],
    }


def analyze_archived_voicebank(archive: Path, *, prefix: str) -> dict:
    """Parse OTO and measure eligible fixed windows without extracting ZIP."""
    if not isinstance(prefix, str) or not prefix or any(c.isspace() for c in prefix):
        raise ValueError("a nonblank fixed alias prefix is required")
    archive = archive.expanduser().resolve(strict=True)
    if not zipfile.is_zipfile(archive):
        raise ValueError("input is not a ZIP archive")
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if sum(i.file_size for i in infos) > 2_000_000_000:
            raise ValueError("archive uncompressed size exceeds safety limit")
        names = {i.filename: i for i in infos}
        oto_paths = [name for name in names if name.endswith("/oto.ini") or name == "oto.ini"]
        if len(oto_paths) != 1:
            raise ValueError("archive must contain exactly one oto.ini")
        oto_path = oto_paths[0]
        root = oto_path[:-len("oto.ini")]
        oto_raw = z.read(oto_path)
        try:
            decoded = oto_raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            decoded = oto_raw.decode("cp932")
        wavs = {Path(name).name: name for name in names
                if name.startswith(root) and name.endswith(".wav")
                and Path(name).name == name[len(root):]}
        alias_counts: Counter[str] = Counter()
        rows = []
        parse_errors = []
        alias_pattern = re.compile(re.escape(prefix) + r"([^\s]+)\s+([^\s]+)\Z")
        for line_no, raw in enumerate(decoded.splitlines(), 1):
            if not raw.strip():
                continue
            try:
                source_wav, tail = raw.split("=", 1)
                alias, *numbers = tail.rsplit(",", 5)
                if len(numbers) != 5:
                    raise ValueError("OTO entry needs 5 numeric fields")
                timings = [float(x) for x in numbers]
                if any(not math.isfinite(v) for v in timings):
                    raise ValueError("nonfinite OTO timing")
                alias_counts[alias] += 1
                match = alias_pattern.fullmatch(alias)
                if not match or match.group(1) == "-":
                    continue
                if source_wav not in wavs:
                    raise ValueError("missing or noncanonical source WAV")
                offset, consonant, cutoff, pre, overlap = timings
                rows.append({
                    "oto_line": line_no, "alias": alias, "wav": source_wav,
                    "left_surface": match.group(1), "right_surface": match.group(2),
                    "offset_ms": round(offset, 3),
                    "consonant_ms": round(consonant, 3),
                    "cutoff_ms": round(cutoff, 3),
                    "preutterance_ms": round(pre, 3),
                    "overlap_ms": round(overlap, 3),
                    "candidate_window_start_ms": round(offset + overlap, 3),
                    "candidate_window_end_ms": round(offset + pre, 3),
                    "candidate_window_ms": round(pre - overlap, 3),
                    "spectral_status": "unassessed",
                })
            except (ValueError, IndexError) as exc:
                parse_errors.append({"oto_line": line_no, "reason": str(exc)})
        if parse_errors:
            raise ValueError(f"malformed OTO entries: {parse_errors[:5]}")
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["wav"]].append(row)
        wav_info = {}
        for wav, items in sorted(grouped.items()):
            info = names[wavs[wav]]
            if info.file_size > 500_000_000:
                raise ValueError("oversized WAV in voicebank ZIP")
            samples, rate, duration = _wav_samples(z.read(wavs[wav]))
            wav_info[wav] = {"sample_rate": rate, "duration_ms": round(duration, 2)}
            for item in items:
                start = item["candidate_window_start_ms"]
                end = item["candidate_window_end_ms"]
                offset = item["offset_ms"]
                cutoff = item["cutoff_ms"]
                # UTAU positive cutoff trims from file end; negative cutoff
                # denotes a duration relative to offset. No resynthesis here.
                trimmed_end = duration - cutoff if cutoff >= 0 else offset - cutoff
                if not (0 <= offset < duration and
                        0 <= start <= end <= trimmed_end <= duration + 1):
                    item["spectral_status"] = "invalid_oto_window"
                    continue
                feature = _spectral_window(samples, rate, start, end)
                if feature is None:
                    item["spectral_status"] = "too_short_for_fixed_16ms_proxy"
                else:
                    item.update(feature)
                    item["spectral_status"] = (
                        "provisional_oto_relative_only" if feature["rms"] >= 1e-4
                        else "very_low_energy_proxy_unreliable"
                    )
    count = Counter(row["spectral_status"] for row in rows)
    return {
        "schema": SCHEMA,
        "status": "surface_alias_oto_proxies_not_phoneme_boundaries",
        "source": {
            "archive_name": archive.name, "archive_sha256": _digest_file(archive),
            "oto_sha256": _digest_bytes(oto_raw),
            "oto_archive_path": oto_path, "prefix": prefix,
        },
        "summary": {
            "oto_records_total": sum(alias_counts.values()),
            "unique_oto_alias_strings": len(alias_counts),
            "duplicate_alias_names": sum(v > 1 for v in alias_counts.values()),
            "extra_records_on_duplicate_aliases": sum(v - 1 for v in alias_counts.values() if v > 1),
            "vc_shape_records": len(rows),
            "vc_shape_distinct_aliases": len({r["alias"] for r in rows}),
            "wav_total": len(wavs), "wav_used_by_vc": len(wav_info),
            "spectral_status": dict(sorted(count.items())),
        },
        "wav_info": wav_info,
        "left_surface_counts": dict(sorted(Counter(r["left_surface"] for r in rows).items())),
        "candidates": rows,
        "limitations": [
            "VC syntax is inferred from source alias spelling, not recognized phones or IPA.",
            "The OTO overlap-to-preutterance window is not a validated transition boundary.",
            "Repeated OTO aliases do not imply independent recording repetitions.",
            "Normalized log-power fingerprints are not proof of a useful split, share, or merge.",
            "No production aliases, OTO timings or source WAVs are modified.",
        ],
        "production_alias_export": "disabled",
        "oto_timing_changes": "none",
    }


def map_reviewed_graph_edges(oto: dict, graph: dict, reviews: dict) -> dict:
    """Join explicitly reviewed VC alias identities to symbolic graph edges.

    Requires affirmative *identity* review; OTO audio boundary must still
    be reviewed. Unmapped identities are not guessed or declared absent.
    """
    if oto.get("schema") != SCHEMA or graph.get("schema") != GRAPH_SCHEMA:
        raise ValueError("incompatible OTO / phonetic graph evidence")
    if reviews.get("schema") != REVIEW_SCHEMA:
        raise ValueError("invalid identity review")
    edges = {}
    source_features = (
        graph["phonetic_policy"]["uncovered_features"] +
        [feat for row in graph["phonetic_policy"]["selected_lines"]
         for feat in row["new_features"]]
    )
    for feature in source_features:
        if feature.get("kind") != "transition":
            continue
        key = (feature["phase"], feature["pattern"],
               feature["left"], feature["right"])
        edges[key] = feature
    by_alias = defaultdict(list)
    for item in oto["candidates"]:
        by_alias[item["alias"]].append(item)
    linked = []
    used_aliases = set()
    for review in reviews.get("links", []):
        if not isinstance(review, dict) or review.get("identity_reviewed") is not True:
            raise ValueError("every requested link must be explicitly identity-reviewed")
        alias = review.get("alias")
        edge = (review.get("phase"), review.get("pattern"),
                review.get("left_phone"), review.get("right_phone"))
        if (not isinstance(alias, str) or alias in used_aliases
                or alias not in by_alias or edge not in edges):
            raise ValueError("duplicate/unmatched alias or nonexisting graph transition")
        used_aliases.add(alias)
        records = by_alias[alias]
        if (review.get("left_surface") != records[0]["left_surface"]
                or review.get("right_surface") != records[0]["right_surface"]):
            raise ValueError("identity review conflicts with observed source alias")
        linked.append({
            "alias": alias,
            "phonetic_transition": edges[edge],
            "source_oto_lines": [r["oto_line"] for r in records],
            "source_wavs": sorted({r["wav"] for r in records}),
            "spectral_windows_with_provisional_features": sum(
                r["spectral_status"] == "provisional_oto_relative_only" for r in records),
            "status": "candidate_link_identity_reviewed_boundary_and_synthesis_unverified",
        })
    return {
        "schema": LINK_SCHEMA,
        "status": "review_only_no_synthesis_validation",
        "source_oto_sha256": oto["source"]["oto_sha256"],
        "source_graph_lexicon_sha256": graph["source"]["lexicon_sha256"],
        "matched_links": linked,
        "unmapped_distinct_aliases": len(set(by_alias) - used_aliases),
        "production_ready": False,
        "recordings_authorized": 0,
    }


def write_audit(report: dict, output: Path) -> Path:
    """Publish an audit folder without overwriting source or previous output."""
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "oto_line", "wav", "alias", "left_surface", "right_surface",
        "offset_ms", "consonant_ms", "cutoff_ms", "preutterance_ms",
        "overlap_ms", "candidate_window_start_ms", "candidate_window_end_ms",
        "candidate_window_ms", "spectral_status", "rms", "spectral_centroid_hz",
    ]
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".pw-oto-vc-") as temp:
        stage = Path(temp) / "audit"
        stage.mkdir()
        (stage / "vc_oto_evidence.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        with (stage / "vc_review_queue.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in report["candidates"]:
                writer.writerow({key: item.get(key, "") for key in fieldnames})
        with (stage / "identity_mapping_template.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow([
                "alias", "left_surface", "right_surface", "oto_count",
                "left_phone_review_required", "right_phone_review_required",
                "phase", "pattern", "identity_reviewed", "boundary_reviewed",
            ])
            counter = Counter(item["alias"] for item in report["candidates"])
            by_alias = {item["alias"]: item for item in report["candidates"]}
            for alias, count in sorted(counter.items()):
                item = by_alias[alias]
                writer.writerow([
                    alias, item["left_surface"], item["right_surface"], count,
                    "", "", "", "", "false", "false",
                ])
        if output.exists():
            raise FileExistsError("output appeared while writing audit")
        os.rename(stage, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave audit-oto-vc-zip")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--prefix", required=True, help="literal alias prefix, e.g. D4")
    parser.add_argument("--output", required=True, type=Path,
                        help="new non-overwriting output directory")
    args = parser.parse_args(argv)
    try:
        report = analyze_archived_voicebank(args.archive, prefix=args.prefix)
        write_audit(report, args.output)
    except (OSError, ValueError, UnicodeError, KeyError,
            zipfile.BadZipFile) as error:
        parser.error(str(error))
    print("VC-form OTO rows:", report["summary"]["vc_shape_records"])
    print("Spectral status:", report["summary"]["spectral_status"])
    print("Manual review required before linking phonetic identities or splitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
