"""Source-audited, *non-inferential* VC transition review from an archived UTAU bank.

Produces context audio and time-resolved QC around the existing OTO timing.
It neither discovers true phoneme boundaries nor decides a split/merge.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import tempfile
import wave
import zipfile

import numpy as np

SCHEMA = "phonoweave.vc_boundary_feasibility_gate.v0.1"
PROFILE_SCHEMA = "phonoweave.author_confirmed_vc_nucleus_labels.v0.1"
SHIFTS_MS = tuple(range(-160, 161, 20))
CLIP_BEFORE_MS = 240
CLIP_AFTER_MS = 360
FRAME_MS = 16


def digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _validate_archive(zipfile_obj: zipfile.ZipFile, profile: dict) -> tuple[str, list[dict], dict]:
    if profile.get("schema") != PROFILE_SCHEMA or profile.get("review_scope") != "nucleus_label_semantics_only":
        raise ValueError("missing compatible author-confirmed nucleus profile")
    labels = profile.get("labels")
    if not isinstance(labels, list) or len(labels) != 3:
        raise ValueError("expected exactly three confirmed nucleus labels")
    left_labels = {}
    for x in labels:
        if not isinstance(x, dict) or x.get("author_confirmed") is not True:
            raise ValueError("unconfirmed nucleus identity")
        k, v = x.get("alias_left_surface"), x.get("nucleus_class_id")
        if not k or not v or k in left_labels:
            raise ValueError("invalid or duplicate confirmed nucleus identity")
        left_labels[k] = v
    infos = zipfile_obj.infolist()
    if len({i.filename for i in infos}) != len(infos):
        raise ValueError("duplicate ZIP members")
    if sum(i.file_size for i in infos) > 2_000_000_000:
        raise ValueError("archive too large for safe review")
    oto_names = [i.filename for i in infos if i.filename == "oto.ini" or i.filename.endswith("/oto.ini")]
    if len(oto_names) != 1:
        raise ValueError("ZIP must contain exactly one oto.ini")
    oto_name = oto_names[0]
    raw = zipfile_obj.read(oto_name)
    if digest(raw) != profile.get("source_oto_sha256"):
        raise ValueError("source OTO hash does not match confirmed labels")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp932")
    prefix = profile.get("alias_prefix")
    if not isinstance(prefix, str) or not prefix or any(c.isspace() for c in prefix):
        raise ValueError("invalid alias prefix")
    root = oto_name[:-len("oto.ini")]
    permitted = {i.filename[len(root):]: i for i in infos
                 if i.filename.startswith(root) and i.filename.endswith(".wav")
                 and "/" not in i.filename[len(root):]}
    entries = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line or "=" not in line:
            continue
        wav, fields = line.split("=", 1)
        parts = fields.rsplit(",", 5)
        if len(parts) != 6:
            continue
        alias = parts[0]
        if not alias.startswith(prefix):
            continue
        surface = alias[len(prefix):].split()
        if len(surface) != 2 or surface[0] not in left_labels:
            continue
        try:
            offset, consonant, cutoff, pre, overlap = map(float, parts[1:])
        except ValueError:
            continue
        if not all(map(math.isfinite, (offset, consonant, cutoff, pre, overlap))):
            continue
        if wav not in permitted:
            raise ValueError(f"referenced WAV unavailable: {wav}")
        entries.append({
            "oto_line": line_no, "alias": alias, "source_wav": wav,
            "left": surface[0], "right": surface[1],
            "nucleus_class_id": left_labels[surface[0]],
            "offset_ms": offset, "preutterance_ms": pre,
            "overlap_ms": overlap, "consonant_ms": consonant,
            "cutoff_ms": cutoff,
            "oto_window_start_ms": offset + overlap,
            "oto_window_end_ms": offset + pre,
            "anchor_ms": offset + (overlap + pre) / 2,
        })
    return root, entries, permitted


def _read_mono_pcm16(raw: bytes) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(raw), "rb") as stream:
            if (stream.getnchannels() != 1 or stream.getsampwidth() != 2
                    or stream.getcomptype() != "NONE"):
                raise ValueError("source must be mono uncompressed 16-bit PCM WAV")
            rate = stream.getframerate()
            frames = stream.getnframes()
            data = stream.readframes(frames)
    except (EOFError, wave.Error, OSError) as exc:
        raise ValueError("invalid WAV") from exc
    if not 8000 <= rate <= 96000 or len(data) != frames * 2:
        raise ValueError("invalid PCM source frames or sample rate")
    return np.frombuffer(data, dtype="<i2"), rate


def _signal_frame(signal: np.ndarray, rate: int, center_ms: float) -> tuple[float, float] | None:
    n = round(rate * FRAME_MS / 1000)
    center = round(center_ms * rate / 1000)
    lo = center - n // 2
    if lo < 0 or lo + n > len(signal):
        return None
    x = signal[lo:lo + n].astype(np.float64) / 32768.0
    rms = math.sqrt(float(np.mean(x*x)))
    y = (x - np.mean(x)) * np.hanning(len(x))
    spec = abs(np.fft.rfft(y))**2
    freq = np.fft.rfftfreq(len(x), 1 / rate)
    mask = (freq >= 150) & (freq <= min(8000, rate/2))
    power = float(np.sum(spec[mask]))
    centroid = float(np.sum(spec[mask] * freq[mask]) / power) if power > 1e-18 else None
    return rms, centroid


def _write_wav(dst: Path, segment: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(np.asarray(segment, dtype="<i2").tobytes())
    content = buf.getvalue()
    dst.write_bytes(content)
    return content


def _selection(rows: list[dict], right: str, labels: list[str]) -> tuple[list[dict], str | None]:
    opts = [r for r in rows if r["right"] == right and r["left"] in labels]
    group = defaultdict(list)
    for r in opts:
        group[r["source_wav"]].append(r)
    if not opts or not set(labels).issubset({r["left"] for r in opts}):
        raise ValueError(f"not all confirmed nuclei have alias for right label {right!r}")
    best_wav = min(group, key=lambda w: (-len({r["left"] for r in group[w]}), w))
    matched = all(any(r["left"] == label for r in group[best_wav]) for label in labels)
    chosen = []
    for label in labels:
        same = [r for r in group[best_wav] if r["left"] == label]
        all_sources = [r for r in opts if r["left"] == label]
        chosen.append(min(same or all_sources, key=lambda r: r["oto_line"]))
    return chosen, (best_wav if matched else None)


def _plot(path: Path, right: str, trajectories: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("--plots requires optional matplotlib") from exc
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    for label in sorted({r["left"] for r in trajectories}):
        part = sorted((r for r in trajectories if r["left"] == label), key=lambda r: r["relative_ms"])
        x = [r["relative_ms"] for r in part]
        axs[0].plot(x, [r["rms_dbfs"] for r in part], marker=".", label=label)
        axs[1].plot(x, [r["spectral_centroid_hz"] for r in part], marker=".", label=label)
    for ax in axs:
        ax.axvline(0, color="gray", linestyle="--", alpha=.6)
        ax.grid(alpha=.2)
        ax.legend(loc="best")
    axs[0].set_ylabel("RMS (dBFS)")
    axs[1].set_ylabel("Spectral centroid (Hz)")
    axs[1].set_xlabel("Relative to OTO window midpoint (ms)")
    fig.suptitle(f"Right-hand alias {right}: time-course QC (OTO anchor, not phone boundary)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compile_boundary_review(archive: Path, labels_path: Path, output: Path,
                            *, right_labels: tuple[str,...]=( "m", "sh"), plots: bool=False) -> dict:
    """Immutable, source-hashed clips plus honest feasibility diagnostics."""
    archive = Path(archive).expanduser().resolve(strict=True)
    labels_path = Path(labels_path).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("review output exists; will not overwrite")
    if output == archive or output == labels_path:
        raise ValueError("output cannot overwrite input")
    if not right_labels or len(right_labels) != len(set(right_labels)):
        raise ValueError("right-hand labels must be distinct and nonempty")
    profile = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = [r["alias_left_surface"] for r in profile.get("labels", [])]
    report = {
        "schema": SCHEMA,
        "status": "auditory_boundary_review_required_no_split_share_evidence",
        "source": {"archive_name": archive.name,
                   "archive_sha256": sha256(archive.read_bytes()).hexdigest(),
                   "oto_sha256": profile.get("source_oto_sha256")},
        "nucleus_classes": {r["alias_left_surface"]: r["nucleus_class_id"]
                            for r in profile.get("labels", [])},
        "selected_right_aliases": list(right_labels),
        "review_groups": [],
        "research_gate": {
            "independently_recorded_repeats_per_condition_confirmed": False,
            "boundary_location_human_verified": False,
            "actual_synthesis_ab_validated": False,
            "can_claim_split_share_benefit": False,
            "next_required_action": "review excerpt audio and mark transitions; then acquire independent takes",
        },
        "production_alias_export": "disabled", "oto_timing_changes": "none",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(
            prefix=".pw-vc-boundary-", dir=output.parent) as work:
        root, entries, allowed = _validate_archive(z, profile)
        data_cache = {}
        stage = Path(work) / "review"
        stage.mkdir()
        for right in right_labels:
            selected, common_wav = _selection(entries, right, labels)
            group_dir = stage / f"right_{right}"
            group_dir.mkdir()
            clips = []
            trajectories = []
            for row in selected:
                wav_name = row["source_wav"]
                if wav_name not in data_cache:
                    member = allowed[wav_name]
                    if member.file_size > 500_000_000:
                        raise ValueError("oversize source WAV")
                    raw = z.read(root + wav_name)
                    data_cache[wav_name] = (*_read_mono_pcm16(raw), digest(raw))
                samples, rate, wav_sha = data_cache[wav_name]
                center = row["anchor_ms"]
                lo = round((center - CLIP_BEFORE_MS) * rate / 1000)
                hi = round((center + CLIP_AFTER_MS) * rate / 1000)
                if lo < 0 or hi > len(samples) or lo >= hi:
                    raise ValueError("candidate cannot be reviewed with full context clip")
                clip_name = f"{row['left']}.wav"
                content = _write_wav(group_dir / clip_name, samples[lo:hi], rate)
                clips.append({
                    "left_surface": row["left"],
                    "nucleus_class_id": row["nucleus_class_id"],
                    "source_wav": wav_name, "source_wav_sha256": wav_sha,
                    "oto_line": row["oto_line"], "original_alias": row["alias"],
                    "oto_window_start_ms": round(row["oto_window_start_ms"], 3),
                    "oto_window_end_ms": round(row["oto_window_end_ms"], 3),
                    "oto_midpoint_ms": round(center, 3),
                    "excerpt_filename": f"right_{right}/{clip_name}",
                    "excerpt_sha256": digest(content),
                    "excerpt_oto_anchor_ms": CLIP_BEFORE_MS,
                    "excerpt_length_ms": (hi - lo) * 1000 / rate,
                    "boundary_verified": False,
                    "right_phone_identity_verified": False,
                })
                for shift in SHIFTS_MS:
                    value = _signal_frame(samples, rate, center + shift)
                    if value is None:
                        raise ValueError("timecourse extends outside source WAV")
                    rms, centroid = value
                    trajectories.append({
                        "left": row["left"], "right": right, "oto_line": row["oto_line"],
                        "relative_ms": shift,
                        "rms_dbfs": round(20 * math.log10(max(rms, 1e-10)), 3),
                        "spectral_centroid_hz": round(centroid, 1) if centroid is not None else "",
                    })
            with (group_dir / "timecourse.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(trajectories[0]))
                writer.writeheader()
                writer.writerows(trajectories)
            if plots:
                _plot(group_dir / "qc_trajectory.png", right, trajectories)
            qc = []
            for left in labels:
                series = [r for r in trajectories if r["left"] == left]
                at_zero = next(r for r in series if r["relative_ms"] == 0)
                after = [r for r in series if 0 < r["relative_ms"] <= 120
                         and r["spectral_centroid_hz"] != ""
                         and r["spectral_centroid_hz"] > 2000]
                qc.append({
                    "left_surface": left,
                    "oto_anchor_spectral_centroid_hz": at_zero["spectral_centroid_hz"],
                    "first_post_anchor_centroid_above_2khz_ms": (
                        after[0]["relative_ms"] if after else None),
                    "informal_alignment_warning": (
                        "large_high_frequency_shift_after_oto_anchor"
                        if (after and at_zero["spectral_centroid_hz"] != ""
                            and at_zero["spectral_centroid_hz"] < 1000)
                        else "no_specific_high_frequency_shift_detected"),
                })
            report["review_groups"].append({
                "right_surface": right, "same_wav_for_three_nuclei": common_wav is not None,
                "common_source_wav": common_wav,
                "different_source_WAVs_confounded": common_wav is None,
                "centroid_timecourse_descriptive_qc_not_boundaries": qc,
                "clip_records": clips,
                "independent_takes_per_class": "not_established_by_one_OTO_row",
                "decision_status": "human_boundary_review_required",
            })
        (stage / "review_manifest.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")
        with (stage / "review_annotations.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["right_surface","left_surface","original_alias","source_wav",
                             "oto_line","boundary_onset_ms_in_excerpt","boundary_offset_ms_in_excerpt",
                             "target_sound_present","listening_comments","reviewer"])
            for group in report["review_groups"]:
                for item in group["clip_records"]:
                    writer.writerow([group["right_surface"],item["left_surface"],
                        item["original_alias"],item["source_wav"],item["oto_line"],
                        "","","unreviewed","",""])
        (stage / "README.md").write_text(
            "# VC boundary feasibility — human review pack\n\n"
            "Listen to each 600ms excerpt, identify whether the expected nucleus "
            "and following onset are audible, and annotate the ACTUAL transition "
            "boundaries in `review_annotations.csv`.\n\n"
            "**Anchor**: existing OTO overlap/preutterance *midpoint* at 240ms "
            "within each excerpt. It is NOT a verified VC boundary.\n\n"
            "`timecourse.csv` shows 16ms spectral/RMS snapshots at -160..+160ms "
            "around that anchor. The curves cannot prove split/share efficacy.\n\n"
            "`m` uses three excerpts from one WAV, but not independent repeats. "
            "`sh` has cross-WAV confounding. There is no holdout or synthesis AB.\n",
            encoding="utf-8")
        if output.exists():
            raise FileExistsError("review output appeared during staging")
        os.rename(stage, output)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="phonoweave vc-boundary-gate")
    p.add_argument("archive", type=Path)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--right", action="append", help="following surface label; default m and sh")
    p.add_argument("--plots", action="store_true", help="requires optional matplotlib")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        report = compile_boundary_review(args.archive, args.labels, args.output,
                    right_labels=tuple(args.right or ("m", "sh")), plots=args.plots)
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
        p.error(str(exc))
    print("Review groups:", len(report["review_groups"]))
    print("Research decision: NOT READY for split/share efficacy claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
