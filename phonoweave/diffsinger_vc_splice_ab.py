"""Executable, fixed-source-duration VC->CV *micro-splice* auditory pilot.

Author-attested DiffSinger direct-rendered CVVC source ZIPs only. This is not
an OpenUtau/resampler render or independently held-out teacher-song A/B.
The 3 K=2 policies differ only in which of i/ir/i0 class donors are retained;
constant 120ms units, constant fixture CV and crossfade for all K=2 arms.
"""
from __future__ import annotations

import argparse
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

SCHEMA = "phonoweave.diffsinger_vc_microsplice_ab.v1"
LEFTS = ("i", "ir", "i0")
RIGHTS = ("m", "sh")
TARGET_SOURCE_WAV = {
    ("i", "m"): "vcsection2.wav", ("ir", "m"): "vcsection2.wav",
    ("i0", "m"): "vcsection2.wav", ("i", "sh"): "vcsection3.wav",
    ("ir", "sh"): "ingi0ir.wav", ("i0", "sh"): "vcsection3.wav",
}
POLICIES = {
    "k2_keep_i_ir": {"i": "i", "ir": "ir", "i0": "i"},
    "k2_keep_i_i0": {"i": "i", "ir": "i", "i0": "i0"},
    "k2_keep_ir_i0": {"i": "ir", "ir": "ir", "i0": "i0"},
    # Above K2 arms match budgets; K3 is a descriptive *higher-cost* ceiling.
    "k3_preserve_all_not_budget_matched": {"i": "i", "ir": "ir", "i0": "i0"},
}
UNIT_MS = 120
CV_MS = 200
FADE_MS = 12
PAD_MS = 40


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _parse_oto(raw: bytes, prefix: str) -> dict[tuple[str, str], list[dict]]:
    try:
        txt = raw.decode("utf-8-sig")
    except UnicodeError:
        txt = raw.decode("cp932")
    indexed: dict[tuple[str, str], list[dict]] = {}
    for no, line in enumerate(txt.splitlines(), 1):
        if not line.strip():
            continue
        try:
            wav, data = line.split("=", 1)
            alias, *values = data.rsplit(",", 5)
            if len(values) != 5:
                raise ValueError("five OTO numerical fields required")
            timing = [float(v) for v in values]
            if not all(math.isfinite(t) for t in timing):
                raise ValueError("nonfinite OTO field")
        except (ValueError, IndexError) as exc:
            raise ValueError(f"invalid OTO line {no}") from exc
        indexed.setdefault((wav, alias), []).append({
            "line": no, "wav": wav, "alias": alias, "time": timing,
        })
    return indexed


def _wav_pcm(z: zipfile.ZipFile, name: str) -> tuple[np.ndarray, int, str]:
    inf = z.getinfo(name)
    if inf.file_size > 200_000_000:
        raise ValueError("oversized source WAV")
    raw = z.read(name)
    try:
        with wave.open(io.BytesIO(raw), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getcomptype() != "NONE":
                raise ValueError("expected mono PCM16 WAV")
            rate = w.getframerate()
            if rate not in (44100, 48000):
                raise ValueError("expected 44100 or 48000 Hz")
            samples = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError("WAV could not be decoded") from exc
    return np.frombuffer(samples, dtype="<i2").copy(), rate, _sha(samples)


def _extract(samples: np.ndarray, rate: int, start_ms: float, duration_ms: float) -> np.ndarray:
    a = round(start_ms * rate / 1000)
    length = round(duration_ms * rate / 1000)
    if a < 0 or length <= 0 or a + length > len(samples):
        raise ValueError("extraction interval outside original WAV")
    return samples[a:a+length].astype(np.float64) / 32768.0


def _wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    if samples.ndim != 1 or not len(samples) or not np.all(np.isfinite(samples)):
        raise ValueError("invalid render waveform")
    if np.max(np.abs(samples)) > 1.0:
        raise ValueError("clipping: reject rather than normalize either arm")
    raw = np.round(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(raw)
    return output.getvalue()


def _crossfade(vc: np.ndarray, cv: np.ndarray, rate: int) -> np.ndarray:
    n = round(FADE_MS * rate / 1000)
    if len(vc) <= n or len(cv) <= n:
        raise ValueError("source units too short for crossfade")
    x = np.linspace(0.0, np.pi / 2, n, endpoint=True)
    transition = vc[-n:] * np.cos(x) + cv[:n] * np.sin(x)
    joined = np.concatenate((vc[:-n], transition, cv[n:]))
    margin = np.zeros(round(PAD_MS * rate / 1000), dtype=np.float64)
    return np.concatenate((margin, joined, margin))


def _bank_data(config: dict):
    archive = Path(config["archive"]).expanduser().resolve(strict=True)
    root, prefix = config["root"], config["prefix"]
    if not isinstance(root, str) or not root.endswith("/") or ".." in root.split("/") or root.startswith("/"):
        raise ValueError("invalid bank ZIP root")
    if not isinstance(prefix, str) or not prefix.isalnum():
        raise ValueError("invalid prefix")
    if not zipfile.is_zipfile(archive):
        raise ValueError("source is not a ZIP")
    with zipfile.ZipFile(archive) as z:
        if sum(info.file_size for info in z.infolist()) > 2_000_000_000:
            raise ValueError("source ZIP exceeds 2GB uncompressed limit")
        oto_name = root + "oto.ini"
        oto_raw = z.read(oto_name)
        oto = _parse_oto(oto_raw, prefix)
        wanted = set(TARGET_SOURCE_WAV.values()) | {"a.wav"}
        decoded = {name: _wav_pcm(z, root + name) for name in wanted}
        rates = {rate for _, rate, _ in decoded.values()}
        if len(rates) != 1:
            raise ValueError("inconsistent WAV sample rates within bank")
        rate = rates.pop()
        donors = {}
        units = {}
        for (left, right), wav in TARGET_SOURCE_WAV.items():
            alias = prefix + left + " " + right
            matches = oto.get((wav, alias), [])
            if len(matches) != 1:
                raise ValueError(f"missing/duplicate OTO donor: {alias} / {wav}")
            row = matches[0]
            off, _cons, _cutoff, pre, overlap = row["time"]
            # OTO-relative center is a provisional alignment, not a phone boundary.
            anchor = off + (pre + overlap) / 2
            pcm, _, pcm_hash = decoded[wav]
            unit = _extract(pcm, rate, anchor - 50.0, UNIT_MS)
            donors[(left, right)] = unit
            units[(left, right)] = {
                "source_wav": wav, "source_pcm_sha256": pcm_hash,
                "original_alias": alias, "oto_line": row["line"],
                "source_start_ms": round(anchor - 50, 3), "source_duration_ms": UNIT_MS,
                "oto_anchor_ms": round(anchor, 3),
                "phonetic_boundary_verified": False,
            }
        cvs, fixtures = {}, {}
        for right in RIGHTS:
            alias = prefix + right + "a"
            matches = oto.get(("a.wav", alias), [])
            if len(matches) != 1:
                raise ValueError(f"missing/duplicate fixed CV fixture: {alias}")
            row = matches[0]
            offset, cons, cut, pre, overlap = row["time"]
            anchor = offset + (pre + overlap) / 2
            pcm, _, pcm_hash = decoded["a.wav"]
            cvs[right] = _extract(pcm, rate, anchor, CV_MS)
            fixtures[right] = {"source_wav": "a.wav", "source_pcm_sha256": pcm_hash,
                               "alias": alias, "oto_line": row["line"],
                               "source_start_ms": round(anchor, 3), "source_duration_ms": CV_MS}
        return {
            "style": config["style"], "pitch": config["pitch"],
            "archive_name": archive.name, "archive_sha256": _sha(archive.read_bytes()),
            "oto_sha256": _sha(oto_raw), "rate": rate,
            "donors": donors, "donor_sources": units,
            "cv": cvs, "cv_sources": fixtures,
        }


def render_microsplice_ab(manifest: dict, output: Path) -> dict:
    """Render fully real PCM from original ZIPs, never modify those ZIPs.

    Note: all source units & CV fixtures are from TRAINING BANKS. There is no
    teacher/reference test set and no held-out sample included in this output.
    """
    if manifest.get("schema") != "phonoweave.diffsinger_vc_splice_input.v1":
        raise ValueError("unexpected source manifest schema")
    if manifest.get("source_provenance") != "author_confirmed_diffsinger_direct":
        raise ValueError("synthetic generated source provenance must be explicit")
    bank_specs = manifest.get("banks")
    if not isinstance(bank_specs, list) or not bank_specs:
        raise ValueError("at least one training subbank required")
    ids = [(v["style"], v["pitch"]) for v in bank_specs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate style/pitch bank")
    output = Path(output).expanduser().resolve()
    archives = [Path(x["archive"]).expanduser().resolve(strict=True) for x in bank_specs]
    if output.exists():
        raise FileExistsError("review package already exists")
    if any(output == p or output in p.parents for p in archives):
        raise ValueError("cannot render over source ZIP")
    banks = [_bank_data(b) for b in bank_specs]
    results = []
    payloads = {}
    for bank in banks:
        for policy_name, mapping in POLICIES.items():
            budget = 2 if policy_name.startswith("k2_") else 3
            actual_by_right = {right: sorted(set(mapping.values())) for right in RIGHTS}
            if any(len(x) != budget for x in actual_by_right.values()):
                raise ValueError("unmatched policy donor count")
            for right in RIGHTS:
                for target in LEFTS:
                    chosen = mapping[target]
                    audio = _crossfade(bank["donors"][(chosen, right)], bank["cv"][right], bank["rate"])
                    rel = (f"{bank['style']}_{bank['pitch']}/"
                           f"{policy_name}/{target}_{right}.wav")
                    encoded = _wav_bytes(audio, bank["rate"])
                    payloads[rel] = encoded
                    results.append({
                        "style": bank["style"], "pitch": bank["pitch"],
                        "policy": policy_name, "target_left_class": target,
                        "right_alias": right, "selected_donor_class": chosen,
                        "source_voicebank_sha256": bank["archive_sha256"],
                        "source_oto_sha256": bank["oto_sha256"],
                        "source_donor": bank["donor_sources"][(chosen,right)],
                        "shared_cv_fixture": bank["cv_sources"][right],
                        "left_connection_budget_units": budget,
                        "total_vc_budget_units_per_bank": 2 * budget,
                        "total_vc_budget_ms_per_bank": 2 * budget * UNIT_MS,
                        "common_cv_ms_per_bank": len(RIGHTS) * CV_MS,
                        "source_pitch_transposition": "none",
                        "rendered_wav": rel, "wav_sha256": _sha(encoded),
                        "render_sample_rate": bank["rate"],
                        "sound_quality_verified": False,
                        "test_sequence_heldout": False,
                    })
    data = {
        "schema": SCHEMA,
        "status": "actual_source_pcm_rendered_microsplice_only_not_singing_benchmark",
        "source_provenance": manifest["source_provenance"],
        "render_settings": {
            "vc_segment_ms": UNIT_MS, "cv_segment_ms": CV_MS,
            "crossfade_ms": FADE_MS, "silence_padding_ms": PAD_MS,
            "source_alignment": "OTO-derived provisional anchor; no independent target phone boundaries",
            "renderer": "fixed_equal_power_crossfade_numpy_no_pitch_transpose_no_resampler",
        },
        "matched_budget_policies": sorted(k for k in POLICIES if k.startswith("k2_")),
        "nonmatched_high_cost_reference": "k3_preserve_all_not_budget_matched",
        "results": results,
        "study_gates": {
            "actual_pcm_wav_rendered": True,
            "K2_policy_budgets_equal": True,
            "test_targets_unseen": False,
            "independent_diffsinger_teacher_reference": False,
            "actual_OpenUtau_resampler_verified": False,
            "blinded_listener_test": False,
            "evidence_of_optimal_split_share": False,
        },
        "limits": [
            "This pilot joins two existing source units, not a full note-conditioned singing phrase.",
            "The three K=2 arms are matched in unit count and raw donor milliseconds; the K=3 ceiling is not.",
            "The same fixed CV source is held across all arms within each style/pitch/right context.",
            "Audio is neither pitch-shifted nor time-stretched and no independently held-out target is scored.",
            "OTO anchors and fixed windows are not independently validated phoneme boundaries.",
            "These source WAVs are DiffSinger synthetic renders, not independent live singing takes.",
            "No raw source bank or source ZIP is uploaded to a public repository.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pw-splice-", dir=output.parent) as tmp:
        stage = Path(tmp) / "results"
        stage.mkdir()
        for rel, encoded in payloads.items():
            p = stage / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(encoded)
        (stage / "render_manifest.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        (stage / "README.txt").write_text(
            "Real source-PCM VC->CV micro-splice listening examples.\n"
            "Compare K=2 policies at the same fixed source-unit budget.\n"
            "K=3 author-preserved is a higher-cost illustration, NOT a budget-matched arm.\n"
            "No unseen DiffSinger target, no OpenUtau resampler, no claim of singing quality.\n",
            encoding="utf-8")
        if output.exists():
            raise FileExistsError("output appeared while processing")
        os.rename(stage, output)
    return data


def main(argv=None):
    p = argparse.ArgumentParser(description="Source-real, constant-budget CVVC microsplice A/B")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    data = json.loads(a.manifest.read_text(encoding="utf-8"))
    r = render_microsplice_ab(data, a.output)
    print(f"Actual source-PCM micro-splices: {len(r['results'])}; no heldout quality claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
