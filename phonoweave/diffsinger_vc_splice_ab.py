"""Real PCM, budget-matched DiffSinger CVVC *micro-splice* research pilot.

This produces audible two-unit splices, not OpenUtau singing synthesis or
evidence of split/share efficacy. Donor sourcing stays within one voice style
and one pitch bank. Fully preserved K3 is explicitly NOT budget matched.
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
SOURCE = {
    ("i", "m"): "vcsection2.wav", ("ir", "m"): "vcsection2.wav",
    ("i0", "m"): "vcsection2.wav", ("i", "sh"): "vcsection3.wav",
    ("ir", "sh"): "ingi0ir.wav", ("i0", "sh"): "vcsection3.wav",
}
POLICIES = {
    "k2_keep_i_ir": {"i": "i", "ir": "ir", "i0": "i"},
    "k2_keep_i_i0": {"i": "i", "ir": "i", "i0": "i0"},
    "k2_keep_ir_i0": {"i": "ir", "ir": "ir", "i0": "i0"},
    "k3_preserve_all_not_budget_matched": {"i": "i", "ir": "ir", "i0": "i0"},
}
UNIT_MS = 120
CV_MS = 200
FADE_MS = 12
PAD_MS = 40


def _sha(b: bytes) -> str:
    return sha256(b).hexdigest()


def _oto(raw: bytes, prefix: str) -> dict[tuple[str, str], list[dict]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError:
        text = raw.decode("cp932")
    result = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            wav, tail = line.split("=", 1)
            alias, *fields = tail.rsplit(",", 5)
            values = [float(x) for x in fields]
            if len(values) != 5 or any(not math.isfinite(v) for v in values):
                raise ValueError("nonfinite or absent timings")
        except (ValueError, IndexError) as err:
            raise ValueError(f"malformed OTO line {lineno}") from err
        result.setdefault((wav, alias), []).append(
            {"wav": wav, "alias": alias, "line": lineno, "values": values}
        )
    return result


def _read_wav(z: zipfile.ZipFile, name: str) -> tuple[np.ndarray, int, str]:
    if z.getinfo(name).file_size > 200_000_000:
        raise ValueError("oversized source WAV")
    with wave.open(io.BytesIO(z.read(name)), "rb") as wav:
        if (wav.getcomptype() != "NONE" or wav.getnchannels() != 1
                or wav.getsampwidth() != 2 or wav.getframerate() not in (44100, 48000)):
            raise ValueError("only mono PCM16 44.1/48kHz supported")
        rate = wav.getframerate()
        pcm = wav.readframes(wav.getnframes())
    return np.frombuffer(pcm, dtype="<i2"), rate, _sha(pcm)


def _slice(pcm: np.ndarray, rate: int, at_ms: float, ms: int) -> np.ndarray:
    begin = round(at_ms * rate / 1000)
    length = round(ms * rate / 1000)
    if begin < 0 or begin + length > len(pcm) or length < 1:
        raise ValueError("source segment outside WAV")
    return pcm[begin:begin + length].astype(np.float64) / 32768.0


def _pcm_wav(signal: np.ndarray, rate: int) -> bytes:
    if not len(signal) or not np.isfinite(signal).all() or np.max(np.abs(signal)) > 1.0:
        raise ValueError("invalid or clipped render; never normalize an arm separately")
    data = np.round(signal * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(data)
    return buffer.getvalue()


def _splice(vc: np.ndarray, cv: np.ndarray, rate: int) -> np.ndarray:
    n = round(FADE_MS * rate / 1000)
    if len(vc) <= n or len(cv) <= n:
        raise ValueError("clip too short to join")
    phases = np.linspace(0.0, np.pi / 2, n)
    mixed = vc[-n:] * np.cos(phases) + cv[:n] * np.sin(phases)
    pad = np.zeros(round(PAD_MS * rate / 1000))
    return np.concatenate((pad, vc[:-n], mixed, cv[n:], pad))


def _source_bank(spec: dict) -> dict:
    archive = Path(spec["archive"]).expanduser().resolve(strict=True)
    root, prefix = spec["root"], spec["prefix"]
    if (not isinstance(root, str) or not root.endswith("/")
            or root.startswith("/") or ".." in root.split("/")
            or not isinstance(prefix, str) or not prefix.isalnum()):
        raise ValueError("noncanonical archive root/prefix")
    with zipfile.ZipFile(archive) as z:
        if sum(info.file_size for info in z.infolist()) > 2_000_000_000:
            raise ValueError("uncompressed archive size exceeds 2GB")
        oto_raw = z.read(root + "oto.ini")
        entries = _oto(oto_raw, prefix)
        source_names = set(SOURCE.values()) | {"a.wav"}
        audio = {wav: _read_wav(z, root + wav) for wav in source_names}
        rates = {t[1] for t in audio.values()}
        if len(rates) != 1:
            raise ValueError("inconsistent source rates")
        rate = rates.pop()
        donors, details = {}, {}
        for (left, right), src in SOURCE.items():
            alias = prefix + left + " " + right
            matched = entries.get((src, alias), [])
            if len(matched) != 1:
                raise ValueError(f"missing/duplicate OTO: {src} {alias}")
            rec = matched[0]
            offset, cons, cutoff, pre, overlap = rec["values"]
            anchor = offset + (pre + overlap) / 2
            pcm, _, pcm_sha = audio[src]
            donors[(left, right)] = _slice(pcm, rate, anchor - 50, UNIT_MS)
            details[(left, right)] = {
                "source_wav": src, "original_alias": alias,
                "oto_line": rec["line"], "source_pcm_sha256": pcm_sha,
                "source_start_ms": round(anchor - 50, 3),
                "source_duration_ms": UNIT_MS,
                "phonetic_boundary_verified": False,
            }
        fixtures, fixture_details = {}, {}
        for right in RIGHTS:
            alias = prefix + right + "a"
            matched = entries.get(("a.wav", alias), [])
            if len(matched) != 1:
                raise ValueError(f"missing/duplicate fixed CV fixture {alias}")
            rec = matched[0]
            offset, cons, cutoff, pre, overlap = rec["values"]
            anchor = offset + (pre + overlap) / 2
            pcm, _, pcm_sha = audio["a.wav"]
            fixtures[right] = _slice(pcm, rate, anchor, CV_MS)
            fixture_details[right] = {
                "source_wav": "a.wav", "original_alias": alias,
                "oto_line": rec["line"], "source_pcm_sha256": pcm_sha,
                "source_start_ms": round(anchor, 3),
                "source_duration_ms": CV_MS,
            }
    return {
        "style": spec["style"], "pitch": spec["pitch"], "rate": rate,
        "archive_basename": archive.name,
        "archive_sha256": _sha(archive.read_bytes()),
        "oto_sha256": _sha(oto_raw),
        "donors": donors, "sources": details,
        "cv": fixtures, "cv_sources": fixture_details,
    }


def render_microsplice_ab(config: dict, output: Path) -> dict:
    if config.get("schema") != "phonoweave.diffsinger_vc_splice_input.v1":
        raise ValueError("source manifest schema mismatch")
    if config.get("source_provenance") != "author_confirmed_diffsinger_direct":
        raise ValueError("DiffSinger source provenance must be author-confirmed")
    specs = config.get("banks")
    if not isinstance(specs, list) or not specs:
        raise ValueError("at least one source bank required")
    identities = [(s["style"], s["pitch"]) for s in specs]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate style/pitch")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("non-overwriting output exists")
    banks = [_source_bank(s) for s in specs]
    files, rows = {}, []
    for bank in banks:
        for policy, lookup in POLICIES.items():
            k = 2 if policy.startswith("k2_") else 3
            if len(set(lookup.values())) != k:
                raise ValueError("invalid frozen donor budget")
            for right in RIGHTS:
                for left in LEFTS:
                    chosen = lookup[left]
                    wav = _pcm_wav(_splice(bank["donors"][(chosen, right)],
                                          bank["cv"][right], bank["rate"]), bank["rate"])
                    rel = f"{bank['style']}_{bank['pitch']}/{policy}/{left}_{right}.wav"
                    files[rel] = wav
                    rows.append({
                        "style": bank["style"], "pitch": bank["pitch"], "policy": policy,
                        "target_left_class": left, "right_alias": right,
                        "selected_donor_class": chosen,
                        "source_voicebank_sha256": bank["archive_sha256"],
                        "source_oto_sha256": bank["oto_sha256"],
                        "source_donor": bank["sources"][(chosen, right)],
                        "shared_cv_fixture": bank["cv_sources"][right],
                        "left_connection_budget_units": k,
                        "total_vc_budget_units_per_bank": len(RIGHTS) * k,
                        "total_vc_budget_ms_per_bank": len(RIGHTS) * k * UNIT_MS,
                        "common_cv_ms_per_bank": len(RIGHTS) * CV_MS,
                        "rendered_wav": rel, "wav_sha256": _sha(wav),
                        "render_sample_rate": bank["rate"],
                        "phonetic_boundary_verified": False,
                        "heldout_teacher_evaluated": False,
                    })
    report = {
        "schema": SCHEMA, "source_provenance": config["source_provenance"],
        "status": "actual_PCM_microsplice_only_not_full_singing_synthesis",
        "render_settings": {
            "vc_segment_ms": UNIT_MS, "cv_segment_ms": CV_MS,
            "crossfade_ms": FADE_MS, "padding_ms": PAD_MS,
            "renderer": "same_fixed_equal_power_crossfade_numpy_no_pitch_or_duration_render",
            "units": "fixed_provisional_OTO_relative_windows",
        },
        "matched_budget_policies": sorted(p for p in POLICIES if p.startswith("k2_")),
        "nonmatched_higher_budget_reference": "k3_preserve_all_not_budget_matched",
        "results": rows,
        "study_gates": {
            "actual_pcm_wav_rendered": True,
            "K2_policy_budgets_equal": True,
            "independent_diffsinger_teacher_reference": False,
            "test_targets_unseen": False,
            "actual_OpenUtau_resampler_verified": False,
            "blinded_listener_test": False,
            "evidence_of_optimal_split_share": False,
        },
        "limitations": [
            "A genuine 2-unit PCM splice, not note-time/pitch-conditioned OpenUtau singing.",
            "K2 variants have equal VC source-unit count and milliseconds; K3 does not.",
            "All arms use identical post-VC CV fixture within same style/pitch/right context.",
            "No held-out teacher utterance or test sequence is included or scored.",
            "Transferred unverified OTO anchor does not establish phonetic boundaries.",
            "The WAVs are author-confirmed DiffSinger renders, not independent live singing.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pw-ds-vc-", dir=output.parent) as tmp:
        stage = Path(tmp) / "render"
        stage.mkdir()
        for rel, wav in files.items():
            dest = stage / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(wav)
        (stage / "render_manifest.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")
        if output.exists():
            raise FileExistsError("output appeared during render")
        os.rename(stage, output)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave diffsinger-vc-microsplice-ab")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = render_microsplice_ab(config, args.output)
    print(f"Real PCM micro-splice WAVs: {len(report['results'])}; not a singing-quality benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
