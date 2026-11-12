"""v0.9: held-out acoustic proxy for the frozen 36-take VC pilot.

This is *not* a rendered-singing evaluation or an automatic split/share tool.
Only S01 supplies prototype estimates; S02 is held out for prediction-error
comparison. Both conditions use the exact same S01 recordings.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import wave

import numpy as np

from .vc36_capture import audit_manifest, _secure_slot, _digest, _shape_errors

SCHEMA = "phonoweave.vc36.heldout_acoustic_proxy.v0.9"
LEFT = ("i", "ir", "i0")
RIGHT = ("m", "sh")
EDGES = np.geomspace(150.0, 8000.0, 13)
TIMEPOINTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _features(path: Path, a_ms: float, b_ms: float) -> np.ndarray:
    """Time-normalized 5*12 log-spectral profile of the *reviewed VC interval*.

    These are not isolated-phoneme measures and may include both the preceding
    vowel and following consonant. No whole-utterance onset cue is analyzed.
    """
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("expected mono PCM16 source")
        rate = wav.getframerate()
        signal = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    if not all(math.isfinite(float(t)) for t in (a_ms, b_ms)):
        raise ValueError("non-finite manual VC interval")
    if b_ms-a_ms < 40:
        raise ValueError("manual VC interval below 40ms, insufficient for stable sampling")
    frame = round(rate*0.016)
    if frame < 2:
        raise ValueError("invalid sampling rate")
    centers = np.linspace(a_ms+8, b_ms-8, len(TIMEPOINTS))
    freq = np.fft.rfftfreq(frame, 1.0/rate)
    win = np.hanning(frame)
    profiles = []
    for center_ms in centers:
        start = int(round(center_ms*rate/1000.0))-frame//2
        if start < 0 or start+frame > len(signal):
            raise ValueError("VC spectral frame extends beyond WAV")
        sample = signal[start:start+frame]
        if float(np.sqrt(np.mean(sample*sample))) < .0001:
            raise ValueError("silent/near-silent annotated VC interval")
        power = np.abs(np.fft.rfft(win*(sample-sample.mean())))**2
        bands = [math.log10(float(power[(freq>=lo)&(freq<hi)].mean())+1e-9)
                 for lo, hi in zip(EDGES[:-1], EDGES[1:])]
        row = np.asarray(bands, dtype=np.float64)
        profiles.append(row-row.mean())
    out = np.stack(profiles)
    if out.shape != (5, 12) or not np.isfinite(out).all():
        raise ValueError("invalid spectral profile")
    return out


def _mse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x-y)**2))


def evaluate(manifest_path: Path) -> dict:
    """Evaluate two S01-only estimators on distinct S02 takes. Fail closed."""
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    root = manifest_path.parent
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = audit_manifest(data, root)
    if not audit.get("analysis_ready"):
        raise ValueError("VC36 manifest not analysis_ready: capture, QC and manual annotations required")
    if _shape_errors(data):
        raise ValueError("frozen VC36 manifest changed")

    profiles = {}
    for r in data["recordings"]:
        key = (r["session_id"], r["left_label"], r["right_label"], r["repeat"])
        if key in profiles:
            raise ValueError("duplicate VC36 recording slot")
        source = _secure_slot(root, r["relative_wav"])
        profiles[key] = {
            "features": _features(source, r["transition_onset_ms"], r["transition_offset_ms"]),
            "take_id": r["take_id"], "source_wav_sha256": r["wav_sha256"],
        }

    expected = {(sid,left,right,rep) for sid in ("S01", "S02")
                for left in LEFT for right in RIGHT for rep in (1,2,3)}
    if set(profiles) != expected:
        raise ValueError("missing or mismatched session/context/repetition slots")

    # The pooled baseline is genuinely estimated over all nine S01 takes
    # for each RIGHT, not a post-hoc donor swap or class-label substitution.
    separate = {(left,right):np.mean([profiles[("S01",left,right,r)]["features"]
                                    for r in (1,2,3)],axis=0)
                for left in LEFT for right in RIGHT}
    pooled = {right:np.mean([profiles[("S01",left,right,r)]["features"]
                            for left in LEFT for r in (1,2,3)],axis=0)
              for right in RIGHT}

    rows = []
    for right in RIGHT:
        for left in LEFT:
            for rep in (1,2,3):
                item = profiles[("S02",left,right,rep)]
                x = item["features"]
                err_sep = _mse(x,separate[(left,right)])
                err_pool = _mse(x,pooled[right])
                rows.append({
                    "take_id":item["take_id"], "left_surface":left, "right_surface":right,
                    "repeat":rep, "source_wav_sha256":item["source_wav_sha256"],
                    "heldout_separate_mse":round(err_sep,8),
                    "heldout_pooled_mse":round(err_pool,8),
                    "pooled_minus_separate_mse":round(err_pool-err_sep,8),
                })

    strata = []
    for right in RIGHT:
        for left in LEFT:
            s=[r for r in rows if r["right_surface"]==right and r["left_surface"]==left]
            strata.append({"left_surface":left, "right_surface":right,
                           "heldout_takes":len(s),
                           "mean_pooled_minus_separate_mse":round(float(np.mean(
                               [x["pooled_minus_separate_mse"] for x in s])),8),
                           "separate_closer_count":sum(x["pooled_minus_separate_mse"]>0 for x in s)})
    delta=float(np.mean([r["pooled_minus_separate_mse"] for r in rows]))
    # Real-singer repeated samples are still correlated; no p-value or
    # universal split threshold is assigned to this small exploratory pilot.
    return {
        "schema":SCHEMA, "status":"heldout_acoustic_proxy_only_no_split_share_approval",
        "source_manifest_sha256":_digest(manifest_path),
        "protocol_sha256":data["protocol_sha256"],
        "speaker_id":data["speaker_id"],
        "train_session":"S01", "heldout_session":"S02",
        "training_unique_wavs":18, "heldout_unique_wavs":18,
        "same_training_material_both_estimators":True,
        "method":{
            "region":"human_reviewed_vowel_to_next_consonant_transition_not_isolated_phone",
            "windows_ms":16, "timepoints":5,
            "spectral_band_edges_hz":[round(float(x),3) for x in EDGES],
            "features":"5x12 time-resolved mean-centered log10 band power",
            "separate_estimator":"per-left-and-right mean of 3 S01 takes",
            "pooled_estimator":"per-right mean of all 9 S01 takes",
            "loss":"per-coefficient squared error on S02 manual transition",
            "delta_sign":"positive = separate acoustic prototype closer",
        },
        "per_take":rows, "per_condition":strata,
        "mean_pooled_minus_separate_mse":round(delta,8),
        "separate_closer_heldout_takes":sum(r["pooled_minus_separate_mse"]>0 for r in rows),
        "research_decision":{
            "acoustic_direction":("separate_prototype_lower_heldout_mse" if delta>0 else
                                   "pooled_prototype_lower_heldout_mse" if delta<0 else "tied"),
            "automatic_split_or_merge_approved":False,
            "real_synthesis_gain_established":False,
            "recording_list_effect_established":False,
            "remaining_requirement":"matched-budget real singing synthesis/rendering plus blind listening",
        },
    }


def export_evaluation(result: dict, dest: Path, *, manifest_path:Path) -> Path:
    dest=Path(dest).expanduser().resolve()
    root=Path(manifest_path).expanduser().resolve(strict=True).parent
    if dest==root or dest.is_relative_to(root):
        raise ValueError("output must be outside the recording corpus")
    if dest.exists():
        raise FileExistsError("evaluation output exists")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phonoweave-vc36-eval-",dir=dest.parent) as tmp:
        stage=Path(tmp)/"report"
        stage.mkdir()
        (stage/"heldout_acoustic_proxy.json").write_text(json.dumps(
            result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        with (stage/"heldout_per_take.csv").open("x",newline="",encoding="utf-8") as f:
            cols=tuple(result["per_take"][0])
            writer=csv.DictWriter(f,fieldnames=cols)
            writer.writeheader()
            # take ids are generated from the fixed protocol; no dynamic formulas.
            writer.writerows(result["per_take"])
        os.rename(stage,dest)
    return dest


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description="held-out VC36 acoustic proxy (NOT split decision)")
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    try:
        result=evaluate(args.manifest)
        export_evaluation(result,args.output,manifest_path=args.manifest)
    except (ValueError,OSError,KeyError,TypeError,json.JSONDecodeError,wave.Error) as e:
        p.error(str(e))
    print(json.dumps({"status":result["status"],
                     "mean_pooled_minus_separate_mse":result["mean_pooled_minus_separate_mse"],
                     "automatic_split_or_merge_approved":False},ensure_ascii=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
