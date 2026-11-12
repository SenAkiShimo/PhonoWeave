"""CV-first singer-specific accent discovery preview.

Input is solely original WAV + AUTHOR DECLARED (CV,C,V). Never open oto.ini,
never infer C/V from spelling, Mandarin phonotactics, or established CVVC
recording conventions. All differences are exploratory, NOT split decisions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import tempfile
import wave

import numpy as np

SCHEMA = "phonoweave.cv_first_manifest.v1"
RESULT_SCHEMA = "phonoweave.cv_first_accent_screen.v4.0"
BANDS = np.geomspace(150, 8000, 13)


def _token(x, name):
    if (not isinstance(x, str) or not x or x != x.strip()
            or len(x) > 120 or any(ord(char) < 32 for char in x)):
        raise ValueError(f"invalid author-declared {name}")
    return x


def read_manifest(path: Path) -> tuple[Path, list[dict]]:
    """Parse explicit acoustic/CV data: no OTO/legacy-alias fields permitted."""
    path = Path(path).expanduser().resolve(strict=True)
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict) or obj.get("schema") != SCHEMA:
        raise ValueError("requires phonoweave.cv_first_manifest.v1")
    allowed_top = {"schema", "recordings"}
    if set(obj) != allowed_top:
        raise ValueError("manifest accepts only schema and recordings; no OTO/alias inputs")
    data = obj["recordings"]
    if not isinstance(data, list) or not data:
        raise ValueError("at least one author-labeled CV WAV required")
    root = path.parent
    seen = set()
    cleaned = []
    for rec in data:
        fields = {"id", "cv", "consonant", "vowel", "wav"}
        if not isinstance(rec, dict) or set(rec) != fields:
            raise ValueError("each CV record needs id,cv,consonant,vowel,wav only")
        label = {field: _token(rec[field], field) for field in fields}
        if label["id"] in seen:
            raise ValueError("duplicate recording id")
        seen.add(label["id"])
        # Identity is not inferred from concatenating letters: "sh"+"i" is
        # NOT the source of truth for the author's written CV label.
        relative = Path(label["wav"])
        if (relative.is_absolute() or ".." in relative.parts or
                relative.suffix.lower() != ".wav"):
            raise ValueError("relative WAV filename required")
        wav = (root / relative).resolve()
        if not wav.is_relative_to(root):
            raise ValueError("WAV escapes manifest directory")
        if not wav.is_file():
            raise ValueError(f"missing CV WAV: {label['wav']}")
        label["path"] = wav
        cleaned.append(label)
    return root, cleaned


def _read_pcm(path: Path) -> tuple[np.ndarray, int, str]:
    digest = sha256()
    with path.open("rb") as inp:
        while chunk := inp.read(1024 * 1024):
            digest.update(chunk)
    try:
        with wave.open(str(path), "rb") as f:
            if f.getcomptype() != "NONE" or f.getnchannels() != 1 or f.getsampwidth() != 2:
                raise ValueError("requires unprocessed mono PCM16 WAV")
            sr = f.getframerate()
            if sr < 22050:
                raise ValueError("sample rate too low for 8kHz band features")
            frames = f.readframes(f.getnframes())
    except wave.Error as e:
        raise ValueError("cannot decode WAV") from e
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    if not (.12 <= len(audio) / sr <= 12):
        raise ValueError("CV WAV must be between 120ms and 12s")
    return audio, sr, digest.hexdigest()


def _logbands(x: np.ndarray, sr: int) -> np.ndarray:
    win = np.hanning(len(x))
    z = np.abs(np.fft.rfft((x - np.mean(x)) * win))**2
    freq = np.fft.rfftfreq(len(x), d=1/sr)
    return np.asarray([math.log10(float(np.mean(z[(freq >= a) & (freq < b)])) + 1e-10)
                       for a,b in zip(BANDS[:-1], BANDS[1:])])


def _audio_features(audio: np.ndarray, sr: int) -> dict:
    """Locate acoustic activity/transition without any known C-V timing.

    The transition candidate is a maximal spectral change, not a validated
    consonant/vowel boundary. Descriptive early and late portions are retained
    separately; users must not treat them as gold phones.
    """
    width = round(.025*sr)
    hop = round(.010*sr)
    if width < 128 or hop < 1:
        raise ValueError("invalid WAV sample rate")
    starts = np.arange(0, len(audio)-width+1, hop)
    if len(starts) < 12:
        raise ValueError("too few acoustic frames")
    rms = np.asarray([np.sqrt(np.mean(audio[i:i+width]**2)) for i in starts])
    peak = float(rms.max())
    if peak < .001:
        raise ValueError("mostly silent CV WAV")
    active = np.flatnonzero(rms >= max(.001, peak*.12))
    if len(active) < 10:
        raise ValueError("too few active acoustic frames")
    first, last = int(active[0]), int(active[-1])
    if last - first < 8:
        raise ValueError("insufficient active CV interval")
    spectrum = np.asarray([_logbands(audio[i:i+width], sr) for i in starts[first:last+1]])
    # Normalize each frame to remove gain, preserving spectral shape.
    spectrum -= spectrum.mean(axis=1, keepdims=True)
    length = len(spectrum)
    # Hypothesis-free spectral change candidate in the middle 20–75% of
    # active sound, WITHOUT treating it as ground-truth C/V boundary.
    lo = max(2, int(length*.20))
    hi = min(length-2, int(length*.75))
    if hi <= lo:
        raise ValueError("unable to locate an acoustic transition candidate")
    change = np.linalg.norm(np.diff(spectrum, axis=0),axis=1)
    candidate = lo + int(np.argmax(change[lo:hi]))
    # Fixed temporally separated excerpts remain available even if transition
    # is misleading (e.g. a stop burst or diphthong), never called phones.
    early = spectrum[int(length*.12):max(int(length*.12)+1, int(length*.35))]
    late = spectrum[int(length*.65):max(int(length*.65)+1,int(length*.88))]
    if len(early) < 2 or len(late) < 2:
        raise ValueError("insufficient windows for CV profile")
    early_vec = early.mean(axis=0)
    late_vec = late.mean(axis=0)
    return {
        "early_profile": early_vec.tolist(),
        "late_profile": late_vec.tolist(),
        "acoustic_transition_candidate_ms": round(
            float((starts[first+candidate]+width/2) / sr * 1000),2),
        "transition_is_validated_CV_boundary": False,
        "active_interval_ms": [
            round(float(starts[first] / sr * 1000),2),
            round(float((starts[last]+width) / sr * 1000),2)],
    }


def analyze(manifest: Path) -> dict:
    """Compare same author C across V, and same author V across C.

    Report only source-independent WAV-level exploratory distances. Never read
    or reproduce any old split labels, filename-derived phoneme IDs or OTO.
    """
    root, records = read_manifest(manifest)
    observations=[]
    observed_digest=set()
    for rec in records:
        audio, sr, file_digest = _read_pcm(rec["path"])
        pcm_digest = sha256(audio.tobytes()).hexdigest()
        if pcm_digest in observed_digest:
            raise ValueError("duplicated decoded WAV samples; not independent CV observations")
        observed_digest.add(pcm_digest)
        descriptor = _audio_features(audio,sr)
        observations.append({
            "id":rec["id"],"cv":rec["cv"],
            "consonant":rec["consonant"],"vowel":rec["vowel"],
            "source_wav":rec["wav"],
            "file_sha256":file_digest, "pcm_sha256":pcm_digest,
            **descriptor
        })

    results=[]
    # For same consonant compare the EARLY (not necessarily consonant) profile
    # across different author-specified vowels. Reverse for same vowel.
    for axis, fixed_field, varied_field, feature in (
        ("same_C_across_V", "consonant", "vowel", "early_profile"),
        ("same_V_across_C", "vowel", "consonant", "late_profile"),
    ):
        by_fixed=defaultdict(lambda:defaultdict(list))
        for obs in observations:
            by_fixed[obs[fixed_field]][obs[varied_field]].append(obs)
        for fixed in sorted(by_fixed):
            labels=sorted(by_fixed[fixed])
            for idx, a in enumerate(labels):
                for b in labels[idx+1:]:
                    left=by_fixed[fixed][a];right=by_fixed[fixed][b]
                    # No pseudo-replicate statistics from a one-shot CV.
                    eligible=min(len(left),len(right)) >= 2
                    # Descriptive observed-centroid distance, not split merit.
                    delta = None
                    if eligible:
                        center_a=np.mean([o[feature] for o in left],axis=0)
                        center_b=np.mean([o[feature] for o in right],axis=0)
                        delta=round(float(np.sqrt(np.mean((center_a-center_b)**2))),8)
                    results.append({
                        "comparison": axis,"same_author_symbol":fixed,
                        "varied_author_symbols":[a,b],
                        "CV_labels":[sorted({o["cv"] for o in left}),
                                     sorted({o["cv"] for o in right})],
                        "independent_WAV_count":[len(left),len(right)],
                        "early_or_late_profile":"early" if axis.startswith("same_C") else "late",
                        "descriptive_profile_distance":delta,
                        "status":"acoustic_screen_only" if eligible else "insufficient_independent_CV_recordings",
                        "candidate_new_split":"UNRESOLVED",
                        "existing_OTO_used":False,
                    })
    return {
        "schema":RESULT_SCHEMA,
        "source_schema":SCHEMA,
        "input_mode":"author_declared_CV_C_V_plus_raw_PCM_only",
        "recording_count":len(observations),
        "CV_observations":observations,
        "cross_context_comparisons":results,
        "candidate_count":len(results),
        "new_split_labels":[],
        "automatic_reclist":[],
        "limitations":[
            "CV letters are author tokens; no IPA/phonology is inferred.",
            "Fixed-time early/late sound regions and spectral-change peaks do not verify C/V boundaries.",
            "Differences can be normal coarticulation, pitch, energy, vowel leakage or style, not accent-only allophony.",
            "At least two distinct WAVs per cell permit description but not statistical independence/session validation.",
            "No synthesis relevance, matched-budget savings or splitting decision has been demonstrated.",
        ],
    }


def export(report: dict, manifest:Path, output:Path):
    root=Path(manifest).expanduser().resolve(strict=True).parent
    output=Path(output).expanduser().resolve()
    if output.is_relative_to(root):
        raise ValueError("output must be outside CV source directory")
    if output.exists():
        raise FileExistsError("output already exists")
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phonoweave-cv-", dir=output.parent) as tmp:
        stage=Path(tmp)/"results"
        stage.mkdir()
        (stage/"cv_acoustic_discovery.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
            encoding="utf-8")
        cols=["comparison","same_author_symbol","varied_a","varied_b",
              "count_a","count_b","descriptive_profile_distance","status"]
        with (stage/"cv_comparison.csv").open("x",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(cols)
            for r in report["cross_context_comparisons"]:
                # Do not enable formulas in CSV when an author label begins =,+,-,@
                value=[r["comparison"],r["same_author_symbol"],
                       *r["varied_author_symbols"],*r["independent_WAV_count"],
                       r["descriptive_profile_distance"],r["status"]]
                w.writerow([("'"+v) if isinstance(v,str) and v.lstrip().startswith(("=","+","-","@"))
                            else v for v in value])
        os.rename(stage,output)


def main(argv=None):
    p=argparse.ArgumentParser(description="CV-first acoustic analysis; strictly NO OTO inputs")
    p.add_argument("manifest",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    try:
        result=analyze(args.manifest)
        export(result,args.manifest,args.output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        p.error(str(e))
    print(json.dumps({
        "CV_recordings":result["recording_count"],
        "comparison_pairs":result["candidate_count"],
        "automatic_split_labels":0,
        "output":str(args.output),
    },ensure_ascii=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
