"""Non-destructive inspection pack for observed /a/-carrier frication triggers."""
from __future__ import annotations
import json
import wave
from pathlib import Path
import numpy as np
from .audio import read_wav
from .calibration_io import load_calibration_session
from .calibration_transition_onset import acoustic_tracks

TRIGGERS=((6,1,1612.0,1916.667),(26,0,992.0,1083.333))
NAME="calibration_carrier_trigger_review_v1.8.json"

def _write_wav(path: Path, samples: np.ndarray, sr: int)->None:
    pcm=(np.clip(samples,-1,1)*32767).astype("<i2")
    with wave.open(str(path),"wb") as fh:
        fh.setnchannels(1);fh.setsampwidth(2);fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())

def review_trigger_segments(session_dir: Path)->dict:
    session_dir=session_dir.expanduser().resolve()
    s=load_calibration_session(session_dir)
    out=session_dir/"analysis"/"carrier_trigger_review"
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index, token_index, estimated, cue in TRIGGERS:
        p=s["protocol"]["prompts"][index]
        tokens=p.get("tokens") or p["spoken_pattern"].split()
        if tokens[token_index] != "a":
            raise ValueError(f"protocol mismatch at {index}:{token_index}")
        x,sr=read_wav(session_dir/"recordings"/s["recordings"][index]["wav"])
        start=max(0.,estimated-450)
        end=min(1000.*len(x)/sr,estimated+550)
        lo=int(round(start*sr/1000))
        hi=int(round(end*sr/1000))
        name=f"{index+1:03d}_{p['base_unit']}_carrier_{token_index}_review.wav"
        _write_wav(out/name,x[lo:hi],sr)
        f=acoustic_tracks(x,sr)
        t=f["centers_ms"]
        region=(t>=estimated-120)&(t<=estimated+250)
        rows.append({
            "prompt_index":index,"wav":s["recordings"][index]["wav"],
            "spoken_pattern":" ".join(tokens),"carrier_token_index":token_index,
            "previous_token":tokens[token_index-1] if token_index>0 else None,
            "candidate_ms":estimated,"nominal_carrier_cue_ms":cue,
            "candidate_minus_carrier_cue_ms":round(estimated-cue,3),
            "clip_filename":name,"clip_start_absolute_ms":round(start,3),
            "candidate_within_clip_ms":round(estimated-start,3),
            "high_db_range_near_candidate":[round(float(np.min(f["high_dbfs"][region])),2),round(float(np.max(f["high_dbfs"][region])),2)],
            "high_fraction_db_range_near_candidate":[round(float(np.min(f["high_fraction_db"][region])),2),round(float(np.max(f["high_fraction_db"][region])),2)],
            "interpretation":"needs listening and spectrogram; not a verified false-positive",
        })
    result={"analysis":"carrier_trigger_review","version":"1.8-dev","session_id":session_dir.name,
        "rows":rows,"note":"Original clips are not modified. Exports aid manual inspection, not automatic truth labels."}
    dest=session_dir/"analysis"/NAME
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result
