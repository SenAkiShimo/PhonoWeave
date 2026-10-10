"""Paired CV-consistency robustness, independent of detection threshold tuning."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from .audio import read_wav
from .calibration_io import load_calibration_session
from .calibration_dual_labels import load_dual_labels
from .calibration_robustness import SCENARIOS, perturb
from .calibration_sibilant_onset import SIBILANTS
from .calibration_transition_onset import acoustic_tracks, nominal_target_cue_ms
from .calibration_cv_transition import cv_tracks
from .calibration_cv_consistency import classify_syllable
from .calibration_negative_controls import CORPUS_DIR, CATEGORIES

def evaluate_paired_noise(session_dir: Path) -> dict:
    session_dir=session_dir.expanduser().resolve()
    session=load_calibration_session(session_dir)
    manual=load_dual_labels(session_dir)["labels"]
    root=session_dir.parent
    rows=[]; negatives=[]; errors=[]
    for index, rec in sorted(session["recordings"].items()):
        p=session["protocol"]["prompts"][index]
        if p.get("base_unit") not in SIBILANTS:continue
        tokens=p.get("tokens") or str(p.get("spoken_pattern","")).split()
        positions=[i for i,t in enumerate(tokens) if t==p.get("syllable")]
        if len(positions)!=2:continue
        x,sr=read_wav(session_dir/"recordings"/rec["wav"])
        for j,scenario in enumerate(SCENARIOS):
            altered=perturb(np.asarray(x,dtype=float),scenario,np.random.default_rng(8500+index*11+j))
            a,b=acoustic_tracks(altered,sr),cv_tracks(altered,sr)
            for occurrence, token_idx in enumerate(positions,1):
                mark=manual.get(str(index),{}).get("occurrences",{}).get(str(occurrence),{}).get("consonant_onset")
                gold=float(mark["absolute_anchor_ms"]) if isinstance(mark,dict) and mark.get("status")=="ok" and not mark.get("outside_expected_slot",False) else None
                r=classify_syllable(a,b,nominal_target_cue_ms(token_idx))
                rows.append({"scenario":scenario,"prompt_index":index,"occurrence":occurrence,
                    "status":r.status,"gold_ms":gold,
                    "onset_ms":r.consonant_onset_ms,"cv_ms":r.cv_boundary_ms,
                    "signed_error_ms":round(r.consonant_onset_ms-gold,3) if gold is not None and r.status=="paired_candidate" else None})
    folder=root/CORPUS_DIR
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            try:
                m=json.loads(path.read_text(encoding="utf-8"))
                if m.get("schema")!="negative_control_v1":continue
                if m.get("category") not in CATEGORIES:raise ValueError("unsupported category")
                wav=(folder/str(m["wav"])).resolve()
                if wav.parent!=folder:raise ValueError("out-of-corpus WAV path")
                x,sr=read_wav(wav)
                for j,scenario in enumerate(SCENARIOS):
                    y=perturb(np.asarray(x,dtype=float),scenario,np.random.default_rng(9200+j*111+len(negatives)))
                    a,b=acoustic_tracks(y,sr),cv_tracks(y,sr)
                    r=classify_syllable(a,b,1250.)
                    negatives.append({"category":m["category"],"id":m["id"],"scenario":scenario,"status":r.status,
                        "onset_trigger":r.consonant_onset_ms is not None,"paired_trigger":r.status=="paired_candidate"})
            except (OSError,ValueError,KeyError,TypeError) as exc:
                errors.append({"filename":path.name,"reason":str(exc)})
    summary={}
    for s in SCENARIOS:
        pos=[r for r in rows if r["scenario"]==s]
        valid=[r for r in pos if r["gold_ms"] is not None]
        matched=[r for r in valid if r["signed_error_ms"] is not None]
        err=np.asarray([r["signed_error_ms"] for r in matched],dtype=float)
        neg=[r for r in negatives if r["scenario"]==s]
        summary[s]={"positive_total":len(pos),"positive_paired":sum(r["status"]=="paired_candidate" for r in pos),
            "positive_eligible":len(valid),"positive_matched":len(matched),
            "positive_mae_ms":round(float(np.mean(np.abs(err))),3) if len(err) else None,
            "negative_total":len(neg),"negative_paired":sum(r["paired_trigger"] for r in neg),
            "negative_onset_only":sum(r["onset_trigger"] for r in neg)}
    report={"analysis":"paired_cv_noise_robustness","version":"2.2-development","session_id":session_dir.name,
            "summary":summary,"positives":rows,"negatives":negatives,"failures":errors,
            "limitations":"Development session and self-reported negatives; all negative windows centered at 1250ms. Synthetic white noise and amplitude changes are not independent real environments; matched-only MAE must be considered with positive coverage. Paired-candidate counts do not constitute validated specificity."}
    path=session_dir/"analysis"/"calibration_paired_noise_v2.2.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report
