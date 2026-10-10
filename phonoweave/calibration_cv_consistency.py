"""Experimental CV-consistency gate for target fricative syllables.

Distinguishes an onset-only acoustic event from an onset followed by a
vowel-like transition. The gate is intentionally NOT applied to isolated
consonants or breath/aspiration study tasks.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
from collections import Counter

from .audio import read_wav
from .calibration_cv_transition import cv_tracks, detect_cv_boundary
from .calibration_frication_shoulder import detect_frication_shoulder
from .calibration_transition_onset import acoustic_tracks, nominal_target_cue_ms
from .calibration_io import load_calibration_session
from .calibration_dual_labels import load_dual_labels
from .calibration_sibilant_onset import SIBILANTS
from .calibration_negative_controls import CORPUS_DIR, CATEGORIES

@dataclass(frozen=True)
class SyllableCandidate:
    consonant_onset_ms: float | None
    cv_boundary_ms: float | None
    status: str
    reason: str


def classify_syllable(onset_tracks, vowel_tracks, cue_ms: float) -> SyllableCandidate:
    onset = detect_frication_shoulder(onset_tracks, cue_ms)
    if onset.shoulder_onset_ms is None:
        return SyllableCandidate(None,None,"unresolved",onset.reason)
    cv = detect_cv_boundary(vowel_tracks, onset.core_onset_ms)
    if cv.absolute_ms is None:
        return SyllableCandidate(onset.shoulder_onset_ms,None,"review","no_cv_confirmation:"+cv.reason)
    if not onset.shoulder_onset_ms < cv.absolute_ms:
        return SyllableCandidate(onset.shoulder_onset_ms,cv.absolute_ms,"review","invalid_boundary_order")
    return SyllableCandidate(onset.shoulder_onset_ms,cv.absolute_ms,"paired_candidate","frication_followed_by_vowel_transition")


def evaluate_cv_consistency(session_dir: Path, corpus_root: Path | None = None) -> dict:
    session_dir=session_dir.expanduser().resolve()
    root=(corpus_root or session_dir.parent).expanduser().resolve()
    session=load_calibration_session(session_dir)
    annotations=load_dual_labels(session_dir)["labels"]
    positives=[]
    for index,rec in sorted(session["recordings"].items()):
        prompt=session["protocol"]["prompts"][index]
        if prompt.get("base_unit") not in SIBILANTS:
            continue
        tokens=prompt.get("tokens") or str(prompt.get("spoken_pattern","")).split()
        positions=[i for i,t in enumerate(tokens) if t==prompt.get("syllable")]
        if len(positions)!=2: continue
        x,sr=read_wav(session_dir/"recordings"/rec["wav"])
        f1=acoustic_tracks(x,sr)
        f2=cv_tracks(x,sr)
        for occurrence,token_index in enumerate(positions,1):
            guess=classify_syllable(f1,f2,nominal_target_cue_ms(token_index))
            mark=annotations.get(str(index),{}).get("occurrences",{}).get(str(occurrence),{}).get("consonant_onset")
            eligible=isinstance(mark,dict) and mark.get("status")=="ok" and not mark.get("outside_expected_slot",False)
            manual=float(mark["absolute_anchor_ms"]) if eligible else None
            positives.append({"prompt_index":index,"base_unit":prompt["base_unit"],"occurrence":occurrence,
                **asdict(guess),"manual_onset_ms":manual,
                "signed_error_ms":None if manual is None or guess.status!="paired_candidate" else round(guess.consonant_onset_ms-manual,3)})
    negatives=[]
    failures=[]
    folder=root/CORPUS_DIR
    if folder.is_dir():
        for p in sorted(folder.glob("*.json")):
            try:
                m=json.loads(p.read_text(encoding="utf-8"))
                if m.get("schema")!="negative_control_v1":continue
                if m.get("category") not in CATEGORIES:raise ValueError("invalid category")
                wav=(folder/m["wav"]).resolve()
                if wav.parent!=folder:raise ValueError("recording path outside corpus")
                x,sr=read_wav(wav)
                f1,f2=acoustic_tracks(x,sr),cv_tracks(x,sr)
                # Same single standardized window as v1.9 for paired comparison.
                result=classify_syllable(f1,f2,1250.0)
                negatives.append({"id":m["id"],"category":m["category"],**asdict(result)})
            except (OSError,ValueError,KeyError,TypeError) as exc:
                failures.append({"metadata":p.name,"reason":str(exc)})
    eligible=[r for r in positives if r["manual_onset_ms"] is not None]
    matched=[r for r in eligible if r["signed_error_ms"] is not None]
    import numpy as np
    errors=np.array([r["signed_error_ms"] for r in matched],dtype=float)
    result={"analysis":"cv_consistency_screen","version":"2.1-development",
        "session_id":session_dir.name,
        "summary":{
            "positive_occurrences":len(positives),"eligible_positive":len(eligible),
            "positive_paired":sum(r["status"]=="paired_candidate" for r in positives),
            "positive_gold_matched":len(matched),
            "positive_coverage":round(len(matched)/len(eligible),4) if eligible else None,
            "positive_mae_ms":round(float(np.mean(np.abs(errors))),3) if len(errors) else None,
            "negative_recordings":len(negatives),
            "negative_onset_only":sum(r["consonant_onset_ms"] is not None for r in negatives),
            "negative_paired":sum(r["status"]=="paired_candidate" for r in negatives),
            "negative_paired_by_category":dict(Counter(r["category"] for r in negatives if r["status"]=="paired_candidate")),
            "failures":len(failures),
        },
        "positives":positives,"negatives":negatives,"failures":failures,
        "limitations":"Development-only, same-session positive audio. Negative labels self-reported. CV consistency applies only to elicited consonant-vowel syllables; unvoiced fricatives in isolation should not be rejected for lack of vowel. Missing CV confirmation is review, not proof of non-speech. The fixed 1.25s negative comparison does not scan full recordings."}
    out=session_dir/"analysis"/"calibration_cv_consistency_v2.1.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result
