"""Exploratory noise-floor contrast diagnostics without changing detection."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .audio import read_wav
from .calibration_io import load_calibration_session
from .calibration_robustness import SCENARIOS, perturb
from .calibration_transition_onset import (
    acoustic_tracks, nominal_target_cue_ms, LEFT_MS, RIGHT_MS,
    MIN_HIGH_RISE_DB,
)
from .calibration_cv_transition import cv_tracks
from .calibration_cv_consistency import classify_syllable
from .calibration_sibilant_onset import SIBILANTS


def power_excess_db(high_db: np.ndarray, baseline_db: float) -> np.ndarray:
    """Energy above estimated local noise floor, clamped at zero."""
    high_power=np.power(10.,np.asarray(high_db,dtype=float)/10.)
    floor_power=10.**(float(baseline_db)/10.)
    excess=np.maximum(high_power-floor_power,0.)
    return 10.*np.log10(excess+1.e-18)


def inspect_contrast(tracks: dict, cue: float) -> dict:
    t=np.asarray(tracks["centers_ms"])
    high=np.asarray(tracks["high_dbfs"])
    indices=np.flatnonzero((t>=cue-LEFT_MS)&(t<=cue+RIGHT_MS))
    if len(indices)<50:raise ValueError("insufficient analysis window")
    high_segment=high[indices]
    baseline=float(np.percentile(high_segment[:45],20))
    excess=power_excess_db(high_segment,baseline)
    return {
        "baseline_dbfs":round(baseline,3),
        "high_rise_passing_frames":int(np.sum(high_segment>=baseline+MIN_HIGH_RISE_DB)),
        "high_rise_max_db":round(float(np.max(high_segment-baseline)),3),
        "excess_energy_p50_dbfs":round(float(np.median(excess)),3),
        "excess_energy_p90_dbfs":round(float(np.percentile(excess,90)),3),
        "high_fraction_p90_db":round(float(np.percentile(
            np.asarray(tracks["high_fraction_db"])[indices],90)),3),
    }


def run_contrast_audit(session_dir: Path) -> dict:
    session_dir=session_dir.expanduser().resolve()
    session=load_calibration_session(session_dir)
    rows=[];failures=[]
    for index,rec in sorted(session["recordings"].items()):
        p=session["protocol"]["prompts"][index]
        if p.get("base_unit") not in SIBILANTS:continue
        tokens=p.get("tokens") or str(p.get("spoken_pattern","")).split()
        positions=[i for i,token in enumerate(tokens) if token==p.get("syllable")]
        if len(positions)!=2:continue
        try:
            x,sr=read_wav(session_dir/"recordings"/rec["wav"])
            for j,scenario in enumerate(SCENARIOS):
                y=perturb(np.asarray(x,dtype=float),scenario,
                          np.random.default_rng(8500+index*11+j))
                tracks=acoustic_tracks(y,sr)
                vowel=cv_tracks(y,sr)
                for occurrence,token_index in enumerate(positions,1):
                    cue=nominal_target_cue_ms(token_index)
                    result=classify_syllable(tracks,vowel,cue)
                    rows.append({
                        "prompt_index":index,"base_unit":p["base_unit"],
                        "context":p.get("context_family"),"occurrence":occurrence,
                        "scenario":scenario,"paired_status":result.status,
                        "contrast":inspect_contrast(tracks,cue)
                    })
        except (ValueError,OSError,KeyError,TypeError) as exc:
            failures.append({"prompt_index":index,"reason":str(exc)})
    summary={}
    for scenario in SCENARIOS:
        group=[r for r in rows if r["scenario"]==scenario]
        paired=[r for r in group if r["paired_status"]=="paired_candidate"]
        missing=[r for r in group if r["paired_status"]!="paired_candidate"]
        summary[scenario]={
            "targets":len(group),"paired":len(paired),"unpaired":len(missing),
            "median_high_rise_peak_db_paired":round(float(np.median([
                r["contrast"]["high_rise_max_db"] for r in paired])),3) if paired else None,
            "median_high_rise_peak_db_unpaired":round(float(np.median([
                r["contrast"]["high_rise_max_db"] for r in missing])),3) if missing else None,
        }
    report={
        "analysis":"high_band_noise_floor_contrast_audit",
        "version":"2.4-development",
        "session_id":session_dir.name,
        "summary":summary,"rows":rows,"failures":failures,
        "limitations":[
            "Power subtraction is a descriptive feature, not a proposed positive classifier.",
            "Local baseline can contain earlier consonant or vowel energy.",
            "The same development recordings and synthetic white noise are reused.",
            "No thresholds or existing onset/CV predictions are altered.",
        ]
    }
    path=session_dir/"analysis"/"calibration_noise_floor_contrast_v2.4.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report
