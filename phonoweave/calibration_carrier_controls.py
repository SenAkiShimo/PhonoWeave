"""Adversarial carrier-token probe; not a clean vowel-only corpus.

The 46-prompt calibration protocol includes repeated /a/ carrier tokens.
These follow preceding consonants, so this audit explicitly reports potential
carry-over speech contamination. No detector thresholds are fitted here.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .audio import read_wav
from .calibration_io import load_calibration_session
from .calibration_transition_onset import acoustic_tracks, nominal_target_cue_ms
from .calibration_frication_shoulder import detect_frication_shoulder
from .calibration_robustness import perturb

SCENARIOS=("original","noise_35db_snr","noise_20db_snr")
OUTPUT="calibration_carrier_negative_controls_v1.7.json"


def evaluate_carriers(session_dir: Path) -> dict:
    session_dir=session_dir.expanduser().resolve()
    session=load_calibration_session(session_dir)
    rows=[]
    for index, rec in sorted(session["recordings"].items()):
        prompt=session["protocol"]["prompts"][index]
        tokens=prompt.get("tokens") or str(prompt.get("spoken_pattern","")).split()
        carriers=[i for i,t in enumerate(tokens) if t=="a" and t!=prompt.get("syllable")]
        if not carriers: continue
        x,sr=read_wav(session_dir/"recordings"/str(rec["wav"]))
        for scenario_idx, scenario in enumerate(SCENARIOS):
            y=perturb(np.asarray(x,dtype=float),scenario,
                      np.random.default_rng(7100+index*31+scenario_idx))
            features=acoustic_tracks(y,sr)
            for token_index in carriers:
                cue=nominal_target_cue_ms(token_index)
                candidate=detect_frication_shoulder(features,cue)
                rows.append({
                    "prompt_index":index,"base_unit":prompt.get("base_unit"),
                    "carrier_token_index":token_index,"carrier":"a",
                    "scenario":scenario,"nominal_cue_ms":round(cue,3),
                    "core_status":"candidate" if candidate.core_onset_ms is not None else candidate.status,
                    "core_onset_ms":candidate.core_onset_ms,
                    "shoulder_onset_ms":candidate.shoulder_onset_ms,
                    "shoulder_status":candidate.status,
                    "candidate_offset_from_carrier_cue_ms":None if candidate.shoulder_onset_ms is None else round(candidate.shoulder_onset_ms-cue,3),
                    "possible_previous_token_overlap":True,
                })
    summary={}
    for s in SCENARIOS:
        group=[r for r in rows if r["scenario"]==s]
        summary[s]={
            "carrier_tokens":len(group),
            "core_candidate_count":sum(r["core_onset_ms"] is not None for r in group),
            "shoulder_candidate_count":sum(r["shoulder_onset_ms"] is not None for r in group),
            "core_candidate_fraction":round(sum(r["core_onset_ms"] is not None for r in group)/len(group),4) if group else None,
            "shoulder_candidate_fraction":round(sum(r["shoulder_onset_ms"] is not None for r in group)/len(group),4) if group else None,
        }
    result={"analysis":"adversarial_carrier_token_probe","version":"1.7-development",
        "session_id":session_dir.name,"summary":summary,"rows":rows,
        "limitations":[
            "A-carrier tokens are not verified isolated-vowel negative samples.",
            "The search window may include frication from adjacent previous tokens, and nominal browser cue timing can drift.",
            "Reported candidate fractions are contamination/false-trigger indicators, NOT a measured true false-positive rate.",
            "For production specificity, record separate vowel-only, silence, breath and noise clips with an independent session."
        ]}
    dest=session_dir/"analysis"/OUTPUT
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result
