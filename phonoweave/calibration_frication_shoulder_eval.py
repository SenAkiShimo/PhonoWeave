"""Side-by-side, non-destructive comparison of v1.4 and shoulder candidate."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .audio import read_wav
from .calibration_io import load_calibration_session
from .calibration_dual_labels import load_dual_labels
from .calibration_transition_onset import acoustic_tracks, nominal_target_cue_ms
from .calibration_frication_shoulder import detect_frication_shoulder
from .calibration_sibilant_onset import SIBILANTS

def compare_shoulder_session(session_dir: Path) -> dict:
    session_dir=session_dir.expanduser().resolve()
    session=load_calibration_session(session_dir)
    labels=load_dual_labels(session_dir)["labels"]
    rows=[]
    for index, rec in sorted(session["recordings"].items()):
        prompt=session["protocol"]["prompts"][index]
        if prompt.get("base_unit") not in SIBILANTS:
            continue
        tokens=prompt.get("tokens") or str(prompt.get("spoken_pattern","")).split()
        matches=[i for i,t in enumerate(tokens) if t==prompt.get("syllable")]
        if len(matches)!=2:
            continue
        audio,rate=read_wav(session_dir/"recordings"/rec["wav"])
        features=acoustic_tracks(audio,rate)
        for occurrence,token_index in enumerate(matches,1):
            proposed=detect_frication_shoulder(features,nominal_target_cue_ms(token_index))
            mark=labels.get(str(index),{}).get("occurrences",{}).get(str(occurrence),{}).get("consonant_onset")
            eligible=isinstance(mark,dict) and mark.get("status")=="ok" and not mark.get("outside_expected_slot",False) and mark.get("source")=="manual_v0.3"
            manual=float(mark["absolute_anchor_ms"]) if eligible else None
            rows.append({
                "prompt_index":index,"base_unit":prompt["base_unit"],
                "context":prompt["context_family"],"occurrence":occurrence,
                "manual_absolute_ms":manual,"legacy_core_absolute_ms":proposed.core_onset_ms,
                "experimental_shoulder_absolute_ms":proposed.shoulder_onset_ms,
                "backtrace_ms":proposed.backtrace_ms,"status":proposed.status,
                "core_signed_error_ms":None if manual is None or proposed.core_onset_ms is None else round(proposed.core_onset_ms-manual,3),
                "shoulder_signed_error_ms":None if manual is None or proposed.shoulder_onset_ms is None else round(proposed.shoulder_onset_ms-manual,3),
            })
    def stats(k):
        eligible=[r for r in rows if r["manual_absolute_ms"] is not None]
        errors=[r[k] for r in eligible if r[k] is not None]
        return {"eligible":len(eligible),"matched":len(errors),
            "mae_ms":round(float(np.mean(np.abs(errors))),3) if errors else None,
            "p90_absolute_error_ms":round(float(np.percentile(np.abs(errors),90)),3) if errors else None}
    result={"analysis":"experimental_frication_shoulder_comparison","version":"1.5-development",
        "session_id":session_dir.name,"summary":{"v1_4_core":stats("core_signed_error_ms"),
        "v1_5_shoulder":stats("shoulder_signed_error_ms")},
        "warning":"Same speaker and labeled development set; do not treat an improvement as held-out accuracy.",
        "rows":rows}
    dest=session_dir/"analysis"/"calibration_frication_shoulder_v1.5.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result
