"""Shadow-only onset backtrace comparison on selected annotated fricatives."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .audio import read_wav
from .calibration_io import load_calibration_session
from .calibration_dual_labels import load_dual_labels
from .calibration_transition_onset import acoustic_tracks, nominal_target_cue_ms
from .calibration_cv_transition import cv_tracks
from .calibration_excess_fallback import detect_excess_fallback
from .calibration_early_frication import trace_early_frication
from .calibration_multiseed_noise import add_calibrated_noise, _recording_seed

TARGETS=((12,"x","plain"),(13,"x","rounded"),(14,"sh","plain"),(15,"sh","rounded"))

def evaluate(session_dir: Path) -> dict:
    session_dir=session_dir.expanduser().resolve()
    session=load_calibration_session(session_dir)
    labels=load_dual_labels(session_dir)["labels"]
    rows=[]
    for index,base,context in TARGETS:
        prompt=session["protocol"]["prompts"][index]
        if (prompt["base_unit"],prompt["context_family"])!=(base,context):
            raise ValueError(f"protocol mismatch at index {index}")
        recording=session["recordings"].get(index)
        if not recording:raise ValueError(f"missing recording {index}")
        audio,sr=read_wav(session_dir/"recordings"/recording["wav"])
        tokens=prompt.get("tokens") or prompt["spoken_pattern"].split()
        slots=[i for i,token in enumerate(tokens) if token==prompt["syllable"]]
        if len(slots)!=2:raise ValueError("expected two target occurrences")
        for noise,seed in (("original",None),("white_20db",103),("pink_20db",103)):
            y=audio if seed is None else add_calibrated_noise(
                audio,sr,20.0,noise.split("_")[0],
                _recording_seed(seed,recording["wav"],noise.split("_")[0]))
            onset_tracks=acoustic_tracks(y,sr); vowel_tracks=cv_tracks(y,sr)
            for occurrence,position in enumerate(slots,1):
                cue=nominal_target_cue_ms(position)
                result=detect_excess_fallback(onset_tracks,vowel_tracks,cue)
                shadow=trace_early_frication(onset_tracks,result.consonant_onset_ms,cue)
                mark=labels.get(str(index),{}).get("occurrences",{}).get(str(occurrence),{}).get("consonant_onset")
                gold=(float(mark["absolute_anchor_ms"]) if isinstance(mark,dict)
                      and mark.get("status")=="ok" and mark.get("source")=="manual_v0.3"
                      and not mark.get("outside_expected_slot",False) else None)
                original=result.consonant_onset_ms
                proposed=shadow.get("onset_ms")
                rows.append({
                    "prompt_index":index,"base_unit":base,"context":context,
                    "occurrence":occurrence,"condition":noise,
                    "manual_onset_ms":gold,"baseline_onset_ms":original,
                    "shadow_onset_ms":proposed,"status":shadow["status"],
                    "backtrace_ms":shadow.get("backtrace_ms"),
                    "baseline_error_ms":round(original-gold,3) if gold is not None and original is not None else None,
                    "shadow_error_ms":round(proposed-gold,3) if gold is not None and proposed is not None else None,
                })
    summaries={}
    for condition in ("original","white_20db","pink_20db"):
        group=[r for r in rows if r["condition"]==condition]
        eligible=[r for r in group if r["manual_onset_ms"] is not None]
        summary={"targets":len(group),"eligible_labels":len(eligible)}
        for method in ("baseline","shadow"):
            errs=[abs(r[f"{method}_error_ms"]) for r in eligible if r[f"{method}_error_ms"] is not None]
            summary[method]={"matched":len(errs),
                "conditional_mae_ms":round(float(np.mean(errs)),3) if errs else None,
                "errors_over_40ms":sum(x>40 for x in errs)}
        summaries[condition]=summary
    out={"analysis":"early_frication_shadow","version":"2.7-development",
         "session_id":session_dir.name,"summary":summaries,"rows":rows,
         "caution":"Shadow candidates are not production predictions. Comparisons reuse development recordings; four previously missing manual anchors must be placed independently."}
    dest=session_dir/"analysis"/"calibration_early_frication_v2.7.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return out

def main(argv=None):
    p=argparse.ArgumentParser(description="Compare early-frication shadow candidate")
    p.add_argument("session",type=Path)
    out=evaluate(p.parse_args(argv).session)
    for condition,data in out["summary"].items():
        print(condition,data)
    print("Saved: analysis/calibration_early_frication_v2.7.json")

if __name__=="__main__":
    main()
