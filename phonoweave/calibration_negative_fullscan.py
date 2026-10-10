"""Full-recording negative control audit using unchanged onset detectors."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .audio import read_wav
from .calibration_transition_onset import acoustic_tracks
from .calibration_frication_shoulder import detect_frication_shoulder
from .calibration_negative_controls import CORPUS_DIR, CATEGORIES

def cluster_events(times: list[float], tolerance_ms: float=80.0) -> list[float]:
    """Coalesce overlapping-window detections into distinct onset events."""
    grouped: list[list[float]]=[]
    for t in sorted(times):
        if not grouped or t-grouped[-1][-1] > tolerance_ms:
            grouped.append([t])
        else:
            grouped[-1].append(t)
    return [round(float(np.median(g)),3) for g in grouped]

def fullscan(root: Path) -> dict:
    folder=root.expanduser().resolve()/CORPUS_DIR
    if not folder.is_dir():
        raise ValueError("negative controls directory not found")
    rows=[];failures=[]
    for path in sorted(folder.glob("*.json")):
        try:
            m=json.loads(path.read_text(encoding="utf-8"))
            if m.get("schema")!="negative_control_v1":
                continue
            if m.get("category") not in CATEGORIES:
                raise ValueError("unsupported category")
            wav=folder / str(m["wav"])
            if wav.resolve().parent != folder:
                raise ValueError("invalid recording path")
            samples,sr=read_wav(wav)
            duration_ms=1000*len(samples)/sr
            tracks=acoustic_tracks(samples,sr)
            # Search windows overlap. Require complete -350/+500ms support.
            cues=np.arange(350.0,duration_ms-500.0+0.001,200.0)
            core=[];shoulder=[]
            for cue in cues:
                result=detect_frication_shoulder(tracks,float(cue))
                if result.core_onset_ms is not None:
                    core.append(result.core_onset_ms)
                if result.shoulder_onset_ms is not None:
                    shoulder.append(result.shoulder_onset_ms)
            rows.append({"id":m["id"],"category":m["category"],
                         "device_category":m.get("device_category"),
                         "duration_ms":round(duration_ms,2),
                         "windows_checked":len(cues),
                         "core_onsets_ms":cluster_events(core),
                         "shoulder_onsets_ms":cluster_events(shoulder),
                         "has_core_trigger":bool(core),
                         "has_shoulder_trigger":bool(shoulder)})
        except (KeyError,TypeError,ValueError,OSError) as exc:
            failures.append({"file":path.name,"reason":str(exc)})
    summary={}
    for category in CATEGORIES:
        group=[r for r in rows if r["category"]==category]
        if group:
            summary[category]={
                "recordings":len(group),
                "recordings_with_core_trigger":sum(r["has_core_trigger"] for r in group),
                "recordings_with_shoulder_trigger":sum(r["has_shoulder_trigger"] for r in group),
                "distinct_shoulder_events":sum(len(r["shoulder_onsets_ms"]) for r in group)}
    result={"analysis":"negative_fullscan","version":"2.0-development","rows":rows,
            "summary":summary,"failures":failures,
            "limitations":"Overlapping cue windows share detections; onset lists are coalesced within 80ms. First/last 350/500ms may lack complete support. Self-reported negative labels need acoustic adjudication; this is not validated specificity."}
    out=folder/"negative_fullscan_v2.0.json"
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result
