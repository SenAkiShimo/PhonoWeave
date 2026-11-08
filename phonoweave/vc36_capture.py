"""Frozen, single-speaker, independently repeated VC pilot acquisition gate.

This module creates *only* an acquisition plan and audits provenance.
S01 is training-only, S02 is held-out. No source clips, synthesis rankings,
phone-boundary assumptions, or automatic split/share conclusions are created.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import wave

import numpy as np

SCHEMA = "phonoweave.vc36.independent_capture.v1"
AUDIT_SCHEMA = "phonoweave.vc36.capture_audit.v1"
ROLES = (("S01", "training_only"), ("S02", "heldout_only"))
REPEATS = 3
# Source labels are declared by the singer; they are not universal IPA.
TARGETS = (
    ("i", "m", "li ma"), ("ir", "m", "shi ma"), ("i0", "m", "zi ma"),
    ("i", "sh", "li sha"), ("ir", "sh", "shi sha"), ("i0", "sh", "zi sha"),
)


def _digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _record_id(session: str, nucleus: str, right: str, rep: int) -> str:
    return f"{session}_{nucleus}_{right}_take{rep:02d}"


def _stable_order(sid: str, ids: list[str]) -> list[str]:
    # A deterministic, balanced shuffle that cannot depend on observed audio.
    return sorted(ids, key=lambda x: sha256(f"VC36-A:{sid}:{x}".encode()).hexdigest())


def make_plan(*, speaker_id: str, pitch_hz: float = 293.66,
              duration_s: float = 3.0, first_note_ms: int = 750,
              second_note_ms: int = 750) -> dict:
    if not isinstance(speaker_id, str) or not speaker_id.strip() or len(speaker_id) > 100:
        raise ValueError("nonempty speaker_id is required")
    if (isinstance(pitch_hz, bool) or not isinstance(pitch_hz, (int, float))
            or not math.isfinite(pitch_hz) or not 60 <= pitch_hz <= 1000):
        raise ValueError("pitch_hz must be finite and in 60..1000")
    if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
            or not math.isfinite(duration_s) or not 1.0 <= duration_s <= 8.0):
        raise ValueError("duration_s must be finite and in 1..8")
    for value in (first_note_ms, second_note_ms):
        if type(value) is not int or not 200 <= value <= 3000:
            raise ValueError("note durations must be whole milliseconds within 200..3000")
    if (first_note_ms + second_note_ms) / 1000 + .4 > duration_s:
        raise ValueError("recording duration must leave >=400ms for pre/post-silence")
    sessions = [
        {"session_id": sid, "role": role, "capture_date": None,
         "device_id": None, "room_id": None, "gain_setting": None,
         "distance_cm": None, "setup_reset_confirmed": False}
        for sid, role in ROLES
    ]
    recs = []
    for sid, role in ROLES:
        unsorted = []
        for nucleus, right, prompt in TARGETS:
            for rep in range(1, REPEATS + 1):
                take_id = _record_id(sid, nucleus, right, rep)
                unsorted.append({
                    "take_id": take_id, "session_id": sid, "left_label": nucleus,
                    "right_label": right, "prompt": prompt, "repeat": rep,
                    "relative_wav": f"audio/{sid}/{take_id}.wav",
                    "wav_sha256": None, "capture_screen": None,
                    "qc_manual_reviewed": False,
                    "transition_onset_ms": None,
                    "transition_offset_ms": None,
                    "boundary_reviewer": None,
                })
        order = _stable_order(sid, [r["take_id"] for r in unsorted])
        ranking = {key: i + 1 for i, key in enumerate(order)}
        for r in unsorted:
            r["presentation_order"] = ranking[r["take_id"]]
        recs.extend(unsorted)
    result = {
        "schema": SCHEMA, "study_id": "vc36_independent_session_pilot_v1",
        "speaker_id": speaker_id,
        "frozen_score": {"pitch_hz": pitch_hz, "first_note_ms": first_note_ms,
                         "second_note_ms": second_note_ms,
                         "legato": True, "target_tone": "level",
                         "duration_s": duration_s,
                         "instructions": "Sing the entire 2-syllable prompt freshly on one pitch; no copied clips."},
        "frozen_targets": [list(x) for x in TARGETS],
        "sessions": sessions,
        "recordings": recs,
        "scientific_scope": "single_singer_pilot_only_no_efficacy_claim",
    }
    frozen = {k: result[k] for k in ("schema", "study_id", "speaker_id",
              "frozen_score", "frozen_targets", "scientific_scope")}
    result["protocol_sha256"] = sha256(json.dumps(
        frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    return result


def _shape_errors(data: dict) -> list[str]:
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return ["unsupported_manifest_schema"]
    try:
        expected = make_plan(
            speaker_id=data["speaker_id"],
            pitch_hz=data["frozen_score"]["pitch_hz"],
            duration_s=data["frozen_score"]["duration_s"],
            first_note_ms=data["frozen_score"]["first_note_ms"],
            second_note_ms=data["frozen_score"]["second_note_ms"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        return [f"invalid_frozen_protocol:{exc}"]
    errors = []
    for key in ("schema", "study_id", "speaker_id", "frozen_score", "frozen_targets",
                "scientific_scope", "protocol_sha256"):
        if data.get(key) != expected[key]:
            errors.append(f"frozen_protocol_modified:{key}")
    sessions = data.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 2:
        errors.append("session_count_or_structure_modified")
    else:
        if [(s.get("session_id"), s.get("role")) for s in sessions if isinstance(s, dict)] != list(ROLES):
            errors.append("session_roles_modified")
    recs = data.get("recordings")
    if not isinstance(recs, list) or len(recs) != 36:
        errors.append("recording_grid_size_modified")
    else:
        a = {x["take_id"]: x for x in expected["recordings"]}
        seen = set()
        for row in recs:
            if not isinstance(row, dict) or not isinstance(row.get("take_id"), str):
                errors.append("malformed_recording"); continue
            id = row["take_id"]
            if id in seen or id not in a:
                errors.append(f"duplicate_or_unexpected_take:{id}"); continue
            seen.add(id)
            for key in ("session_id", "left_label", "right_label", "prompt", "repeat",
                        "relative_wav", "presentation_order"):
                if row.get(key) != a[id][key]:
                    errors.append(f"{id}:frozen_slot_modified:{key}")
        if seen != set(a):
            errors.append("missing_or_duplicate_take_ids")
    return errors


def _read_pcm16(path: Path) -> tuple[dict, bytes]:
    try:
        with wave.open(str(path), "rb") as wav:
            if (wav.getcomptype() != "NONE" or wav.getnchannels() != 1
                    or wav.getsampwidth() != 2 or wav.getframerate() not in (44100, 48000, 96000)):
                raise ValueError("mono PCM16 at 44.1/48/96 kHz required")
            hz = wav.getframerate()
            frames = wav.getnframes()
            raw = wav.readframes(frames)
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError("unreadable PCM16 WAV") from exc
    if len(raw) != frames * 2:
        raise ValueError("truncated PCM16 WAV")
    seconds = frames / hz
    if not 1.0 <= seconds <= 8.0:
        raise ValueError("WAV duration outside 1..8 seconds")
    s = np.frombuffer(raw, dtype="<i2")
    x = s.astype(np.float64) / 32768
    rms = float(np.sqrt(np.mean(x*x)))
    peak = float(np.max(np.abs(x)))
    return {
        "rate": hz, "frames": frames, "duration_s": round(seconds, 4),
        "rms_dbfs": round(20 * math.log10(max(rms, 1e-12)), 2),
        "peak_dbfs": round(20 * math.log10(max(peak, 1e-12)), 2),
        "clipped_samples": int(np.count_nonzero((s >= 32760) | (s <= -32760))),
        "screen_flags": [flag for condition,flag in (
            (peak >= .999,"digital_clipping"),
            (rms < .01,"low_level"),
            (peak > .95,"low_headroom"),
        ) if condition],
    }, raw


def _secure_slot(root: Path, rel: str) -> Path:
    if not isinstance(rel, str):
        raise ValueError("invalid relative path")
    part = Path(rel)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("path traversal rejected")
    p = (root / part).resolve()
    if not p.is_relative_to(root):
        raise ValueError("slot escapes corpus root")
    return p


def _save_manifest(path: Path, data: dict) -> None:
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".vc36-manifest-",
                                         suffix=".tmp", encoding="utf-8", delete=False) as stream:
            tmp = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _session_gate(data: dict, sid: str, capture_date: str,
                  device_id: str, room_id: str, gain_setting: str,
                  distance_cm: float, setup_reset: bool) -> dict:
    if _shape_errors(data):
        raise ValueError("frozen recording plan modified")
    sessions = {x["session_id"]: x for x in data["sessions"]}
    if sid not in sessions:
        raise ValueError("invalid session")
    try:
        when = date.fromisoformat(capture_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("ISO date required") from exc
    if not all(isinstance(x, str) and x.strip() for x in (device_id,room_id,gain_setting)):
        raise ValueError("device, room and gain identifiers required")
    if not isinstance(distance_cm, (int,float)) or isinstance(distance_cm,bool) or not math.isfinite(distance_cm) or not 5 <= distance_cm <= 150:
        raise ValueError("microphone distance must be 5..150cm")
    if setup_reset is not True:
        raise ValueError("independent session setup reset must be confirmed")
    if sid == "S02":
        a = sessions["S01"]
        if not a["capture_date"] or not all(r["wav_sha256"] for r in data["recordings"] if r["session_id"] == "S01"):
            raise ValueError("S01 capture must finish before S02")
        if when <= date.fromisoformat(a["capture_date"]):
            raise ValueError("S02 must occur on a later calendar day than S01")
    proposed = {"capture_date": capture_date,"device_id":device_id,"room_id":room_id,
                "gain_setting":gain_setting,"distance_cm":distance_cm,
                "setup_reset_confirmed":True}
    current = sessions[sid]
    for k,v in proposed.items():
        if current.get(k) not in (None, False, v):
            raise ValueError(f"cannot change recorded session metadata: {k}")
    return proposed


def ingest_session(manifest: Path, source_dir: Path, *, session_id: str,
                   capture_date: str, device_id: str, room_id: str,
                   gain_setting: str, distance_cm: float, setup_reset: bool) -> dict:
    """Ingest canonical WAVs from a DAW folder. Resume safe, never overwrite."""
    manifest = manifest.expanduser().resolve(strict=True)
    root = manifest.parent
    source_dir = source_dir.expanduser().resolve(strict=True)
    if not source_dir.is_dir() or source_dir == root or root in source_dir.parents:
        raise ValueError("source must be separate external capture directory")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    metadata = _session_gate(data, session_id, capture_date, device_id, room_id,
                            gain_setting, distance_cm, setup_reset)
    slots = sorted((r for r in data["recordings"] if r["session_id"] == session_id),
                   key=lambda x:x["presentation_order"])
    # Prevalidate ALL supplied files and all existing slots before writing.
    to_add=[]
    current_hashes = {r["wav_sha256"] for r in data["recordings"] if r.get("wav_sha256")}
    if len(current_hashes) != sum(bool(r.get("wav_sha256")) for r in data["recordings"]):
        raise ValueError("duplicate SHA-256 already present in the manifest")
    existing_formats=set()
    existing_pcm=set()
    for r in data["recordings"]:
        target = _secure_slot(root,r["relative_wav"])
        if target.exists():
            if not r.get("wav_sha256") or _digest(target)!=r["wav_sha256"]:
                raise ValueError("existing output WAV not registered or source was altered")
            info,raw=_read_pcm16(target)
            existing_formats.add(info["rate"])
            existing_pcm.add(sha256(raw).hexdigest())
        elif r.get("wav_sha256"):
            raise ValueError("manifest references missing source WAV")
    if len(existing_formats)>1:
        raise ValueError("incompatible sample rates in already captured takes")
    seen_names=set()
    seen_pcm=set()
    for r in slots:
        target=_secure_slot(root,r["relative_wav"])
        src=(source_dir/(r["take_id"]+".wav")).resolve()
        if not src.is_relative_to(source_dir):
            raise ValueError("source symlink escapes capture folder")
        if target.exists():
            continue
        if not src.exists():
            continue
        info,pcm=_read_pcm16(src)
        digest=_digest(src)
        pcm_hash=sha256(pcm).hexdigest()
        if digest in current_hashes or digest in seen_names:
            raise ValueError(f"duplicated take audio: {r['take_id']}")
        seen_names.add(digest)
        if pcm_hash in seen_pcm or pcm_hash in existing_pcm:
            raise ValueError(f"duplicate decoded PCM audio: {r['take_id']}")
        seen_pcm.add(pcm_hash)
        if existing_formats and info["rate"] not in existing_formats:
            raise ValueError("sample rate mismatch against already captured session")
        if to_add and info["rate"] != to_add[0][3]["rate"]:
            raise ValueError("sample rate mismatch across incoming takes")
        to_add.append((r,target,src,info,digest,pcm_hash))
    if not to_add:
        return {"imported":0,"session":session_id,"missing_count":sum(
            not _secure_slot(root,r["relative_wav"]).exists() for r in slots)}
    # Mutations are exclusive. If any copy fails, cleanup only copies created here.
    created=[]
    try:
        for r,target,src,info,digest,pcm_hash in to_add:
            target.parent.mkdir(parents=True,exist_ok=True)
            with src.open("rb") as inp, target.open("xb") as out:
                shutil.copyfileobj(inp,out)
            created.append(target)
            if _digest(target) != digest:
                raise ValueError("import copy integrity failure")
            r["wav_sha256"]=digest
            r["capture_screen"]={k:info[k] for k in (
                "rate","frames","duration_s","rms_dbfs","peak_dbfs",
                "clipped_samples","screen_flags")}
            r["capture_screen"]["pcm_sha256"] = pcm_hash
        for k,v in metadata.items():
            next(s for s in data["sessions"] if s["session_id"] == session_id)[k]=v
        _save_manifest(manifest,data)
    except BaseException:
        for p in created:
            p.unlink(missing_ok=True)
        raise
    return {"imported":len(created),"session":session_id,
            "missing_count":len(slots)-sum(bool(r["wav_sha256"]) for r in slots),
            "qc_flags":sum(bool(r.get("capture_screen") and r["capture_screen"]["screen_flags"])
                           for r in slots)}


def audit_manifest(data: dict, root: Path) -> dict:
    """Independent-session and verified PCM integrity gate. Never infer outcomes."""
    errors=_shape_errors(data)
    missing=[]; no_hash=[]; unannotated=[]; review_needed=[]
    formats=set(); observed_hashes=set(); observed_pcm=set(); counts=Counter(); verified=0
    duration_by_context=Counter()
    root=root.expanduser().resolve(strict=True)
    if errors:
        return {"schema":AUDIT_SCHEMA,"capture_complete":False,
                "analysis_ready":False,"errors":errors,"missing":[],
                "outcome_status":"not_evaluated"}
    sessions=data["sessions"]
    stamps=[]
    for s in sessions:
        try:
            stamp=date.fromisoformat(str(s.get("capture_date")))
            stamps.append(stamp)
        except (TypeError,ValueError):
            if s.get("capture_date") is not None:
                errors.append(f"invalid_capture_date:{s['session_id']}")
        for name in ("device_id","room_id","gain_setting","distance_cm"):
            if not s.get(name): review_needed.append(f"{s['session_id']}:{name}")
        if s.get("setup_reset_confirmed") is not True:
            review_needed.append(f"{s['session_id']}:setup_reset_confirmed")
    if len(stamps)==2 and not stamps[1]>stamps[0]:
        errors.append("S02_must_follow_S01_on_different_day")
    for r in data["recordings"]:
        id=r["take_id"]
        try:
            p=_secure_slot(root,r["relative_wav"])
            if not p.is_file():
                missing.append(id);continue
            info,raw=_read_pcm16(p)
            if _digest(p)!=r.get("wav_sha256"):
                errors.append(f"checksum_mismatch:{id}");continue
            digest=r["wav_sha256"]
            if digest in observed_hashes:
                errors.append(f"duplicate_audio:{id}");continue
            observed_hashes.add(digest); verified+=1
            pcm_hash=sha256(raw).hexdigest()
            if pcm_hash in observed_pcm:
                errors.append(f"duplicate_decoded_pcm:{id}")
            observed_pcm.add(pcm_hash)
            formats.add(info["rate"])
            duration_by_context[(r["session_id"],r["left_label"],r["right_label"])] += info["duration_s"]
            counts[(r["session_id"],r["left_label"],r["right_label"])]+=1
            if r.get("capture_screen") is None:
                review_needed.append(f"{id}:capture_screen_missing")
            elif (r["capture_screen"].get("rate")!=info["rate"] or
                  r["capture_screen"].get("frames")!=info["frames"] or
                  r["capture_screen"].get("pcm_sha256")!=pcm_hash):
                errors.append(f"screen_metadata_mismatch:{id}")
            if info["clipped_samples"]:
                errors.append(f"clipped_audio:{id}")
            if info["screen_flags"] and r.get("qc_manual_reviewed") is not True:
                review_needed.append(f"{id}:qc_flags_unreviewed")
            a,b=r.get("transition_onset_ms"),r.get("transition_offset_ms")
            if a is None or b is None or not r.get("boundary_reviewer"):
                unannotated.append(id)
            elif (isinstance(a,bool) or isinstance(b,bool) or
                  not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or
                  not math.isfinite(a) or not math.isfinite(b) or
                  not 0<=a<b<=info["duration_s"]*1000):
                errors.append(f"invalid_transition_boundary:{id}")
        except (ValueError,OSError,EOFError,wave.Error) as exc:
            errors.append(f"invalid_audio:{id}:{exc}")
    if len(formats)>1:
        errors.append("mixed_audio_sample_rates")
    expected=(len(ROLES)*len(TARGETS)*REPEATS)
    capture_complete=verified==expected and not (errors or missing or no_hash or review_needed)
    analysis_ready=capture_complete and not unannotated
    return {"schema":AUDIT_SCHEMA,"planned_takes":expected,"verified_wav_files":verified,
            "per_session_context_verified":{
                sid:{f"{left}_{right}":counts[(sid,left,right)] for left,right,_ in TARGETS}
                for sid,_ in ROLES},
            "capture_complete":capture_complete,"analysis_ready":analysis_ready,
            "captured_audio_duration_seconds":round(sum(duration_by_context.values()),3),
            "per_session_context_duration_seconds":{
                sid:{f"{left}_{right}":round(duration_by_context[(sid,left,right)],3)
                     for left,right,_ in TARGETS} for sid,_ in ROLES},
            "missing":sorted(set(missing)),"unreviewed":sorted(set(review_needed)),
            "unannotated":sorted(set(unannotated)),"errors":errors,
            "outcome_status":"not_evaluated","speaker_adaptive_effect":"not_assessed"}


def write_plan(path: Path, **kwargs) -> dict:
    path=path.expanduser().resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    result=make_plan(**kwargs)
    with path.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    return result


def _write_new_json(data:dict,path:Path,source_root:Path)->None:
    path=path.expanduser().resolve()
    if path==source_root or source_root in path.parents:
        raise ValueError("audit report must be outside the recording corpus")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def write_cue_sheet(data: dict, path: Path) -> Path:
    if _shape_errors(data):
        raise ValueError("frozen recording plan altered")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("session", "order", "take_id", "stimulus_pinyin", "right",
                         "left", "repeat", "expected_filename", "session_date", "notes"))
        for r in sorted(data["recordings"],
                        key=lambda x: (x["session_id"],x["presentation_order"])):
            writer.writerow((r["session_id"],r["presentation_order"],r["take_id"],
                             r["prompt"],r["right_label"],r["left_label"],r["repeat"],
                             r["take_id"] + ".wav", "", ""))
    return path


def apply_boundary_csv(manifest: Path, annotations: Path) -> dict:
    """Attach manually approved intervals; never derive any onset from OTO."""
    manifest = manifest.expanduser().resolve(strict=True)
    root=manifest.parent
    data=json.loads(manifest.read_text(encoding="utf-8"))
    if _shape_errors(data):
        raise ValueError("frozen recording protocol was modified")
    with annotations.expanduser().open(newline="",encoding="utf-8-sig") as stream:
        reader=csv.DictReader(stream)
        fields=("take_id", "source_wav_sha256", "transition_onset_ms",
                "transition_offset_ms", "boundary_reviewer", "qc_manual_reviewed")
        if not reader.fieldnames or set(reader.fieldnames)!=set(fields):
            raise ValueError("boundary CSV must match the documented column names")
        rows=list(reader)
    expected={x["take_id"]:x for x in data["recordings"]}
    seen=set(); pending=[]
    for row in rows:
        key=row["take_id"]
        if key not in expected or key in seen:
            raise ValueError("unexpected or repeated take in boundary CSV")
        seen.add(key)
        item=expected[key]
        if not item.get("wav_sha256") or row["source_wav_sha256"]!=item["wav_sha256"]:
            raise ValueError("boundary annotation source hash mismatch")
        p=_secure_slot(root,item["relative_wav"])
        if not p.is_file() or _digest(p)!=item["wav_sha256"]:
            raise ValueError("reviewed WAV missing or changed")
        info,_=_read_pcm16(p)
        reviewer=row["boundary_reviewer"].strip()
        if not reviewer:
            raise ValueError("reviewer required for every approved boundary")
        try:
            a,b=float(row["transition_onset_ms"]),float(row["transition_offset_ms"])
        except (TypeError,ValueError) as exc:
            raise ValueError("all approved boundaries need numeric onset/offset") from exc
        if not math.isfinite(a) or not math.isfinite(b) or not 0<=a<b<=info["duration_s"]*1000:
            raise ValueError("manual boundary outside WAV or reversed")
        qc=row["qc_manual_reviewed"].strip().lower()
        if qc not in ("true", "false"):
            raise ValueError("qc_manual_reviewed must be true or false")
        changes={"transition_onset_ms":a,"transition_offset_ms":b,
                 "boundary_reviewer":reviewer,"qc_manual_reviewed":qc=="true"}
        if item.get("boundary_reviewer"):
            for field,value in changes.items():
                if item[field]!=value:
                    raise ValueError("existing reviewed boundary cannot be silently replaced")
        pending.append((item,changes))
    for item,changes in pending:
        item.update(changes)
    _save_manifest(manifest,data)
    return {"annotations_accepted":len(pending),"remaining_unannotated":sum(
        not r.get("boundary_reviewer") for r in data["recordings"])}


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(prog="phonoweave vc36")
    sub=p.add_subparsers(dest="action",required=True)
    plan=sub.add_parser("plan")
    plan.add_argument("--manifest",required=True,type=Path)
    plan.add_argument("--speaker-id",required=True)
    plan.add_argument("--pitch-hz",type=float,default=293.66)
    plan.add_argument("--duration-s",type=float,default=3.0)
    plan.add_argument("--first-note-ms",type=int,default=750)
    plan.add_argument("--second-note-ms",type=int,default=750)
    cues=sub.add_parser("cues")
    cues.add_argument("--manifest",required=True,type=Path)
    cues.add_argument("--output",required=True,type=Path)
    annotate=sub.add_parser("annotate")
    annotate.add_argument("--manifest",required=True,type=Path)
    annotate.add_argument("--csv",required=True,type=Path)
    ingest=sub.add_parser("ingest")
    ingest.add_argument("--manifest",required=True,type=Path)
    ingest.add_argument("--source",required=True,type=Path)
    ingest.add_argument("--session",choices=("S01","S02"),required=True)
    ingest.add_argument("--date",required=True)
    ingest.add_argument("--device-id",required=True)
    ingest.add_argument("--room-id",required=True)
    ingest.add_argument("--gain-setting",required=True)
    ingest.add_argument("--distance-cm",type=float,required=True)
    ingest.add_argument("--confirm-reset",action="store_true")
    audit=sub.add_parser("audit")
    audit.add_argument("--manifest",required=True,type=Path)
    audit.add_argument("--output",required=True,type=Path)
    args=p.parse_args(argv)
    try:
        if args.action=="plan":
            report=write_plan(args.manifest,speaker_id=args.speaker_id,
                              pitch_hz=args.pitch_hz,duration_s=args.duration_s,
                              first_note_ms=args.first_note_ms,
                              second_note_ms=args.second_note_ms)
            print(f"Frozen VC capture targets: {len(report['recordings'])}")
        elif args.action=="cues":
            data=json.loads(args.manifest.read_text(encoding="utf-8"))
            print("Cue sheet:",write_cue_sheet(data,args.output))
        elif args.action=="annotate":
            print(json.dumps(apply_boundary_csv(args.manifest,args.csv),indent=2))
        elif args.action=="ingest":
            report=ingest_session(args.manifest,args.source,session_id=args.session,
                                  capture_date=args.date,device_id=args.device_id,
                                  room_id=args.room_id,gain_setting=args.gain_setting,
                                  distance_cm=args.distance_cm,setup_reset=args.confirm_reset)
            print(json.dumps(report,indent=2))
        else:
            manifest=args.manifest.expanduser().resolve(strict=True)
            report=audit_manifest(json.loads(manifest.read_text(encoding="utf-8")),manifest.parent)
            _write_new_json(report,args.output,manifest.parent)
            print(json.dumps(report,indent=2))
            return 0 if report["analysis_ready"] else 2
    except (OSError,ValueError,TypeError,KeyError,EOFError,json.JSONDecodeError) as exc:
        p.error(str(exc))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
