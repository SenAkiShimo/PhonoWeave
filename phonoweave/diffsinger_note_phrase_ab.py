"""Note-timed DiffSinger-derived CVVC phrase *preview* and source-accounting gate.

Builds multi-note connected PCM from a frozen six-context score and existing
DiffSinger-rendered source ZIPs, under fixed donor budgets. Uses FFmpeg's
Rubber Band audio filter for actual time-scale/pitch processing (not an UTAU
resampler or OpenUtau/wavtool). An independent DiffSinger test-reference is
required for any comparative accuracy/effectiveness claim and is NOT created
here. No source archive or original OTO is modified.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import wave
import zipfile

import numpy as np

from .diffsinger_vc_splice_ab import (
    FADE_MS, LEFTS, POLICIES, RIGHTS, _bank_data, _crossfade,
    _extract, _parse_oto, _wav_bytes, _wav_pcm,
)

SCORE_SCHEMA = "phonoweave.diffsinger_note_phrase_score.v1"
OUTPUT_SCHEMA = "phonoweave.diffsinger_note_phrase_ab.v1"
TEACHER_SCHEMA = "phonoweave.external_diffsinger_teacher_receipt.v1"
SYLLABLES = {
    "i": ("li", "eriin.wav"),
    "ir": ("shi", "ingi0ir.wav"),
    "i0": ("zi", "ingi0ir.wav"),
}
RIGHT_SYLLABLES = {"m": "ma", "sh": "sha"}
SOURCE_CV_MS = 300
SOURCE_SECOND_MS = 380
MS_FADE_PAIRS = 12
PAIR_EDGE_FADE_MS = 6
BUDGET_UNIT_MS = 120


def _hash_json(obj: dict) -> str:
    return sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def frozen_score() -> dict:
    """Fixed six-target two-phrase pilot. Pitches are relative to each bank root.

    This is not a separately generated DiffSinger target: these are score
    targets chosen for engineering demonstrations and later teacher rendering.
    """
    p1 = [
        {"left": "i", "right": "m", "semitones": [0, 2], "durations_ms": [400, 510], "rest_ms": 80},
        {"left": "ir", "right": "sh", "semitones": [3, 2], "durations_ms": [430, 460], "rest_ms": 80},
        {"left": "i0", "right": "m", "semitones": [0, -1], "durations_ms": [390, 520], "rest_ms": 0},
    ]
    p2 = [
        {"left": "i0", "right": "sh", "semitones": [-1, 1], "durations_ms": [425, 485], "rest_ms": 80},
        {"left": "i", "right": "sh", "semitones": [2, 4], "durations_ms": [395, 505], "rest_ms": 80},
        {"left": "ir", "right": "m", "semitones": [2, 0], "durations_ms": [430, 490], "rest_ms": 0},
    ]
    return {
        "schema": SCORE_SCHEMA,
        "score_id": "lynneqing_vc_six_context_score_freeze_v1",
        "root_pitch_semantics": "subbank_pitch_name_is_reference_not_measured_f0",
        "source_unit_policy": "same_style_same_subbank_only",
        "target_reference": "absent_external_diffsinger_render_required",
        "phrases": [
            {"id": "phrase_a", "pairs": p1},
            {"id": "phrase_b", "pairs": p2},
        ],
    }


def _frozen_validate(score: dict) -> None:
    if score.get("schema") != SCORE_SCHEMA or score.get("target_reference") != "absent_external_diffsinger_render_required":
        raise ValueError("invalid source-only, frozen note-score schema")
    if score.get("root_pitch_semantics") != "subbank_pitch_name_is_reference_not_measured_f0":
        raise ValueError("pitch reference semantics changed")
    if score.get("source_unit_policy") != "same_style_same_subbank_only":
        raise ValueError("cross-style or cross-subbank policy not permitted")
    rows=score.get("phrases")
    if not isinstance(rows,list) or len(rows)!=2 or [r.get("id") for r in rows]!=["phrase_a","phrase_b"]:
        raise ValueError("frozen score needs two named phrases")
    ids=set()
    for p in rows:
        pairs=p.get("pairs")
        if not isinstance(pairs,list) or len(pairs)!=3:raise ValueError("three note-pairs per phrase required")
        for j,event in enumerate(pairs):
            if not isinstance(event,dict) or event.get("left") not in LEFTS or event.get("right") not in RIGHTS:
                raise ValueError("unknown syllable transition class")
            key=(event["left"],event["right"])
            if key in ids:raise ValueError("source-only test score repeated a VC target")
            ids.add(key)
            offsets=event.get("semitones")
            durations=event.get("durations_ms")
            rest=event.get("rest_ms")
            if (not isinstance(offsets,list) or len(offsets)!=2 or
                any(type(x) is not int or abs(x)>5 for x in offsets) or
                not isinstance(durations,list) or len(durations)!=2 or
                any(type(x) is not int or not 300<=x<=850 for x in durations) or
                type(rest) is not int or not 0<=rest<=200 or
                (j==2 and rest!=0)):
                raise ValueError("bad note pitch/duration/rest")
    if ids != set((left,right) for left in LEFTS for right in RIGHTS):
        raise ValueError("each target must appear exactly once across score")


def _pitch_root(name: str) -> int:
    m=re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d)",name)
    if not m:raise ValueError("subbank pitch must use scientific pitch notation")
    base={"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}[m.group(1).upper()]
    accidental={"":0,"#":1,"b":-1}[m.group(2)]
    value=(int(m.group(3))+1)*12+base+accidental
    if not 36<=value<=96:raise ValueError("implausible subbank reference pitch")
    return value


def _true_wav_from_zip(z:zipfile.ZipFile,root:str,path:str) -> tuple[np.ndarray,int,str]:
    if not root.endswith('/') or '..' in root.split('/') or root.startswith('/'):
        raise ValueError("unsafe source ZIP root")
    return _wav_pcm(z,root+path)


def _read_syllables(config:dict, source_rate:int) -> tuple[dict,dict,dict]:
    """Freeze source CV identities and extract original PCM, not constructed tones."""
    archive=Path(config["archive"]).expanduser().resolve(strict=True)
    root,prefix=config["root"],config["prefix"]
    with zipfile.ZipFile(archive) as z:
        oto=_parse_oto(z.read(root+'oto.ini'),prefix)
        required={"eriin.wav","ingi0ir.wav","a.wav"}
        decoded={name:_true_wav_from_zip(z,root,name) for name in required}
        starts={};lasts={};source={}
        all_syllables={**SYLLABLES, **{k:(v,"a.wav") for k,v in RIGHT_SYLLABLES.items()}}
        for name,(surface,wav) in all_syllables.items():
            matches=oto.get((wav,prefix+surface),[])
            if len(matches)!=1:
                raise ValueError(f"syllable source missing or ambiguous: {surface}/{wav}")
            record=matches[0];timing=record["time"]
            pcm,rate,pcm_hash=decoded[wav]
            if rate!=source_rate:raise ValueError("different source WAV sample rates")
            duration=SOURCE_SECOND_MS if name in RIGHTS else SOURCE_CV_MS
            segment=_extract(pcm,rate,timing[0],duration)
            if float(np.sqrt(np.mean(segment**2)))<1e-5:
                raise ValueError("source syllable is silent")
            source[name]={
                "original_alias":prefix+surface,"original_wav":wav,
                "oto_line":record["line"],"wav_pcm_sha256":pcm_hash,
                "source_start_ms":round(timing[0],3),"source_duration_ms":duration,
                "source_phone_boundaries_human_checked":False,
            }
            if name in RIGHTS:lasts[name]=segment
            else:starts[name]=segment
        return starts,lasts,source


def rubberband_process(signal:np.ndarray, sample_rate:int, *,
                       target_frames:int, semitones:int, ffmpeg:str="ffmpeg") -> np.ndarray:
    """FFmpeg Rubber Band transform; error if backend unavailable, never fake.

    The note source pitch is assumed from the subbank *label*, and remains a
    calibration caveat. Non-pitch-locked fricative tails may be affected by
    this whole-note processing; all policy arms use precisely same backend.
    """
    if signal.ndim!=1 or not len(signal) or not np.all(np.isfinite(signal)):
        raise ValueError("invalid source note")
    if target_frames < sample_rate//10 or type(semitones) is not int or abs(semitones)>5:
        raise ValueError("invalid note target")
    if not shutil.which(ffmpeg):raise RuntimeError("FFmpeg with rubberband filter is required")
    ratio=2**(semitones/12)
    tempo=len(signal)/target_frames
    if not .35 <= tempo <= 2.5:raise ValueError("excessive note time stretch")
    cmd=[ffmpeg,"-hide_banner","-nostdin","-loglevel","error","-threads","1",
         "-f","f32le","-ar",str(sample_rate),"-ac","1","-i","pipe:0",
         "-af",f"rubberband=tempo={tempo:.9f}:pitch={ratio:.9f}:formant=preserved:transients=smooth",
         "-f","f32le","-ar",str(sample_rate),"-ac","1","pipe:1"]
    try:
        result=subprocess.run(cmd,input=np.asarray(signal,dtype='<f4').tobytes(),
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=40,check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Rubber Band note render timed out") from exc
    if result.returncode!=0:
        raise RuntimeError("FFmpeg rubberband failed: "+result.stderr.decode(errors="replace")[:500])
    values=np.frombuffer(result.stdout,dtype='<f4').astype(np.float64)
    if len(values)==0 or not np.all(np.isfinite(values)):
        raise RuntimeError("Rubber Band produced empty/nonfinite PCM")
    discrepancy=abs(len(values)-target_frames)
    if discrepancy>max(round(sample_rate*.030),round(target_frames*.03)):
        raise RuntimeError(f"Rubber Band time output unexpected ({len(values)} vs {target_frames})")
    if len(values)<target_frames:
        # At most 30ms; expected transient-frame difference is logged.
        values=np.pad(values,(0,target_frames-len(values)))
    elif len(values)>target_frames:
        values=values[:target_frames]
    return values


def _midi_vlq(value:int)->bytes:
    if type(value) is not int or not 0<=value<2**28:
        raise ValueError("MIDI delta-time outside valid range")
    result=[value&0x7f]
    value >>= 7
    while value:
        result.insert(0,(value&0x7f)|0x80)
        value >>= 7
    return bytes(result)


def _midi_score(phrase:dict,root_midi:int)->bytes:
    """Portable SMF-0 melody with Pinyin syllable lyric meta events.

    120 BPM, 480 PPQ: 0.96 ticks/ms. This is a score to render later,
    *not* an independently synthesized DiffSinger teacher reference.
    """
    import struct
    events=[(0,0,b"\xff\x51\x03\x07\xa1\x20")] # 500000 us/beat
    position_ms=0
    for row in phrase["pairs"]:
        a,b=row["durations_ms"]
        for left,delta,dur in [
            (SYLLABLES[row['left']][0],row['semitones'][0],a),
            (RIGHT_SYLLABLES[row['right']],row['semitones'][1],b),
        ]:
            note=root_midi+delta
            if not 0<=note<=127:raise ValueError('MIDI pitch out of range')
            start=round(position_ms*0.96)
            end=round((position_ms+dur)*0.96)
            lyric=left.encode('ascii')
            events.append((start,1,b'\xff\x05'+_midi_vlq(len(lyric))+lyric))
            events.append((start,2,bytes([0x90,note,100])))
            events.append((end,0,bytes([0x80,note,0])))
            position_ms+=dur
        position_ms+=row['rest_ms']
    events.sort(key=lambda x:(x[0],x[1]))
    track=bytearray();prior=0
    for at,priority,body in events:
        track+=_midi_vlq(at-prior)+body
        prior=at
    track+=b'\x00\xff\x2f\x00'
    return b'MThd'+struct.pack('>IHHH',6,0,1,480)+b'MTrk'+struct.pack('>I',len(track))+bytes(track)


def _fade_edges(samples:np.ndarray,rate:int,ms:int=PAIR_EDGE_FADE_MS)->np.ndarray:
    arr=samples.copy();n=min(len(arr)//2,round(rate*ms/1000))
    if n>1:
        arr[:n]*=np.linspace(0,1,n)
        arr[-n:]*=np.linspace(1,0,n)
    return arr


def _join_pair(a:np.ndarray,b:np.ndarray,rate:int)->np.ndarray:
    n=round(MS_FADE_PAIRS*rate/1000)
    if n<2 or min(len(a),len(b))<n:raise ValueError("notes too short")
    ramp=np.linspace(0,np.pi/2,n)
    blend=a[-n:]*np.cos(ramp)+b[:n]*np.sin(ramp)
    return np.concatenate((a[:-n],blend,b[n:]))


def _one_phrase(bank:dict,starts:dict,lasts:dict,score:dict,
                policy:str, processed:callable) -> tuple[np.ndarray,list[dict]]:
    rate=bank["rate"]
    mapping=POLICIES[policy]
    sounds=[];ledger=[];cursor=0
    for event in score["pairs"]:
        left,right=event["left"],event["right"]
        donor=mapping[left]
        start=starts[left]
        vc=bank["donors"][(donor,right)]
        # Initial CV and variable VC donor form the first syllable. The second
        # syllable CV is identical across policy arms for the given context.
        initial=_join_pair(start,vc,rate)
        d1,d2=event["durations_ms"]
        f1=round((d1+MS_FADE_PAIRS)*rate/1000)
        f2=round(d2*rate/1000)
        a=processed(initial,rate,target_frames=f1,semitones=event["semitones"][0])
        b=processed(lasts[right],rate,target_frames=f2,semitones=event["semitones"][1])
        clip=_fade_edges(_join_pair(a,b,rate),rate)
        expected=round((d1+d2)*rate/1000)
        # Only 1 sample error permitted on each note grid.
        if abs(len(clip)-expected)>1:raise RuntimeError("render failed frozen note grid")
        ledger.append({"target":f"{SYLLABLES[left][0]} {RIGHT_SYLLABLES[right]}",
                       "left":left,"right":right,"selected_donor":donor,
                       "note_midi_offsets":event["semitones"],
                       "note_durations_ms":event["durations_ms"],
                       "pair_start_ms":round(cursor*1000/rate,3),
                       "pair_duration_ms":d1+d2,"rest_after_ms":event["rest_ms"]})
        sounds.append(clip)
        restframes=round(rate*event["rest_ms"]/1000)
        if restframes:
            sounds.append(np.zeros(restframes))
        cursor+=len(clip)+restframes
    return np.concatenate(sounds),ledger


def _validate_input(manifest:dict,score:dict,output:Path):
    if manifest.get("schema")!="phonoweave.diffsinger_vc_splice_input.v1" or manifest.get("source_provenance")!="author_confirmed_diffsinger_direct":
        raise ValueError("original author-confirmed DiffSinger manifest required")
    banks=manifest.get("banks")
    if not isinstance(banks,list) or not banks:raise ValueError("missing bank list")
    keys=[(v.get("style"),v.get("pitch")) for v in banks]
    if len(keys)!=len(set(keys)):raise ValueError("duplicate bank")
    for b in banks:
        if (not isinstance(b,dict) or not isinstance(b.get('style'),str) or
            not re.fullmatch(r'[a-z][a-z0-9_-]{0,30}',b['style']) or
            b.get('prefix')!=b.get('pitch')):
            raise ValueError("unsafe style or inconsistent prefix/pitch in source manifest")
    _frozen_validate(score)
    if output.exists():raise FileExistsError("output directory exists")
    for b in banks:
        source=Path(b["archive"]).expanduser().resolve(strict=True)
        if output==source or source in output.parents or output in source.parents:
            raise ValueError("source archive and output must not overwrite each other")
        _pitch_root(b["pitch"])


def render_note_phrase_ab(manifest:dict,score:dict,output:Path,
                          *,transform=rubberband_process) -> dict:
    """Write real note-scheduled PCM WAVs; no teacher target or test-quality claim."""
    output=Path(output).expanduser().resolve()
    _validate_input(manifest,score,output)
    payloads={};records=[];budgets=[];score_midis={}
    for cfg in manifest["banks"]:
        bank=_bank_data(cfg)
        starts,lasts,cvsources=_read_syllables(cfg,bank["rate"])
        bank_id=f"{bank['style']}_{bank['pitch']}"
        for phrase in score['phrases']:
            score_midis[f"score_midi/{bank_id}_{phrase['id']}.mid"] = _midi_score(
                phrase, _pitch_root(bank['pitch']))
        transform_cache = {}
        def cached_transform(signal,rate,*,target_frames,semitones):
            signature=(sha256(np.asarray(signal,dtype='<f4').tobytes()).hexdigest(),
                       rate,target_frames,semitones)
            if signature not in transform_cache:
                transform_cache[signature]=transform(signal,rate,
                    target_frames=target_frames,semitones=semitones)
            return transform_cache[signature]
        for policy,mapping in POLICIES.items():
            expected=2 if policy.startswith('k2_') else 3
            actual={r:set(mapping.values()) for r in RIGHTS}
            if any(len(v)!=expected for v in actual.values()):
                raise ValueError("policy violates registered inventory budget")
            budget=2*expected*BUDGET_UNIT_MS
            budgets.append({"bank":bank_id,"policy":policy,"VC_units":2*expected,
                            "VC_source_ms":budget,
                            "CV_initial_fixture_ms":len(LEFTS)*SOURCE_CV_MS,
                            "CV_second_fixture_ms":len(RIGHTS)*SOURCE_SECOND_MS,
                            "total_deduplicated_source_ms":budget+len(LEFTS)*SOURCE_CV_MS+len(RIGHTS)*SOURCE_SECOND_MS})
            for phrase in score["phrases"]:
                audio,ledger=_one_phrase(bank,starts,lasts,phrase,policy,cached_transform)
                encoded=_wav_bytes(audio,bank["rate"])
                rel=f"{bank_id}/{policy}/{phrase['id']}.wav"
                payloads[rel]=encoded
                records.append({
                    "bank":bank_id,"style":bank['style'],"pitch":bank['pitch'],
                    "root_pitch_midi_assumed":_pitch_root(bank['pitch']),
                    "policy":policy,"score_phrase":phrase['id'],"score_sha256":_hash_json(score),
                    "target_note_count":2*len(phrase['pairs']),
                    "actual_milliseconds":round(len(audio)*1000/bank['rate'],3),
                    "source_archive_sha256":bank['archive_sha256'],
                    "source_oto_sha256":bank['oto_sha256'],
                    "selected_vc_source_entries":{
                        f"{left} {right}":bank['donor_sources'][(mapping[left],right)]
                        for left,right in ((e['left'],e['right']) for e in phrase['pairs'])
                    },
                    "source_CV_aliases":{k:v['original_alias'] for k,v in cvsources.items()},
                    "note_events":ledger,"rendered_wav":rel,"wav_sha256":sha256(encoded).hexdigest(),
                    "test_target_was_independently_generated":False,
                    "teacher_quality_scored":False,
                })
    if len({r['bank'] for r in records})!=len(manifest['banks']):
        raise RuntimeError("bank render count differs from planned")
    report={
        "schema":OUTPUT_SCHEMA,
        "status":"actual_rubberband_note_phrase_preview_not_quality_benchmark",
        "frozen_score":score,"score_sha256":_hash_json(score),
        "source_provenance":manifest['source_provenance'],
        "engine":("ffmpeg_rubberband_pitch_and_time_formant_preserved_not_OpenUtau"
                   if transform is rubberband_process else "injected_test_transform_not_external_engine"),
        "backend_version": (
            subprocess.run(['ffmpeg','-version'],capture_output=True,text=True,timeout=10)
                      .stdout.splitlines()[0] if transform is rubberband_process else 'test_shim'),
        "source_units":"actual_source_ZIP_PCM_OTO_relative_unreviewed_except_D4",
        "budget_ledger":budgets,"renders":records,
        "score_midi_files":sorted(score_midis),
        "experiment_gates":{
            "real_source_pcm":True,"multiple_notes_timed_and_pitched":True,
            "same_engine_and_score_for_every_policy":True,
            "all_K2_vc_budgets_matched":True,
            "K3_reference_budget_matched":False,
            "independent_DiffSinger_teacher_reference_present":False,
            "heldout_syllable_sequences_verified":False,
            "OpenUtau_resampler_or_wavtool_verified":False,
            "listener_quality_scores_present":False,
            "split_share_quality_gain_established":False,
        },
        "limitations":[
            "Two-syllable note-pair concatenation with six-note mock phrases is not an OpenUtau score and not a claim of natural singing fidelity.",
            "Neither genuine independent teacher vocal targets nor their separately generated DiffSinger WAVs are available.",
            "All note units come from the same candidate-source inventory; the phrase is not held-out source audio.",
            "Initial CV, VC and following CV are OTO-relative, not independently human-adjusted across subbanks.",
            "Pitch reference is the subbank label, not an independently measured F0; Rubber Band may affect frication.",
            "Budget counts unique preselected VC donors, not replicated note waveform duration; full K3 cannot enter K2 matched comparisons.",
            "The same voicebank has two voice styles. This does not test independent speakers, model checkpoints or natural singing generalization.",
        ],
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pw-note-phrase-',dir=output.parent) as d:
        stage=Path(d)/'render';stage.mkdir()
        for rel,binary in {**payloads,**score_midis}.items():
            path=stage/rel;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(binary)
        (stage/'render_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (stage/'frozen_score.json').write_text(json.dumps(score,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (stage/'teacher_reference_REQUIRED.json').write_text(json.dumps({
            "schema":TEACHER_SCHEMA,"source_score_sha256":_hash_json(score),
            "status":"required_unprovided_not_scored",
            "requirements":"Separately render the exact frozen score in DiffSinger, same voice style and target pitches, with no reuse of source ZIP vocal waveforms. Record model checkpoint, score, phoneme durations, F0 and output WAV hash.",
            "reference_wavs":[],
            "frozen_score_midis":sorted(score_midis),
        },ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (stage/'README.txt').write_text(
            'Real FFmpeg Rubber Band note-conditioned six-note phrase previews.\n'
            'Source = user-supplied DiffSinger CVVC subbank WAV, not generated from an external TTS model here.\n'
            'Compare K2 policies ONLY within the same voice style and phrase.\n'
            'K3 is a higher-cost qualitative ceiling, never a matched-budget competitor.\n'
            'Rubber Band pitch/time DSP is real, but is NOT an OpenUtau-compatible resampler/wavtool.\n'
            'A separate DiffSinger reference matching frozen_score.json is REQUIRED for any quality claim.\n'
            'No ground truth objective metrics, listener scores, or automatic optimal-sharing claims.\n',encoding='utf-8')
        if output.exists():raise FileExistsError("output appeared concurrently")
        os.rename(stage,output)
    return report


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description='Frozen DiffSinger note-conditioned VC phrase preview')
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--score',type=Path,help='optional predeclared score; default frozen 6-context pilot')
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args(argv)
    try:
        manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
        score=json.loads(args.score.read_text(encoding='utf-8')) if args.score else frozen_score()
        result=render_note_phrase_ab(manifest,score,args.output)
    except (OSError,ValueError,RuntimeError,json.JSONDecodeError,zipfile.BadZipFile) as exc:
        p.error(str(exc))
    print(f"Rendered {len(result['renders'])} note-timed WAV previews. No independent teacher quality claim.")
    return 0


if __name__=='__main__':
    raise SystemExit(main())
