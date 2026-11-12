"""v4.3: audible, source-traceable marginal-value preview of EXISTING VC units.

A: SAME left-CV vowel and right-CV vowel, inserting current VC transition.
B: SAME left-CV vowel and right-CV vowel, inserting CV consonant instead.
This is a controlled short audio-splice AB, NOT singing synthesis, a validated
recording-budget result, or a reason to remove an existing alias. VC aliases
select AB targets only, NEVER determine phonemic/allophone categories.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
import wave

import numpy as np

from .audio import AudioReadError, read_wav
from .oto import OtoEntry, load_voicebank
from .cv_oto_allophones_v41 import _cv_from_entry, NOTE_PREFIX
from .mandarin import _BASES, _FINALS
from .source_aware_recording_v17 import pcm_identity

SCHEMA="phonoweave.vc_ab_cases.v1"
RESULT="phonoweave.vc_marginal_value_audio_preview.v4.3"
VC_RE=re.compile(r"^([a-züv]+)\s+([a-züv]+)$")


@dataclass(frozen=True)
class Case:
    identifier:str
    bank:str
    preceding_vowel:str
    next_consonant:str
    left:OtoEntry
    vc:OtoEntry
    right:OtoEntry


def _bank(root:Path,entry:OtoEntry)->str:
    parent=entry.oto_path.parent.resolve()
    if not parent.is_relative_to(root):
        raise ValueError("OTO outside bank")
    return parent.relative_to(root).as_posix() or "."


def _allowed(root:Path,entry:OtoEntry)->bool:
    return (entry.oto_path.resolve().is_relative_to(root)
            and entry.wav_path.resolve().is_relative_to(root)
            and entry.wav_path.is_file())


def _item_by_alias(index,bank,alias)->OtoEntry:
    options=index.get((bank,alias),[])
    if len(options)!=1:
        raise ValueError(f"alias not unique/present: bank={bank} alias={alias!r}")
    return options[0]


def explicit_cases(root:Path,path:Path,entries:list[OtoEntry])->list[Case]:
    obj=json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(obj,dict) or set(obj)!={"schema","cases"} or obj["schema"]!=SCHEMA:
        raise ValueError("requires vc_ab_cases.v1 schema and cases")
    if not isinstance(obj["cases"],list) or not obj["cases"]:
        raise ValueError("empty cases")
    index=defaultdict(list)
    for e in entries:
        if _allowed(root,e):
            index[(_bank(root,e),e.alias)].append(e)
    allowed={"id","bank","preceding_vowel","next_consonant",
             "left_cv_alias","vc_alias","right_cv_alias"}
    seen=set()
    out=[]
    for spec in obj["cases"]:
        if not isinstance(spec,dict) or set(spec)!=allowed:
            raise ValueError("invalid explicit A/B case columns")
        for key in allowed:
            if not isinstance(spec[key],str) or not spec[key] or any(ord(c)<32 for c in spec[key]):
                raise ValueError("invalid explicit A/B case text")
        ident=spec["id"]
        if ident in seen:
            raise ValueError("duplicate A/B case id")
        seen.add(ident)
        bank=spec["bank"]
        out.append(Case(ident,bank,spec["preceding_vowel"],spec["next_consonant"],
                        _item_by_alias(index,bank,spec["left_cv_alias"]),
                        _item_by_alias(index,bank,spec["vc_alias"]),
                        _item_by_alias(index,bank,spec["right_cv_alias"])))
    return out


def automatic_cases(root:Path,entries:list[OtoEntry],max_pairs:int,
                    consonant:str|None=None)->tuple[list[Case],dict]:
    """Mandarin-only convenience matcher: existing VC names locate test objects.

    Full CV phonemes are derived ONLY from CV rows; a VC name is NEVER used
    as acoustic subtype or as confirmation the unit is valuable.
    """
    if not (1<=max_pairs<=50):
        raise ValueError("max_pairs must be 1..50")
    cv=defaultdict(list)
    vc=defaultdict(list)
    stats=Counter()
    for e in entries:
        if not _allowed(root,e):
            continue
        bank=_bank(root,e)
        match=_cv_from_entry(e.alias,lexicon=None,prefixes=[],adapter="mandarin")
        if match is not None:
            token,c,v=match
            cv[(bank,c,v)].append(e)
            continue
        candidate=NOTE_PREFIX.sub("",e.alias.strip()).strip().lower()
        hit=VC_RE.fullmatch(candidate)
        if hit is None:
            continue
        vowel,c=hit.groups()
        if vowel not in _FINALS or c not in _BASES:
            continue
        if consonant is not None and consonant!=c:
            continue
        vc[(bank,vowel,c)].append(e)
    # Group left CV available by vowel and bank, right CV by C and bank
    left=defaultdict(list)
    right=defaultdict(list)
    for (bank,c,v),all_entries in cv.items():
        for e in all_entries:
            left[(bank,v)].append((c,e))
            right[(bank,c)].append((v,e))
    possible=[]
    for (bank,vowel,c),vc_entries in sorted(vc.items()):
        lefts=sorted(left.get((bank,vowel),[]),key=lambda x:(x[0],x[1].alias))
        rights=sorted(right.get((bank,c),[]),key=lambda x:(x[0]!=vowel,x[0],x[1].alias))
        if not lefts or not rights:
            stats["vc_no_matching_CV_donors"]+=1
            continue
        left_entries=[e for label,e in lefts if label!=c] or [e for _,e in lefts]
        for e in vc_entries:
            l=next((x for x in left_entries if x.wav_path!=e.wav_path),left_entries[0])
            r=next((x for v,x in rights if x.wav_path!=e.wav_path),rights[0][1])
            if len({(x.oto_path,x.line_number) for x in (l,e,r)})<3:
                continue
            possible.append((vowel,c,Case(
                f"{c}_{vowel}_{bank}_{e.line_number}",bank,vowel,c,l,e,r)))
    # Spread the limited set over consonants before taking second examples.
    by_c=defaultdict(list)
    for vowel,c,case in possible:
        by_c[c].append((vowel,case))
    selected=[]
    for c in sorted(by_c):
        by_c[c].sort(key=lambda x:sha256((x[0]+"|"+x[1].bank+"|"+x[1].identifier).encode()).hexdigest())
    while len(selected)<max_pairs and any(by_c.values()):
        for c in sorted(by_c):
            if by_c[c] and len(selected)<max_pairs:
                selected.append(by_c[c].pop(0)[1])
    return selected,{
        "eligible_existing_VC_candidate_groups":len(vc),
        "possible_matched_cases":len(possible),
        "auto_selected_cases":len(selected),
        "skipped":dict(sorted(stats.items())),
        "VC_alias_used_only_for_test_case_selection":True,
    }


def _cut(entry:OtoEntry,role:str):
    """OTO approximate timings: offset, preutterance, cutoff — same parser both arms."""
    audio,rate=read_wav(entry.wav_path)
    if rate<16000:
        raise ValueError("unsupported WAV rate")
    prior=entry.offset+entry.preutterance
    duration_ms=len(audio)*1000/rate
    limit=(entry.offset-entry.cutoff if entry.cutoff<0 else duration_ms-entry.cutoff)
    if role=="left_vowel":
        a=prior+24
        b=min(prior+225,limit-12)
        # Use later, sustained portion of this same independent CV source.
        a=max(a,b-155)
        min_ms=70
    elif role=="right_vowel":
        a=prior+20
        b=min(prior+210,limit-12)
        min_ms=70
    elif role=="vc_transition":
        a=entry.offset+8
        b=prior-8
        min_ms=48
    elif role=="cv_consonant":
        a=entry.offset+8
        b=prior-8
        min_ms=48
    else:
        raise ValueError("invalid extraction role")
    if not (np.isfinite(a) and np.isfinite(b)) or b-a<min_ms:
        raise ValueError("unusable OTO timing for "+role)
    begin=max(0,int(round(a*rate/1000)))
    end=min(len(audio),int(round(b*rate/1000)))
    if end-begin<int(round(min_ms*rate/1000)):
        raise ValueError("WAV segment out of bounds")
    clip=audio[begin:end].astype(np.float64,copy=False)
    if float(np.sqrt(np.mean(clip*clip)))<.0007:
        raise ValueError("silent extracted segment")
    return clip,rate,(a,b)


def _crossfade(a:np.ndarray,b:np.ndarray,rate:int,ms:float=8)->np.ndarray:
    n=min(len(a),len(b),int(round(rate*ms/1000)))
    if n<2:
        raise ValueError("too-short crossfade")
    weight=np.linspace(0,1,n)
    return np.r_[a[:-n],a[-n:]*(1-weight)+b[:n]*weight,b[n:]]


def _preview(left:np.ndarray,transition:np.ndarray,right:np.ndarray,rate:int):
    out=_crossfade(_crossfade(left,transition,rate),right,rate)
    pad=np.zeros(int(rate*.09))
    return np.r_[pad,out,pad]


def _write_wav(path:Path,x:np.ndarray,rate:int):
    peak=float(np.max(np.abs(x)))
    if peak>1.0:
        x=x/peak*.98
    pcm=(np.clip(x,-1,1)*32767).astype("<i2")
    with path.open("xb") as fd:
        with wave.open(fd,"wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(pcm.tobytes())


def _overlap(a:OtoEntry,ra:tuple[float,float],
             b:OtoEntry,rb:tuple[float,float])->bool:
    return (a.wav_path.resolve()==b.wav_path.resolve()
            and min(ra[1],rb[1])-max(ra[0],rb[0])>1)


def _prepare(case:Case,root:Path):
    for e in (case.left,case.vc,case.right):
        if not _allowed(root,e):
            raise ValueError("WAV or OTO moved outside voicebank")
    lv,lr,lvspan=_cut(case.left,"left_vowel")
    tr,vr,trspan=_cut(case.vc,"vc_transition")
    cr,rr,crspan=_cut(case.right,"cv_consonant")
    rv,vrv,rvspan=_cut(case.right,"right_vowel")
    if len({lr,vr,rr,vrv})!=1:
        raise ValueError("source sample rates differ")
    if (_overlap(case.left,lvspan,case.vc,trspan)
            or _overlap(case.right,crspan,case.vc,trspan)
            or _overlap(case.left,lvspan,case.right,crspan)):
        raise ValueError("source regions overlap; cannot interpret VC ablation")
    # Matched time budget: keep the final consonant-region samples, near V onset.
    # NO time-stretch/resampling or pitch change.
    target_length=min(len(tr),len(cr),int(round(lr*.20)))
    if target_length<int(round(lr*.048)):
        raise ValueError("insufficient matched consonant length")
    tr=tr[-target_length:]
    cr=cr[-target_length:]
    a=_preview(lv,tr,rv,lr)
    b=_preview(lv,cr,rv,lr)
    assert len(a)==len(b)
    assert all(np.isfinite(z).all() for z in (a,b))
    return a,b,lr,{
        "left_CV":{"wav":case.left.wav_path.relative_to(root).as_posix(),
                   "alias":case.left.alias,"oto_line":case.left.line_number},
        "existing_VC":{"wav":case.vc.wav_path.relative_to(root).as_posix(),
                       "alias":case.vc.alias,"oto_line":case.vc.line_number},
        "right_CV":{"wav":case.right.wav_path.relative_to(root).as_posix(),
                    "alias":case.right.alias,"oto_line":case.right.line_number},
        "source_pcm_sha256":{
            "left":pcm_identity(case.left.wav_path),
            "vc":pcm_identity(case.vc.wav_path),
            "right":pcm_identity(case.right.wav_path),
        },
        "transition_ms":round(target_length*1000/lr,2),
        "common_CV_contexts":True,
        "same_audio_sample_rate":lr,
        "time_stretching_used":False,
        "same_source_overlap":False,
    }


def run(root:Path,output:Path,*,max_pairs:int=12,
        consonant:str|None=None,cases_file:Path|None=None)->dict:
    root=Path(root).expanduser().resolve(strict=True)
    output=Path(output).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("voicebank root is not a directory")
    if output==root or output.is_relative_to(root):
        raise ValueError("output must be outside voicebank")
    if output.exists():
        raise FileExistsError("refusing overwrite")
    entries,warnings=load_voicebank(root)
    if cases_file is not None:
        cases=explicit_cases(root,cases_file,entries)
        selection={"mode":"explicit_author_triplets"}
    else:
        cases,selection=automatic_cases(root,entries,max_pairs,consonant=consonant)
        selection["mode"]="Mandarin_auto_existing_VC_objects_only"
    accepted=[]
    excluded=Counter()
    # Prepare ALL pairs before creating output; do not leave misleading partial export
    for case in cases:
        try:
            a,b,rate,sources=_prepare(case,root)
        except (ValueError,AudioReadError,OSError,EOFError,KeyError,wave.Error) as e:
            excluded[str(e)[:90]]+=1
            continue
        accepted.append((case,a,b,rate,sources))
    if not accepted:
        raise ValueError("no eligible VC vs CV contrasts (check OTO timing/source overlap)")
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pw-vc-audio-",dir=output.parent) as temp:
        stage=Path(temp)/"output"
        stage.mkdir()
        clips=stage/"blind_audio"
        clips.mkdir()
        human=[]
        private=[]
        for index,(case,a,b,rate,sources) in enumerate(accepted,1):
            name=f"pair_{index:03d}"
            # Deterministically counterbalance which anonymous label contains VC.
            vc_is_A=int(sha256(case.identifier.encode("utf-8")).hexdigest()[-1],16)%2==0
            samples_A=a if vc_is_A else b
            samples_B=b if vc_is_A else a
            file_a=f"{name}_A.wav"
            file_b=f"{name}_B.wav"
            _write_wav(clips/file_a,samples_A,rate)
            _write_wav(clips/file_b,samples_B,rate)
            human.append({"pair_id":name,"A_wav":"blind_audio/"+file_a,
                          "B_wav":"blind_audio/"+file_b,
                          "vowel":case.preceding_vowel,"consonant":case.next_consonant,
                          "preferred":"", "notes":""})
            private.append({"pair_id":name,"bank":case.bank,
                            "case_id":case.identifier,
                            "vc_condition":"A" if vc_is_A else "B",
                            "cv_only_condition":"B" if vc_is_A else "A",
                            "sources":sources})
        with (stage/"listening_sheet.csv").open("x",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=human[0].keys())
            writer.writeheader()
            for row in human:
                # Avoid spreadsheet CSV injection through author names.
                writer.writerow({k:("'"+v if isinstance(v,str) and v.lstrip().startswith(("=","+","-","@")) else v)
                                 for k,v in row.items()})
        (stage/"private_condition_key.json").write_text(json.dumps({
            "schema":RESULT,"type":"NOT_REAL_SINGING_RENDERING",
            "selection":selection,"oto_parse_warnings":len(warnings),
            "selected_case_count":len(cases),"audio_pairs_created":len(private),
            "excluded":dict(sorted(excluded.items())),
            "conditions":private,
            "controls":{
                "common_left_vowel_audio":True,
                "common_right_vowel_audio":True,
                "identical_middle_segment_duration":True,
                "same_number_of_crossfades":2,
                "no_pitch_shift_or_time_stretch":True,
                "uses_existing_VC_audio_only_in_one_condition":True,
                "not_full_renderer_or_recording_budget_equivalence":True,
            },
            "interpretation":"Audible splice audition ONLY. VC labels select units under test; they are NOT allophone ground truth. Human ratings need blinding, multiple source examples and eventual song rendering; do not delete any unit.",
        },ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        os.rename(stage,output)
    return {"status":"AUDIO_PREVIEW_ONLY",
            "selected_cases":len(cases),
            "audible_blind_pairs":len(private),
            "excluded":dict(sorted(excluded.items())),
            "output":str(output),
            "no_VC_removal_approved":True}


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description="Existing-VC optional unit A/B using real source WAV; not synthesized singing")
    p.add_argument("voicebank",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-pairs",type=int,default=12)
    p.add_argument("--consonant",type=str)
    p.add_argument("--cases",type=Path,help="Author-defined exact alias triplets (language independent)")
    args=p.parse_args(argv)
    try:
        report=run(args.voicebank,args.output,max_pairs=args.max_pairs,
                   consonant=args.consonant,cases_file=args.cases)
    except (ValueError,OSError,KeyError,TypeError,json.JSONDecodeError) as e:
        p.error(str(e))
    print(json.dumps(report,ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
