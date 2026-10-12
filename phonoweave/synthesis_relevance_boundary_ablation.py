"""Stage blinded OTO-relative donor-cut ablation; does not render OpenUtau.

All conditions share the identical target-right segment; donor identity is
fixed within each context arm. Relative donor cut positions are OTO -40,
-20 and 0 ms. These are NOT verified phonetic consonant endings.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np

from .audio import read_wav
from .synthesis_relevance_cv_pilot_v02 import _extract
from .synthesis_relevance_ab_staging import (
    _slice, _assemble, _write_mono_pcm16, _file_digest, RIGHT_MS,
    CROSSFADE_MS, MAX_LEFT_MS,
)

VERSION="phonoweave.synthesis_relevance.boundary_ablation.v0.4"
OFFSETS_MS=(-40.,-20.,0.)

def stage_boundary_ablation(source:Path, old_singer:Path, output:Path, *,
                           seed:int=20261011)->dict:
    source=source.expanduser().resolve(strict=True)
    old_singer=old_singer.expanduser().resolve(strict=True)
    output=output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    key=output.parent/(output.name+"_REVEAL.json")
    if key.exists():
        raise FileExistsError(f"Reveal key already exists: {key}")
    if source==output or source in output.parents or old_singer in output.parents:
        raise ValueError("Output must not be within source or prior singer")
    record=json.loads((old_singer/"experiment.json").read_text("utf-8"))
    if record.get("schema")!="phonoweave.synthesis_relevance.rendered_ab_prep.v0.3":
        raise ValueError("Requires previous v0.3 singer provenance")
    base=record["target_alias"]
    # The original target ID is stable across pilot and staging.
    pilot_base="sh"
    events,_=_extract(source,pilot_base)
    by_id={e.identity:e for e in events}
    target=by_id[record["pilot_event_id"]]
    same=by_id[record["donors"]["matched"]["event_id"]]
    cross=by_id[record["donors"]["cross"]["event_id"]]
    if len({target.source_wav,same.source_wav,cross.source_wav})!=3:
        raise ValueError("Three distinct physical WAVs required")
    if target.context==cross.context or target.context!=same.context:
        raise ValueError("Unexpected donor contexts")
    if len({e.subbank for e in (target,same,cross)})!=1:
        raise ValueError("Pitch-subbank mismatch")
    if len({e.sample_rate for e in (target,same,cross)})!=1:
        raise ValueError("Sample-rate mismatch")
    rate=target.sample_rate
    audio={}
    for e in (target,same,cross):
        p=(source/e.source_wav).resolve(strict=True)
        if source not in p.parents:
            raise ValueError("Source WAV escapes singer root")
        audio[e.identity]=read_wav(p)[0]
    right=_slice(audio[target.identity],rate,target.oto_alignment_ms,0,RIGHT_MS)
    # A fixed left-hand window length across conditions controls note timing.
    left_ms=min(MAX_LEFT_MS,*(e.oto_preutterance_ms for e in (target,same,cross))) - max(abs(v) for v in OFFSETS_MS)
    if left_ms<35:
        raise ValueError("Insufficient donor pre-boundary data for cut sensitivity")
    conditions={}
    for group,donor in (("same",same),("cross",cross)):
        for shift in OFFSETS_MS:
            left=_slice(audio[donor.identity],rate,
                        donor.oto_alignment_ms+shift,left_ms,0)
            joined,pre=_assemble(left,right,rate)
            conditions[(group,shift)]=(joined,pre)
    # Unspliced source reference only for orientation: not an experimental arm.
    reference=_slice(audio[target.identity],rate,
                     target.oto_alignment_ms,left_ms,RIGHT_MS)
    maximum=max(np.max(np.abs(x[0])) for x in conditions.values())
    maximum=max(maximum,np.max(np.abs(reference)))
    gain=min(1.,0.95/maximum) if maximum>0 else 1.
    identities=sorted(conditions)
    aliases=[f"PW_T{i:02d}" for i in range(1,len(identities)+1)]
    rng=random.Random(seed)
    rng.shuffle(aliases)
    # Create files in a fresh directory; the mapping key remains OUTSIDE it.
    output.mkdir(parents=True,exist_ok=False)
    oto=[]
    blind=[]
    for identity,alias in zip(identities,aliases):
        samples,pre=conditions[identity]
        filename=alias+".wav"
        _write_mono_pcm16(output/filename,samples*gain,rate)
        oto.append(f"{filename}={alias},0,{pre:.3f},0,{pre:.3f},{pre/3:.3f}")
        blind.append({
            "alias":alias,"condition":identity[0],"cut_shift_ms":identity[1],
            "source_event_id":(same if identity[0]=="same" else cross).identity,
            "source_wav":(same if identity[0]=="same" else cross).source_wav,
            "sha256":_file_digest(output/filename)
        })
    _write_mono_pcm16(output/"PW_REF.wav",reference*gain,rate)
    pre=conditions[identities[0]][1]
    oto.append(f"PW_REF.wav=PW_REF,0,{pre:.3f},0,{pre:.3f},{pre/3:.3f}")
    (output/"oto.ini").write_text("\n".join(oto)+"\n",encoding="utf-8")
    (output/"character.txt").write_text("name=PW boundary ablation pilot\n",encoding="utf-8")
    public={
        "schema":VERSION,"alias_list":sorted(aliases),
        "reference_alias":"PW_REF","sample_rate":rate,
        "fixed_left_ms":round(left_ms,3),"fixed_target_right_ms":RIGHT_MS,
        "fixed_crossfade_ms":CROSSFADE_MS,
        "all_donor_cuts_relative_to_OTO":list(OFFSETS_MS),
        "source_singer_modified":False,
        "warning":"OTO cut shifts are not verified consonant/vowel boundaries; use manual boundary review before inference.",
        "blind_conditions_not_in_public_manifest":True,
    }
    (output/"experiment.json").write_text(
        json.dumps(public,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    key.write_text(json.dumps({
        "schema":VERSION,"target_event_id":target.identity,
        "target_wav":target.source_wav,"target_alias":target.alias,
        "same_donor":same.source_wav,"cross_donor":cross.source_wav,
        "stimuli":blind,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {"output":str(output),"reveal_key":str(key),
            "aliases":sorted(aliases),"reference_alias":"PW_REF","left_ms":left_ms}
