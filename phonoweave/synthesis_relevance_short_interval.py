"""Exploratory short-OTO-interval separability, with matched-population controls.

The OTO offset/preutterance interval is NOT a validated frication segment.
A shorter eligible interval is not automatically evidence of a better sample.
All input WAVs and OTO files are read-only, and no audio is generated.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .audio import AudioReadError, read_wav
from .mandarin import collect_observations, context_for
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .synthesis_relevance_acoustic_separability import (
    Observation, _features as legacy_features, grouped_cv, paired_wav_control,
)
from .synthesis_relevance_validity_audit import leave_one_subbank_out, wav_group_label_null

SCHEMA = "phonoweave.synthesis_relevance.short_interval.v0.7"
MIN_MS = 40.0
EDGE_FRACTION = 0.16
MAX_EDGE_MS = 8.0
FRAME_MS = 12.0
EPS = 1e-15


def _spectrum_features(x: np.ndarray, rate: int) -> tuple[float,float,float]:
    x = (x.astype(float) - float(np.mean(x))) * np.hanning(len(x))
    power = np.abs(np.fft.rfft(x))**2
    hz = np.fft.rfftfreq(len(x), 1.0/rate)
    band = (hz>=1500) & (hz<=min(11000,0.46*rate))
    if np.count_nonzero(band)<4 or float(power[band].sum())<=1e-12:
        raise ValueError("insufficient_fricative_band_power")
    high = (hz>=5000) & band
    p = power[band]
    cog = float(np.dot(hz[band],p)/p.sum()/1000)
    high_ratio = float(power[high].sum()/p.sum())
    flatness = float(np.exp(np.mean(np.log(p+EPS)))/(np.mean(p)+EPS))
    return cog,high_ratio,flatness


def short_features(samples:np.ndarray,rate:int,start_ms:float,end_ms:float,
                   *, region:str="interior") -> tuple[float,...]:
    """Fixed 12ms frames; comparable nine-dimensional vector for 40ms+ OTO marks.

    Interior/early/late are position controls within one OTO interval;
    no feature includes samples beyond the given OTO interval.
    """
    if region not in ("interior","early","late"):
        raise ValueError("unknown region")
    if not np.isfinite(start_ms) or not np.isfinite(end_ms):
        raise ValueError("nonfinite_oto_interval")
    duration=end_ms-start_ms
    if duration<MIN_MS:
        raise ValueError("shorter_than_40ms")
    if rate<16000:
        raise ValueError("sample_rate_below_16khz")
    trim=min(MAX_EDGE_MS,EDGE_FRACTION*duration)
    valid_start=start_ms+trim
    valid_end=end_ms-trim
    half=FRAME_MS/2
    if valid_end-valid_start < FRAME_MS:
        raise ValueError("not_enough_interior_signal")
    if region=="interior":
        positions=np.array([0.15,0.5,0.85])
    elif region=="early":
        positions=np.array([0.0,0.2,0.4])
    else:
        positions=np.array([0.6,0.8,1.0])
    centers=(valid_start+half)+(valid_end-valid_start-FRAME_MS)*positions
    result=[]
    for center in centers:
        lo=int(round((center-half)*rate/1000))
        hi=int(round((center+half)*rate/1000))
        if lo<0 or hi>len(samples) or hi-lo<128:
            raise ValueError("window_outside_wav")
        result.extend(_spectrum_features(samples[lo:hi],rate))
    if not all(np.isfinite(result)):
        raise ValueError("nonfinite_features")
    return tuple(float(v) for v in result)


def _event_iterator(root:Path,base:str):
    entries,_=load_voicebank(root)
    observed=collect_observations(
        (e for e in entries if e.wav_path.is_file()),
        affix_pairs(load_prefix_maps(root)),
    )
    for obs in observed:
        if obs.base_unit==base:
            context=context_for(base,obs.final)
            if context in ("plain","rounded"):
                yield obs.entry,context


def extract(root:Path,*,base:str="sh") -> tuple[dict[str,list[Observation]],dict]:
    if base not in ("sh","s","x"):
        raise ValueError("unsupported base")
    root=root.expanduser().resolve(strict=True)
    cache={}
    banks={k:{} for k in ("short_interior","short_early","short_late","legacy")}
    reason=Counter()
    context_counts=defaultdict(Counter)
    duration_bins=Counter()
    all_physical=set()
    for e,context in _event_iterator(root,base):
        wav=e.wav_path.resolve()
        subbank=str(e.oto_path.parent.relative_to(root))
        duration=float(e.preutterance)
        bucket=("<40" if duration<40 else "40-80" if duration<80 else
                "80-110" if duration<110 else "110+")
        duration_bins[bucket]+=1
        key=(str(wav),round(e.offset+e.preutterance,1),context)
        # All candidate events are deduplicated before quality assessment.
        if key in all_physical:
            reason["duplicate_OTO_event"]+=1
            continue
        all_physical.add(key)
        if not np.isfinite(duration) or duration<MIN_MS:
            reason["shorter_than_40ms"]+=1
            context_counts[context]["shorter_than_40ms"]+=1
            continue
        try:
            if wav not in cache:
                cache[wav]=read_wav(wav)
            samples,rate=cache[wav]
            uid=sha256(str(wav.relative_to(root)).encode()).hexdigest()[:24]
            for region in ("interior","early","late"):
                name="short_"+region
                try:
                    feature=short_features(samples,rate,e.offset,e.offset+duration,region=region)
                except (AudioReadError,ValueError,OverflowError) as exc:
                    reason[name+"_invalid:"+str(exc)[:80]]+=1
                    continue
                banks[name][key]=Observation(uid,subbank,context,feature)
            if duration>=110:
                try:
                    feature=legacy_features(samples,rate,e.offset,e.offset+duration)
                    banks["legacy"][key]=Observation(uid,subbank,context,feature)
                except (AudioReadError,ValueError,OverflowError):
                    reason["legacy_invalid"]+=1
            if key in banks["short_interior"]:
                context_counts[context]["accepted"]+=1
        except (AudioReadError,OSError,ValueError,OverflowError):
            reason["wav_decode_failure"]+=1
    results={k:list(v.values()) for k,v in banks.items()}
    return results,{
        "unique_candidate_events":len(all_physical),
        "valid_events_by_protocol":{k:len(v) for k,v in banks.items()},
        "valid_WAVs_by_protocol":{k:len({o.wav_id for o in v.values()}) for k,v in results.items()},
        "duration_bins":dict(duration_bins),
        "rejection_reasons":dict(reason),
        "per_context":{k:dict(v) for k,v in context_counts.items()},
    }


def evaluate_bank(root:Path,*,base:str="sh",permutations:int=199)->dict:
    records,coverage=extract(root,base=base)
    old=records["legacy"]
    new=records["short_interior"]
    old_by_key={}  # Matching is defined by shared source WAV, subbank, context AND vector occurrence.
    # Primary matched experiment uses intersection of physical WAV/context pairs,
    # aggregated to one vector per WAV/context to avoid alias multiplicity effects.
    def aggregate(rows):
        groups=defaultdict(list)
        for r in rows:
            groups[(r.wav_id,r.subbank,r.context)].append(r.features)
        return {k:Observation(k[0],k[1],k[2],tuple(np.mean(v,axis=0)))
                for k,v in groups.items()}
    old_map=aggregate(old)
    new_map=aggregate(new)
    keys=sorted(set(old_map)&set(new_map))
    legacy_common=[old_map[k] for k in keys]
    short_common=[new_map[k] for k in keys]
    models={}
    for name,rows in records.items():
        cv=grouped_cv(rows)
        models[name]={
            "events":len(rows),
            "physical_wavs":len({r.wav_id for r in rows}),
            "grouped_cv":cv,
            "paired_within_WAV":paired_wav_control(rows,permutations=499),
            "leave_one_subbank_out":leave_one_subbank_out(rows),
        }
    matched={
        "WAV_context_cells":len(keys),
        "legacy_grouped_cv":grouped_cv(legacy_common),
        "short_grouped_cv":grouped_cv(short_common),
        "warning":"Old and new models use different frame lengths; match is WAV/context, not exact alias timing.",
    }
    null=wav_group_label_null(new,permutations=permutations)
    return {
        "coverage":coverage,
        "protocols":models,
        "matched_legacy_vs_short":matched,
        "new_short_label_null":null,
        "notes":[
            "OTO preutterance is a timing-control proxy; feature regions are not phonetically verified frication.",
            "40ms+ eligibility increases coverage but may include neighboring sounds or selection biases.",
            "Paired events from the same WAV are not independent speakers.",
            "This does not prove that separate synthesis units improve generated singing.",
        ]
    }


def run_manifest(manifest:Path,output:Path,*,base:str="sh",permutations:int=199)->dict:
    data=json.loads(manifest.expanduser().resolve(strict=True).read_text("utf-8"))
    if data.get("schema")!="phonoweave.voicebank_replication_manifest.v1":
        raise ValueError("invalid replication manifest")
    results=[]
    for bank in data["voicebanks"]:
        row=evaluate_bank(Path(bank["path"]),base=base,permutations=permutations)
        row["bank_id"]=bank["bank_id"]
        row["speaker_group_id"]=bank["speaker_group_id"]
        results.append(row)
    report={
        "schema":SCHEMA,"base_unit":base,"banks":results,
        "number_of_banks":len(results),
        "independent_speakers":len({b["speaker_group_id"] for b in results}),
        "no_audio_written":True,"production_decisions_modified":False,
    }
    target=output.expanduser().resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n","utf-8")
    return report
