"""Read-only objective donor substitution cost for provisional OTO consonant units.

This is an offline *acoustic proxy*, not actual OpenUtau rendered quality.
All comparisons use target-held-out actual core as reference, fixed target
right boundary and donors matched within the same pitch subbank. No audio
files are written and source voicebanks are never modified.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from .audio import AudioReadError, read_wav
from .mandarin import collect_observations, context_for
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps

SCHEMA="phonoweave.synthesis_relevance.unit_merging_cost.v0.8"
SEED=20261012
WINDOW_MS=20.0
MIN_PREUTTERANCE_MS=55.0
MAX_DONORS_PER_CLASS=3

@dataclass(frozen=True)
class Unit:
    group: str
    wav_id: str
    context: str
    final: str
    duration_ms: float
    core: tuple[float,...]
    late: tuple[float,...]
    core_rms_db: float
    late_rms_db: float


def _signature(samples:np.ndarray,rate:int,center_ms:float)->tuple[tuple[float,...],float]:
    half=WINDOW_MS/2
    lo=int(round((center_ms-half)*rate/1000))
    hi=int(round((center_ms+half)*rate/1000))
    if lo<0 or hi>len(samples) or hi-lo<200:
        raise ValueError("spectral_window_outside_wav")
    values=np.asarray(samples[lo:hi],dtype=float)
    values=values-values.mean()
    rms=float(np.sqrt(np.mean(values**2)))
    if rms<1e-6:
        raise ValueError("near_silent_window")
    power=np.abs(np.fft.rfft(values*np.hanning(len(values))))**2
    hz=np.fft.rfftfreq(len(values),1/rate)
    upper=min(11000.0,0.46*rate)
    if upper<5500:
        raise ValueError("insufficient_sample_rate")
    grid=np.linspace(1500,upper,48)
    p=np.interp(grid,hz,power)
    p=p/(p.sum()+1e-15)
    return tuple(np.log10(p+1e-8).tolist()),float(20*np.log10(rms))


def _distance(a:tuple[float,...],b:tuple[float,...])->float:
    return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))


def _penalty(donor:Unit,target:Unit)->tuple[float,float,float]:
    # Donor-to-target held-out frication and donor-to-fixed-target-right seam.
    spectral=_distance(donor.core,target.core)
    seam=_distance(donor.core,target.late)
    seam_energy=abs(donor.core_rms_db-target.late_rms_db)
    return spectral,seam,seam_energy


def extract(root:Path,*,base:str="sh")->tuple[list[Unit],dict]:
    if base not in ("sh","s","x"):
        raise ValueError("supported bases: sh,s,x")
    root=root.expanduser().resolve(strict=True)
    entries,warnings=load_voicebank(root)
    obs=collect_observations((e for e in entries if e.wav_path.is_file()),
                             affix_pairs(load_prefix_maps(root)))
    cache={}
    valid={}
    reasons=Counter()
    for o in obs:
        if o.base_unit!=base:
            continue
        context=context_for(base,o.final)
        if context not in ("plain","rounded"):
            continue
        e=o.entry
        wav=e.wav_path.resolve()
        key=(str(wav),round(e.offset+e.preutterance,1),context)
        if key in valid:
            reasons["duplicate_successful_oto"]+=1
            continue
        if not np.isfinite(e.offset) or not np.isfinite(e.preutterance):
            reasons["nonfinite_oto"]+=1;continue
        if e.preutterance<MIN_PREUTTERANCE_MS:
            reasons["short_oto_interval"]+=1;continue
        try:
            if wav not in cache:
                cache[wav]=read_wav(wav)
            samples,rate=cache[wav]
            begin=e.offset
            # Provisional frication proxy is centered before preutterance, not
            # after it: avoid borrowing future target vowel by construction.
            core_center=begin+max(14.0,0.42*e.preutterance)
            late_center=begin+e.preutterance-16.0
            core,core_db=_signature(samples,rate,core_center)
            late,late_db=_signature(samples,rate,late_center)
        except (AudioReadError,OSError,ValueError,OverflowError,FloatingPointError):
            reasons["bad_window_or_audio"]+=1;continue
        subbank=str(e.oto_path.parent.relative_to(root))
        wav_id=sha256(str(wav.relative_to(root)).encode()).hexdigest()[:24]
        valid[key]=Unit(subbank,wav_id,context,o.final,
                        float(e.preutterance),core,late,core_db,late_db)
    return list(valid.values()),{
        "oto_entries":len(entries),"oto_warnings":len(warnings),
        "valid_units":len(valid),"independent_wavs":len({u.wav_id for u in valid.values()}),
        "contexts":dict(Counter(u.context for u in valid.values())),
        "exclusions":dict(reasons),
    }


def _select(donors:list[Unit],target:Unit,k:int)->list[Unit]:
    # Match without observing the acoustic outcome: duration and WAV id only.
    return sorted(donors,key=lambda d:(abs(d.duration_ms-target.duration_ms),d.wav_id,d.final))[:k]


def trial_rows(units:list[Unit],*,min_per_class:int=2)->list[dict]:
    bybank=defaultdict(list)
    for u in units: bybank[u.group].append(u)
    trials=[]
    for bank,events in sorted(bybank.items()):
        for t in events:
            same=[d for d in events if d.context==t.context and d.wav_id!=t.wav_id]
            cross=[d for d in events if d.context!=t.context and d.wav_id!=t.wav_id]
            if len(same)<min_per_class or len(cross)<min_per_class:
                continue
            k=min(MAX_DONORS_PER_CLASS,len(same),len(cross))
            a=_select(same,t,k)
            b=_select(cross,t,k)
            def med(donors):
                p=np.array([_penalty(d,t) for d in donors])
                return np.median(p,axis=0)
            same_loss=med(a)
            cross_loss=med(b)
            natural_seam=_distance(t.core,t.late)
            trials.append({
                "wav_id":t.wav_id,"subbank":bank,"context":t.context,
                "target_final":t.final,"target_duration_ms":round(t.duration_ms,4),
                "donors_per_arm":k,
                "same_duration_mismatch_ms":round(float(np.median([abs(d.duration_ms-t.duration_ms) for d in a])),4),
                "cross_duration_mismatch_ms":round(float(np.median([abs(d.duration_ms-t.duration_ms) for d in b])),4),
                "natural_target_seam_spectral":round(natural_seam,7),
                "same_core_reconstruction":round(float(same_loss[0]),7),
                "cross_core_reconstruction":round(float(cross_loss[0]),7),
                "same_fixed_right_seam":round(float(same_loss[1]),7),
                "cross_fixed_right_seam":round(float(cross_loss[1]),7),
                "same_fixed_right_energy_db":round(float(same_loss[2]),7),
                "cross_fixed_right_energy_db":round(float(cross_loss[2]),7),
                "delta_reconstruction":round(float(cross_loss[0]-same_loss[0]),7),
                "delta_seam":round(float(cross_loss[1]-same_loss[1]),7),
                "delta_energy_db":round(float(cross_loss[2]-same_loss[2]),7),
            })
    return trials


def paired_stats(trials:list[dict],field:str,*,iterations:int=1999)->dict:
    # Aggregation to one contrast per physical WAV avoids pseudoreplication
    # when a recording exposes several OTO aliases of the target context.
    grouped=defaultdict(list)
    for t in trials:
        grouped[t["wav_id"]].append(float(t[field]))
    pairs=np.array([float(np.mean(grouped[k])) for k in sorted(grouped)],dtype=float)
    n=len(pairs)
    if n<6:
        return {"status":"insufficient","independent_wavs":n,"reason":"fewer_than_six_WAVs"}
    rng=np.random.default_rng(SEED+len(field))
    observed=float(np.mean(pairs))
    signs=rng.choice([-1.0,1.0],size=(iterations,n))
    null=(signs*pairs).mean(axis=1)
    # Two-sided p; test direction only after reporting estimate.
    p=(1+int(np.count_nonzero(np.abs(null)>=abs(observed)-1e-12)))/(iterations+1)
    indices=rng.integers(0,n,size=(iterations,n))
    means=pairs[indices].mean(axis=1)
    return {"status":"evaluated","independent_wavs":n,
            "mean_cross_minus_same":round(observed,7),
            "median_WAV_delta":round(float(np.median(pairs)),7),
            "bootstrap_95pct_CI":[round(float(v),7) for v in np.quantile(means,[0.025,0.975])],
            "two_sided_WAV_sign_flip_p":round(p,6),
            "sign_flip_trials":iterations,
            "positive_fraction":round(float(np.mean(pairs>0)),6)}


def analyze_bank(root:Path,*,base:str="sh")->dict:
    units,metadata=extract(root,base=base)
    trials=trial_rows(units)
    metrics={k:paired_stats(trials,k) for k in ("delta_reconstruction","delta_seam","delta_energy_db")}
    bybank={}
    for group in sorted({u.group for u in units}):
        ts=[r for r in trials if r["subbank"]==group]
        bybank[group]={
            "candidate_units":sum(u.group==group for u in units),
            "targets":len(ts),"independent_target_wavs":len({t["wav_id"] for t in ts}),
            "reconstruction":paired_stats(ts,"delta_reconstruction"),
            "seam":paired_stats(ts,"delta_seam"),
        }
    return {"extraction":metadata,
            "eligible_targets":len(trials),
            "eligible_target_wavs":len({t["wav_id"] for t in trials}),
            "metrics":metrics,"subbanks":bybank,
            "duration_match_summary":{
                "same_median_ms":round(float(np.median([t["same_duration_mismatch_ms"] for t in trials])),4) if trials else None,
                "cross_median_ms":round(float(np.median([t["cross_duration_mismatch_ms"] for t in trials])),4) if trials else None,
            },
            "trials":trials,
            "interpretation":"Provisional held-out donor spectral and seam proxies; no renderer-quality claims."}


def run_manifest(manifest:Path,output:Path,*,base:str="sh")->dict:
    obj=json.loads(manifest.expanduser().resolve(strict=True).read_text("utf-8"))
    if obj.get("schema")!="phonoweave.voicebank_replication_manifest.v1":
        raise ValueError("invalid voicebank manifest")
    seen=set()
    rows=[]
    for bank in obj["voicebanks"]:
        bid=bank["bank_id"]
        if bid in seen: raise ValueError("duplicate bank ID")
        seen.add(bid)
        result=analyze_bank(Path(bank["path"]),base=base)
        result.update(bank_id=bid,speaker_group_id=bank["speaker_group_id"])
        rows.append(result)
    report={"schema":SCHEMA,"base":base,"banks":rows,
            "no_audio_created":True,"source_WAV_OTO_modified":False,
            "limitations":[
                "This is an offline donor-versus-target spectral proxy, not actual rendered singing or perceptual quality.",
                "OTO preutterance is not a validated phonetic frication boundary; vowel/glide contamination remains possible.",
                "Duration matching is not a full control for recording position, final vowel, pitch, or recording script.",
                "Donors may share audio material despite different WAV paths; WAV identifier controls files only.",
                "Per-bank repeated pitch layers are not independent speaker replications.",
                "A positive reconstruction delta does not independently establish a need for separate recording units.",
            ]}
    output=output.expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n","utf-8")
    return report
