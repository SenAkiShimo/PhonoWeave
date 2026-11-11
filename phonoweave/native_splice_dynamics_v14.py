"""v14: waveform-rendered seam diagnostics and held-out continuation probes.

This research module uses ORIGINAL WAV PCM samples from an OTO-curated historical
clip archive, but neither OTO timestamps nor labels during extraction/ranking.
Audio-only v9 candidate region 2 starts serve as UNVERIFIED seams, not phones.
Target pre-seam audio is an oracle left context, not text-based synthesis input.

Most importantly: lowering a score used to select donors is NOT validation.
The separate right-side trajectory loss against withheld target audio is a
negative-control/diagnostic endpoint, not perceived singing quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import wave
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .native_synthesis_proxy_v13 import (build_clips, freeze_partitions,
    stage_normalize, rank_donors)
from .native_motif_link_v10 import load_v9
from .native_wav_motifs_v9 import pcm16, audio_features

SCHEMA='phonoweave.waveform_splice_dynamics.v1.4'
BUDGET=4
LEFT_MS=90
RIGHT_MS=150
FADE_MS=8
SEED=20261114
EPS=1e-9

@dataclass(frozen=True)
class WavWindow:
    left: np.ndarray   # the only piece of held target audio available to selector
    right: np.ndarray  # held away from selection; used ONLY in independent scoring
    rate: int
    boundary_ms: float


def _wav_bytes(samples:np.ndarray,rate:int)->bytes:
    y=np.clip(np.rint(np.clip(samples,-1.,1.)*32767),-32768,32767).astype('<i2')
    buff=io.BytesIO()
    with wave.open(buff,'wb') as out:
        out.setnchannels(1);out.setsampwidth(2);out.setframerate(rate)
        out.writeframes(y.tobytes())
    return buff.getvalue()


def load_source_windows(v9:Path,archive:Path,clips:list)->tuple[list[WavWindow],dict]:
    rows,_,report=load_v9(Path(v9))
    if hashlib.sha256(Path(archive).read_bytes()).hexdigest()!=report['archive_sha256']:
        raise ValueError('source ZIP SHA256 differs from frozen v9 manifest')
    start={}
    for row in rows:
        if int(row['region_index'])==2:
            key=(str(row['bank']),str(row['clip_group']),str(row['wav_archive_entry']))
            if key in start:raise ValueError('duplicate stage2 start for one source WAV')
            start[key]=(float(row['start_ms']),row['source_pcm_sha256'])
    windows=[]; rates=set()
    with zipfile.ZipFile(archive) as z:
        names=set(z.namelist())
        for clip in clips:
            key=(clip.bank,clip.group,clip.wav)
            if key not in start or clip.wav not in names:
                raise ValueError('missing frozen clip or source ZIP entry')
            bound,sha=start[key]
            if sha!=clip.sha:raise ValueError('v9 source identity disagrees with stage2 index')
            y,rate,actual=pcm16(z.read(clip.wav))
            if actual!=clip.sha:raise ValueError('source PCM SHA256 changed')
            # Read fixed boundary-relative windows, never OTO-based phone boundaries.
            at=int(round(rate*bound/1000.))
            a=at-int(round(rate*LEFT_MS/1000.))
            b=at+int(round(rate*RIGHT_MS/1000.))
            if a<0 or b>len(y):raise ValueError('insufficient physical audio around unanchored seam')
            windows.append(WavWindow(y[a:at].copy(),y[at:b].copy(),rate,bound))
            rates.add(rate)
    if len(rates)!=1:raise ValueError('mixed sample rates require explicit frozen resampling')
    return windows,{'source_zip_sha256_verified':True,'source_pcm_sha256_verified':len(windows),
                    'rate_hz':next(iter(rates)),'seams_are_audio_only_stage2_starts':True}


def mel_signature(y:np.ndarray, rate:int, side:str)->np.ndarray:
    x,_,_=audio_features(np.asarray(y,dtype=np.float32),rate)
    if len(x)<3:raise ValueError('too little waveform for spectrogram')
    return np.mean(x[-3:] if side=='left' else x[:3],axis=0)


def rms(y:np.ndarray)->float:
    return float(np.sqrt(np.mean(np.asarray(y,dtype=float)**2)+EPS))


def periodicity(y:np.ndarray,rate:int)->float|None:
    """Conservative uncalibrated periodicity proxy; None for low energy/ambiguous.

    Not F0 identification. This can be unusable across frication→vowel seams.
    """
    x=np.asarray(y,dtype=np.float64)
    if len(x)<int(.025*rate) or rms(x)<.01:return None
    x=x-x.mean();power=float(x@x)
    if power<EPS:return None
    lo,hi=max(1,int(rate/420)),min(len(x)//2,int(rate/85))
    if lo>=hi:return None
    # Peak normalized by segment energy, not a posterior or voiced phone label.
    a=np.array([np.dot(x[:-l],x[l:])/np.sqrt(np.dot(x[:-l],x[:-l])*np.dot(x[l:],x[l:])+EPS)
                for l in range(lo,hi+1)])
    peak=float(a.max())
    return peak if peak>=.55 else None


def seam_components(left:np.ndarray,right:np.ndarray,rate:int)->dict:
    last=left[-max(1,int(.03*rate)):]
    first=right[:max(1,int(.03*rate))]
    mm=float(np.mean((mel_signature(left,rate,'left')-mel_signature(right,rate,'right'))**2))
    energy=abs(np.log(rms(last))-np.log(rms(first)))
    p,q=periodicity(last,rate),periodicity(first,rate)
    return {'mel_mse':mm,'log_rms_gap':float(energy),
            'periodicity_gap':None if p is None or q is None else abs(p-q),
            'periodicity_pair_valid':p is not None and q is not None}


def train_scales(windows:list[WavWindow],train:np.ndarray)->dict:
    natural=[seam_components(w.left,w.right,w.rate) for w,ok in zip(windows,train) if ok]
    if len(natural)<12:raise ValueError('not enough training seams')
    return {k:max(.02,float(np.median([r[k] for r in natural]))) for k in ('mel_mse','log_rms_gap')}


def seam_score(comp:dict, scales:dict)->float:
    # Fixed before scoring; no auto-tuning on audit or target right audio.
    score=.7*comp['mel_mse']/scales['mel_mse']+.3*comp['log_rms_gap']/scales['log_rms_gap']
    if comp['periodicity_pair_valid']:
        score+=.2*comp['periodicity_gap']
    return float(score)


def select_donor(left:np.ndarray,windows:list[WavWindow],ids:np.ndarray,
                 scales:dict)->int:
    """Never reads any part of the target's held-out right continuation."""
    if len(ids)<2:raise ValueError('selector needs at least 2 candidate donors')
    scores=[]
    for idx in ids:
        w=windows[int(idx)]
        scores.append(seam_score(seam_components(left,w.right,w.rate),scales))
    return int(ids[min(range(len(ids)),key=lambda j:(scores[j],int(ids[j])))])


def render_splice(left:np.ndarray,right:np.ndarray,rate:int)->np.ndarray:
    """Actually render a short linear-PCM splice via an 8-ms equal-power crossfade."""
    n=max(2,int(round(rate*FADE_MS/1000)))
    if len(left)<=n or len(right)<=n:raise ValueError('too short to splice')
    t=np.linspace(0.,np.pi/2,n)
    overlap=left[-n:]*np.cos(t)+right[:n]*np.sin(t)
    return np.r_[left[:-n],overlap,right[n:]].astype(np.float32)


def transient_metrics(splice:np.ndarray, left_size:int,rate:int)->dict:
    """Post-render engineering diagnostics; not a subjective quality measure."""
    n=max(2,int(round(rate*FADE_MS/1000)))
    seam_end=left_size
    a=max(0,seam_end-int(rate*.025));b=min(len(splice),seam_end+int(rate*.025))
    near=splice[a:b]
    before=splice[max(0,a-int(rate*.025)):a]
    after=splice[b:min(len(splice),b+int(rate*.025))]
    local=float(np.sqrt(np.mean(near.astype(float)**2)+EPS))
    return {'splice_local_peak_to_rms':float(np.max(np.abs(near))/max(local,1e-6)),
            'splice_max_sample_step_over_rms':float(np.max(np.abs(np.diff(near)))/max(local,1e-6)),
            'splice_energy_vs_sides':float(local/max((rms(before)+rms(after))/2,1e-6)),
            'splice_energy_deviation_abs_log':abs(float(np.log(local/max((rms(before)+rms(after))/2,1e-6))))}


def continuation_signature(y:np.ndarray,rate:int)->np.ndarray:
    """First 100ms of post-overlap waveform, 24D log-mel time trajectory."""
    n=max(2,int(round(rate*FADE_MS/1000)))
    x,_,_=audio_features(y[n:n+int(rate*.105)],rate)
    if len(x)<6:raise ValueError('not enough waveform to score continuation')
    return np.asarray(x[:8],float)


def fit_trajectory_normalizer(windows:list[WavWindow],train:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    examples=np.concatenate([continuation_signature(w.right,w.rate) for w,ok in zip(windows,train) if ok])
    mu=examples.mean(axis=0);std=examples.std(axis=0)
    floor=max(.1,float(std.max())*.2)
    return mu,np.maximum(std,floor)


def trajectory_loss(reference:np.ndarray,donor:np.ndarray,mu:np.ndarray,scale:np.ndarray)->float:
    # Target reference comes from the held-out right ONLY at evaluation time.
    if reference.shape!=donor.shape:return float('nan')
    return float(np.mean(((reference-donor)/scale)**2))


def _score_split(clips,windows,Z,train,mask,scales,mu,scale,save_audio:Path|None=None)->dict:
    by_group=defaultdict(lambda:defaultdict(list));rows=[]
    for idx in np.flatnonzero(mask):
        idx=int(idx);c=clips[idx];w=windows[idx]
        pool=rank_donors(Z,clips,train,idx,'pooled',BUDGET)
        # Exactly the same four training WAV candidates for both conditions.
        baseline=int(pool[0]);selected=select_donor(w.left,windows,pool,scales)
        target=continuation_signature(w.right,w.rate) # evaluation only
        scores={}
        for condition, donor in [('pooled',baseline),('seam_aware',selected)]:
            donor_w=windows[donor]
            signal=render_splice(w.left,donor_w.right,w.rate)
            trans=transient_metrics(signal,len(w.left),w.rate)
            score=trajectory_loss(target,continuation_signature(donor_w.right,w.rate),mu,scale)
            seam=seam_components(w.left,donor_w.right,w.rate)
            scores[condition]={'trajectory_mse':score, 'predicted_seam_cost':seam_score(seam,scales),
                               **trans,'periodicity_pair_valid':seam['periodicity_pair_valid']}
            for met,value in scores[condition].items():
                if isinstance(value,(float,int)) and met!='periodicity_pair_valid':
                    by_group[c.group][condition+'_'+met].append(float(value))
            if save_audio is not None and c.group in ('ai_to_sh','en0_to_sh'):
                save_audio.mkdir(parents=True,exist_ok=True)
                path=save_audio/(c.bank+'_'+c.group+'_'+condition+'.wav')
                # Optional listening artifacts only; never committed to GitHub.
                path.write_bytes(_wav_bytes(signal,w.rate))
        rows.append({'group':c.group,'bank':c.bank,'source_pcm_sha256':c.sha,
                     'seam_ms_audio_only':w.boundary_ms,'training_candidate_count':len(pool),
                     'pooled_donor_sha256':clips[baseline].sha,
                     'seam_aware_donor_sha256':clips[selected].sha,
                     'same_donor':baseline==selected,
                     **{cond+'_'+key:value for cond,rec in scores.items() for key,value in rec.items()}})
    gr={g:{name:float(np.mean(x)) for name,x in metrics.items()} for g,metrics in by_group.items()}
    common=sorted(gr)
    if len(common)<3:raise ValueError('too few held source families')
    def averaged(metric):return float(np.mean([gr[g][metric] for g in common]))
    outcome={'groups':len(common),'observations':len(rows),'donor_budget_equal':BUDGET,
             'changed_donor_fraction':float(np.mean([not row['same_donor'] for row in rows]))}
    for metric in ('trajectory_mse','predicted_seam_cost','splice_local_peak_to_rms',
                   'splice_max_sample_step_over_rms','splice_energy_deviation_abs_log'):
        base=averaged('pooled_'+metric);alt=averaged('seam_aware_'+metric)
        outcome[metric]={'pooled':round(base,6),'seam_aware':round(alt,6),
                         'relative_gain_positive_is_better':round(1-alt/max(base,EPS),6),
                         'groups_with_lower_seam_aware':sum(gr[g]['seam_aware_'+metric]<gr[g]['pooled_'+metric] for g in common)}
    outcome['periodicity_gate_coverage']=float(np.mean([row['seam_aware_periodicity_pair_valid'] for row in rows]))
    return {'summary':outcome,'rows':rows,'by_group':gr}


def run(v9:Path,raw_zip:Path,output:Path,examples:bool=False)->dict:
    output=Path(output)
    if output.exists():raise FileExistsError('refusing to overwrite previous experiment')
    rows,X,report=load_v9(Path(v9))
    clips,info=build_clips(rows,X)
    windows,source=load_source_windows(Path(v9),Path(raw_zip),clips)
    partition=freeze_partitions(clips)
    Z,norm=stage_normalize(clips,partition['train'])
    scales=train_scales(windows,partition['train'])
    mu,scale=fit_trajectory_normalizer(windows,partition['train'])
    # All metrics are frozen; validation is diagnostic and never used to adjust
    # weights, donor budget, pre/post windows, or audit threshold.
    validation=_score_split(clips,windows,Z,partition['train'],partition['validation'],scales,mu,scale)
    audit=_score_split(clips,windows,Z,partition['train'],partition['audit'],scales,mu,scale,
                      output/'illustrative_splices' if examples else None)
    # No auto split: require improvement in held continuation AND no measured
    # physical transient regression, plus independent replication (unavailable).
    summary={'schema':SCHEMA,'v9_archive_sha256':report['archive_sha256'],
             'audio_source':source,'complete_clips':len(clips),'incomplete_skipped':info['incomplete_clips_skipped'],
             'source_groups_by_partition':{k:len({c.group for c,yes in zip(clips,m) if yes}) for k,m in partition.items()},
             'seam_definition':'v9 unanchored third-region start; NOT verified CV/phone boundary',
             'sources':'historical OTO-curated 600ms clips from one singer; no new full-WAV recording',
             'frozen_rule':{'candidate_budget':BUDGET,'left_ms':LEFT_MS,'right_ms':RIGHT_MS,
                            'crossfade_ms':FADE_MS,'weights':{'logmel':.7,'log_rms':.3,'periodicity_if_valid':.2},
                            'feature_fit':'train_only','selection_access_to_target_right':False},
             'validation':validation['summary'],'audit':audit['summary'],
             'no_automatic_inventory_split_merge':True,
             'not_perceptual_or_singing_synthesis_quality':True,
             'oracle_left_audio_unavailable_to_text_only_synthesizer':True,
             'audit_previous_iterations_already_seen_retrospective_only':True}
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    with (output/'audit_comparisons.csv').open('w',encoding='utf8',newline='') as f:
        fields=list(audit['rows'][0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(audit['rows'])
    return summary


def main(argv=None)->int:
    p=argparse.ArgumentParser(prog='phonoweave v14 waveform splice dynamics')
    p.add_argument('--v9-results',required=True,type=Path)
    p.add_argument('--audio-zip',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--render-examples',action='store_true')
    a=p.parse_args(argv)
    print(json.dumps(run(a.v9_results,a.audio_zip,a.output,a.render_examples),ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())