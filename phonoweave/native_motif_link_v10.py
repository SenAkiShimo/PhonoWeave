"""v10: self-supervised, source-disjoint acoustic motif association (NOT phone IDs).

No IPA, transcripts, OTO timestamps or user labels are used to fit embeddings,
model order, distances, or clusters. Retrospective clip provenance remains.
Model order and unknown thresholds are selected using development/validation
source groups only. Frozen holdout is scored once, descriptively.
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .native_acoustic_subtypes_v9 import adjusted_rand_index, normalized_mutual_information
from .native_wav_motifs_v9 import audio_features, embed_segment, pcm16, propose_segments, MIN_RMS

SCHEMA='phonoweave.cross_recording_acoustic_motifs.v1.0'
SEED=20261110
MAX_K=7


def group_split(rows: list[dict]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Whole clip families stay in one partition, irrespective of bank/style."""
    names=sorted({str(x['clip_group']) for x in rows})
    if len(names)<10:raise ValueError('requires at least 10 distinct recording families')
    names=sorted(names,key=lambda s:hashlib.sha256(('v10split:'+s).encode()).hexdigest())
    n=len(names);a=max(6,round(n*.6));b=max(2,round(n*.2))
    if a+b>=n:raise ValueError('not enough groups for independent selection and audit')
    key={v:('train' if i<a else 'validation' if i<a+b else 'audit') for i,v in enumerate(names)}
    return tuple(np.array([key[str(x['clip_group'])]==s for x in rows],bool)
                 for s in ('train','validation','audit'))


def normalize(X:np.ndarray,train:np.ndarray):
    mean=X[train].mean(axis=0)
    std=X[train].std(axis=0)
    floor=max(0.04, float(np.max(std))*0.2)
    scale=np.maximum(std,floor)
    return np.clip((X-mean)/scale,-6,6),{'floor':floor,'fit_from_train_only':True}


def distance2(X:np.ndarray,C:np.ndarray)->np.ndarray:
    return np.maximum(0.,((X[:,None,:]-C[None,:,:])**2).sum(axis=2))


def fit_kmeans(X:np.ndarray,k:int,seed:int=SEED,restarts:int=9)->np.ndarray:
    """Kmeans++ restarts; deterministic, no user labels or path information."""
    if X.ndim!=2 or len(X)<max(12,3*k) or k<1 or not np.isfinite(X).all():
        raise ValueError('invalid training features or k')
    if k==1:return X.mean(axis=0,keepdims=True)
    rng=np.random.default_rng(seed+1009*k)
    best_loss=math.inf;best=None
    for _ in range(restarts):
        centers=[X[int(rng.integers(len(X)))].copy()]
        for j in range(1,k):
            d=np.min(distance2(X,np.array(centers)),axis=1)
            if d.sum()<1e-10:break
            centers.append(X[int(rng.choice(len(X),p=d/d.sum()))].copy())
        if len(centers)<k:continue
        C=np.array(centers)
        for _ in range(50):
            d=distance2(X,C);labels=d.argmin(axis=1)
            if len(set(labels))<k:break
            new=np.array([X[labels==j].mean(axis=0) for j in range(k)])
            if np.max(np.abs(new-C))<1.e-7:
                C=new;break
            C=new
        d=distance2(X,C);labels=d.argmin(axis=1)
        if len(set(labels))<k:continue
        loss=float(d[np.arange(len(X)),labels].sum())
        if loss<best_loss-1e-9:best_loss=loss;best=C.copy()
    if best is None:raise ValueError('no kmeans solution with all clusters occupied')
    return best


def model_audit(X:np.ndarray,rows:list[dict],train:np.ndarray,validation:np.ndarray,
                audit:np.ndarray, max_k:int=MAX_K)->tuple[dict, np.ndarray, np.ndarray,dict]:
    """Model selection never looks at the held-out audit subset.

    Validation is model selection/calibration, NOT an untouched test result.
    Source families across styles cannot cross any of the three partitions.
    """
    if X.shape!=(len(rows),48) or not np.isfinite(X).all():
        raise ValueError('requires finite 48D waveform-derived features')
    if not (train.shape==validation.shape==audit.shape==(len(rows),)):
        raise ValueError('bad partition masks')
    if np.any((train.astype(int)+validation.astype(int)+audit.astype(int))!=1):
        raise ValueError('partitions must be disjoint and exhaustive')
    if min(int(train.sum()),int(validation.sum()),int(audit.sum()))<15:
        raise ValueError('insufficient regions in one or more partitions')
    names=[{rows[i]['clip_group'] for i in np.flatnonzero(mask)} for mask in (train,validation,audit)]
    if any(names[i]&names[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('recording family leaked across partitions')
    Y,norm=normalize(X,train)
    base=fit_kmeans(Y[train],1)
    e1=float(distance2(Y[validation],base).sum())
    candidates=[];models={1:base}
    for k in range(2,min(max_k,int(train.sum())//7)+1):
        try:C=fit_kmeans(Y[train],k)
        except ValueError:continue
        models[k]=C
        tr=distance2(Y[train],C).argmin(axis=1)
        val_d=distance2(Y[validation],C)
        va=val_d.argmin(axis=1)
        support=[len({r['clip_group'] for r,j in zip(np.array(rows,dtype=object)[train],tr) if j==i}) for i in range(k)]
        v_support=[len({r['clip_group'] for r,j in zip(np.array(rows,dtype=object)[validation],va) if j==i}) for i in range(k)]
        n_tr=np.bincount(tr,minlength=k);n_val=np.bincount(va,minlength=k)
        ev=float(val_d[np.arange(len(va)),va].sum())
        gain=(e1-ev)/max(e1,1e-10)
        sorted_d=np.sort(val_d,axis=1)
        separation=float(np.mean((sorted_d[:,1]-sorted_d[:,0])/np.maximum(sorted_d[:,1],1e-9)>0.18))
        # Conservative guards: each pattern must repeat across actual file families.
        valid=(min(support)>=3 and min(v_support)>=2 and min(n_tr)>=8 and
               min(n_val)>=3 and gain>=0.12 and separation>=0.60)
        score=gain-0.085*(k-1)
        candidates.append({'k':k,'validation_reconstruction_gain_vs_k1':round(gain,5),
                           'validation_clear_fraction':round(separation,5),
                           'train_min_independent_family_support':min(support),
                           'validation_min_independent_family_support':min(v_support),
                           'validation_min_regions_per_motif':int(min(n_val)),
                           'penalized_validation_score':round(score,5),'pre_screen_pass':bool(valid)})
    accepted=[q for q in candidates if q['pre_screen_pass']]
    accepted.sort(key=lambda q:(-q['penalized_validation_score'],q['k']))
    stability_trials=[];chosen=1;stability=None
    # Assess reproducibility under source-group subsampling, without touching audit.
    for cand in accepted:
        k=cand['k'];fullC=models[k]
        val_labels=distance2(Y[validation],fullC).argmin(axis=1)
        groupnames=sorted(names[0]);reps=[]
        for boot in range(6):
            rng=np.random.default_rng(SEED+997*k+boot)
            selected=set(rng.choice(groupnames,size=max(6,round(len(groupnames)*.80)),replace=False))
            sel=train & np.array([r['clip_group'] in selected for r in rows])
            try:
                alt=fit_kmeans(Y[sel],k,seed=SEED+boot,restarts=5)
                reps.append(adjusted_rand_index(val_labels,distance2(Y[validation],alt).argmin(axis=1)))
            except ValueError:reps.append(0.)
        stable=float(np.median(reps))
        stability_trials.append({'k':k,'median_ARI_across_source_group_subsamples':round(stable,5)})
        if stable>=0.75:
            chosen=k;stability=stable;break
    C=models[chosen]
    train_d=distance2(Y[train],C)
    val_d=distance2(Y[validation],C)
    val_lab=val_d.argmin(axis=1)
    tr_lab=train_d.argmin(axis=1)
    # Fit novelty threshold from development-only and source-disjoint validation.
    thresholds=[]
    for k in range(chosen):
        distances=np.r_[train_d[tr_lab==k,k],val_d[val_lab==k,k]]
        if len(distances)<4:distances=train_d[tr_lab==k,k]
        thresholds.append(max(1e-8,float(np.quantile(distances,0.95))*1.10))
    thresholds=np.array(thresholds)
    all_d=distance2(Y,C)
    labels=all_d.argmin(axis=1)
    sorted_d=np.sort(all_d,axis=1)
    clear=np.ones(len(X),bool) if chosen==1 else ((sorted_d[:,1]-sorted_d[:,0])/np.maximum(sorted_d[:,1],1e-9)>=0.18)
    known=(all_d[np.arange(len(X)),labels]<=thresholds[labels])&clear
    # Don't treat a 1-centroid fallback as discovered distinct motif types.
    # It can still reject OOD regions and identify repeated broad acoustic material.
    summary={'selected_k':chosen,'selection_status':('exploratory_multiple_motifs' if chosen>1 else 'abstain_no_stable_multimotif_structure'),
             'max_k_considered':max_k,'tested_orders':candidates,
             'stability_trials':stability_trials,'chosen_stability_median_ari':None if stability is None else round(stability,5),
             'source_groups_train_validation_audit':[len(s) for s in names],
             'regions_train_validation_audit':[int(train.sum()),int(validation.sum()),int(audit.sum())],
             'normalization':norm,'distance_thresholds_fit_without_audit':thresholds.tolist(),
             'audit_regions_known':int((known&audit).sum()),'audit_regions_unknown':int((~known&audit).sum()),
             'audit_coverage_fraction':round(float(known[audit].mean()),5),
             'audit_membership_counts':dict(Counter('unknown' if not known[i] else f'motif_{labels[i]}' for i in np.flatnonzero(audit))),
             'audit_has_no_external_phone_labels':True,
             'model_selection_or_threshold_tuning_on_audit':False,
             'no_new_phoneme_claim_or_alias_changes':True}
    return summary,labels,known,{'normalized':Y,'centers':C,'distances':all_d,'audit':audit,'train':train,'validation':validation}


def _approximate_cross_source_neighbors(rows:list[dict],Y:np.ndarray,labels:np.ndarray,
                                        known:np.ndarray,limit:int=4)->dict[int,list[tuple[float,int]]]:
    """Deterministic random-projection candidate index plus *exact* spectral distances.

    Candidate retrieval is approximate (possible false negatives), never a
    statement about acoustic/phone equivalence. This avoids N x N matrices for
    potentially thousands of whole-recording regions.
    """
    n=len(rows)
    if not (Y.shape[0]==len(labels)==len(known)==n):raise ValueError('length mismatch')
    rng=np.random.default_rng(SEED)
    probes=rng.normal(size=(3,Y.shape[1]))
    probes/=np.maximum(np.linalg.norm(probes,axis=1,keepdims=True),1e-12)
    pool={i:set() for i in np.flatnonzero(known)}
    for k in sorted(set(map(int,labels[known]))):
        ids=np.flatnonzero(known & (labels==k))
        if len(ids)<=1:continue
        # Three one-dimensional projections; inspect a bounded local window.
        for p in probes:
            order=ids[np.argsort(Y[ids]@p,kind='mergesort')]
            half=64
            for ix,i in enumerate(order):
                lo=max(0,ix-half);hi=min(len(order),ix+half+1)
                pool[int(i)].update(int(j) for j in order[lo:hi] if i!=j)
    closest={}
    for i,options in pool.items():
        possible=[j for j in options if rows[i]['source_pcm_sha256']!=rows[j]['source_pcm_sha256']
                  and rows[i]['clip_group']!=rows[j]['clip_group']]
        if not possible:
            closest[i]=[];continue
        js=np.array(sorted(possible),dtype=int)
        d=((Y[js]-Y[i])**2).sum(axis=1)
        nearest=sorted(zip(d.tolist(),js.tolist()),key=lambda z:(z[0],z[1]))[:limit]
        closest[i]=nearest
    return closest


def edges_between_recordings(rows:list[dict],Y:np.ndarray,labels:np.ndarray,known:np.ndarray,
                             calibration_mask:np.ndarray, max_per_region:int=3)->tuple[list[dict],dict]:
    """Mutual approximate-kNN graph under dev-only distance radii.

    Reciprocal links are stronger evidence of repeated *acoustic geometry*,
    not verification that the linked intervals are the same phonetic unit.
    """
    n=len(rows)
    if not (len(Y)==len(labels)==len(known)==len(calibration_mask)==n):
        raise ValueError('region count mismatch')
    # Calibrate radii without audit observations even entering the candidate index.
    calibration_neighbors=_approximate_cross_source_neighbors(rows,Y,labels,known & calibration_mask,limit=max_per_region)
    neighbors=_approximate_cross_source_neighbors(rows,Y,labels,known,limit=max_per_region)
    radius={}
    for k in sorted(set(map(int,labels[known]))):
        # Calibration is strictly train/validation, not untouched audit.
        nd=[cand[0][0] for i,cand in calibration_neighbors.items() if cand and int(labels[i])==k]
        radius[k]=float(np.quantile(nd,0.9))*1.15 if len(nd)>=6 else 0.
    links=[]
    for i,near in neighbors.items():
        for d,j in near:
            if j<=i or d>radius.get(int(labels[i]),0.):continue
            if any(other==i for _,other in neighbors.get(j,[])):
                links.append({'source_region_a':i,'source_region_b':j,
                              'motif_id':f'acoustic_motif_{int(labels[i])}',
                              'squared_scaled_spectral_distance':round(d,5),
                              'status':'mutual_cross_source_candidate_NOT_same_phone'})
    linked={i for e in links for i in (e['source_region_a'],e['source_region_b'])}
    report={'cross_recording_links':len(links),'distinct_regions_with_links':len(linked),
            'known_regions_without_verified_links':int(known.sum())-len(linked),
            'dev_val_calibrated_link_radii':{str(k):round(v,5) for k,v in radius.items()},
            'neighbor_search_approximate_not_exhaustive':True,
            'different_pcm_sha256_required':True,'different_recording_family_required':True,
            'mutual_nearest_neighbor_required':True,
            'links_are_not_phone_identity':True}
    return links,report


def extract_generic_full_wav_zip(archive:Path)->tuple[list[dict],np.ndarray]:
    """Unanchored full audio path for future original WAV archives (never reads OTO).

    Duplicate basenames across subbanks are conservatively grouped as one family
    for train/validation/audit splitting. Folder and bank used only in audit keys.
    """
    rows=[];X=[]
    with zipfile.ZipFile(archive) as z:
        wavs=sorted((i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith('.wav')), key=lambda i:i.filename)
        if not wavs or len(wavs)>5000:raise ValueError('no WAVs or too many archive WAVs')
        if sum(i.file_size for i in wavs)>2_000_000_000:raise ValueError('archive WAVs exceed safe size')
        for item in wavs:
            name=item.filename
            if name.startswith('/') or '..' in Path(name).parts or item.file_size>128_000_000:
                raise ValueError('unsafe or oversized WAV member')
            audio,rate,digest=pcm16(z.read(item))
            mel,rms,times=audio_features(audio,rate)
            spans=propose_segments(mel,rms,times)
            for number,(a,b) in enumerate(spans):
                if float(np.mean(rms[a:b]))<MIN_RMS:continue
                X.append(embed_segment(mel,a,b))
                rows.append({'wav_archive_entry':name,'bank':next((s for s in Path(name).parts if s.startswith(('lu_','mang_'))),'unidentified_bank'),
                             'clip_group':Path(name).name, 'source_pcm_sha256':digest,
                             'region_index':number,'start_ms':round(float(times[a]),3),
                             'end_ms':round(float(times[b-1]),3),
                             'status':'unanchored_spectral_region_NOT_a_phone'})
    return rows,np.stack(X) if X else np.empty((0,48),dtype=np.float32)



def posthoc_position_diagnostics(rows:list[dict],X:np.ndarray)->list[dict]:
    """Exploratory follow-up: are motif types present *within* acoustic order bins?

    Order bins are the segmentation-produced region_index, NOT user transcript
    slots or phoneme IDs. No result here changes global clustering or thresholds.
    """
    out=[]
    for position in sorted({int(x['region_index']) for x in rows}):
        indices=[i for i,r in enumerate(rows) if int(r['region_index'])==position]
        sub=[rows[i] for i in indices]
        result={'acoustic_region_order_index':position,'source_regions':len(indices),
                'distinct_recording_families':len({r['clip_group'] for r in sub}),
                'exploratory_NOT_an_identified_phone':True}
        if len(indices)<45 or result['distinct_recording_families']<10:
            result['status']='abstain_insufficient_order_stratum_evidence'
        else:
            try:
                tr,va,au=group_split(sub)
                m,_,_,_=model_audit(X[indices],sub,tr,va,au)
                result.update({'status':m['selection_status'],
                               'candidate_acoustic_subpatterns':m['selected_k'],
                               'heldout_candidate_coverage':m['audit_coverage_fraction'],
                               'repeatability_median_ari':m['chosen_stability_median_ari']})
            except ValueError as e:
                result.update(status='abstain_invalid_or_small_stratum',reason=str(e))
        out.append(result)
    return out


def load_v9(v9:Path)->tuple[list[dict],np.ndarray,dict]:
    with np.load(v9/'raw_spectral_embeddings.npz',allow_pickle=False) as a:
        X=np.asarray(a['vectors'],np.float64)
    reg=json.loads((v9/'motif_regions.json').read_text(encoding='utf8'))
    report=json.loads((v9/'summary.json').read_text(encoding='utf8'))
    if reg.get('schema')!='phonoweave.audio_only_acoustic_motifs.v0.9' or report.get('schema')!=reg['schema']:
        raise ValueError('v9 schema mismatch')
    rows=reg['regions']
    if len(rows)!=len(X) or len(rows)!=report['regions'] or X.shape[1:]!=(48,):
        raise ValueError('v9 feature and manifest count mismatch')
    return rows,X,report


def run(output:Path, v9:Path|None=None,full_wav_zip:Path|None=None)->dict:
    output=Path(output)
    if output.exists():raise FileExistsError('refusing to overwrite previous research output')
    if (v9 is None)==(full_wav_zip is None):raise ValueError('provide exactly one acoustic source')
    if v9 is not None:
        rows,X,source=load_v9(v9)
        provenance='historical_OTO_curated_600ms_wav_snippets_from_v9'
        source_sha=source['archive_sha256']
    else:
        rows,X=extract_generic_full_wav_zip(Path(full_wav_zip))
        provenance='original_full_recording_wav_zip_with_unanchored_extraction'
        source_sha=hashlib.sha256(Path(full_wav_zip).read_bytes()).hexdigest()
    if not rows:raise ValueError('no acoustic regions')
    if any(len(str(r.get('source_pcm_sha256','')))!=64 or
           any(c not in '0123456789abcdef' for c in str(r['source_pcm_sha256']).lower()) for r in rows):
        raise ValueError('missing or malformed source PCM hash')
    if len(set((r['wav_archive_entry'],r['region_index']) for r in rows))!=len(rows):
        raise ValueError('duplicate acoustic region identities')
    sha_to_family=defaultdict(set)
    for r in rows:sha_to_family[r['source_pcm_sha256']].add(r['clip_group'])
    if any(len(families)>1 for families in sha_to_family.values()):
        raise ValueError('same PCM recording appears in multiple split families; cannot avoid leakage')
    train,validation,audit=group_split(rows)
    s,assigned,known,hidden=model_audit(np.asarray(X,dtype=np.float64),rows,train,validation,audit)
    edges,edgereport=edges_between_recordings(rows,hidden['normalized'],assigned,known,train|validation)
    table=[]
    for i,(r,k,yes) in enumerate(zip(rows,assigned,known)):
        split='training' if train[i] else 'validation_model_selection' if validation[i] else 'frozen_audit'
        table.append({**r,'v10_motif_id':f'acoustic_motif_{int(k)}' if yes else None,
                      'v10_status':'candidate_NOT_a_phone' if yes else 'unknown_or_ambiguous_acoustic_region',
                      'v10_partition':split,'v10_source_index':i,
                      'nearest_prototype_distance2':round(float(hidden['distances'][i,int(k)]),5)})
    eligible=[i for i in range(len(rows)) if known[i]]
    confounds={}
    if len(eligible)>=5 and len(set(int(assigned[i]) for i in eligible))>1:
        label_array=np.asarray([int(assigned[i]) for i in eligible],dtype=int)
        confounds={
          'region_order_nmi':round(normalized_mutual_information([str(rows[i]['region_index']) for i in eligible],label_array),5),
          'recording_family_nmi':round(normalized_mutual_information([str(rows[i]['clip_group']) for i in eligible],label_array),5),
          'bank_nmi':round(normalized_mutual_information([str(rows[i]['bank']) for i in eligible],label_array),5),
        }
    within=posthoc_position_diagnostics(rows,np.asarray(X,dtype=np.float64))
    summary={**s,**edgereport,'within_acoustic_order_posthoc':within,'confound_audit_posthoc':confounds,
             'strong_position_confound_warning':confounds.get('region_order_nmi',0)>=.5,
             'interpretation':('motifs_strongly_correlated_with_acoustic_event_order_NOT_distinct_phones'
                               if confounds.get('region_order_nmi',0)>=.5
                               else 'exploratory_repeated_acoustic_patterns_NOT_distinct_phones'),
             'schema':SCHEMA,'input_provenance':provenance,'input_source_zip_sha256':source_sha,
             'regions_total':len(rows), 'recording_families_total':len(set(r['clip_group'] for r in rows)),
             'motif_count_is_not_phone_count':True,'background_model_selection_is_retrospective':True,
             'user_unit_labels_used':False, 'oto_timestamps_used':False,
             'raw_audio_in_output':False, 'automatic_recording_scheme_changes':False}
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (output/'regions.json').write_text(json.dumps({'schema':SCHEMA,'regions':table},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    with (output/'cross_recording_links.csv').open('w',newline='',encoding='utf8') as stream:
        names=['source_region_a','source_region_b','motif_id','squared_scaled_spectral_distance','status']
        w=csv.DictWriter(stream,fieldnames=names);w.writeheader();w.writerows(edges)
    np.savez_compressed(output/'frozen_centers.npz',centers=hidden['centers'])
    return summary


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(prog='phonoweave v10 source-disjoint motif linking')
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument('--v9-results',type=Path)
    src.add_argument('--full-wav-zip',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    print(json.dumps(run(args.output,args.v9_results,args.full_wav_zip),ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())