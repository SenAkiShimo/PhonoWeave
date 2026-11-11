"""v13: equal-donor-budget acoustic continuation proxy (NOT rendered synthesis).

Predict the third unanchored audio-only region from earlier regions in *the same
held-out audio clip*. This is an optimistic **oracle-context** diagnostic: a real
synthesizer does not have access to the target's prerecorded context audio.
Only train clips are donors; the held-out third region is used for scoring ONLY.
No phoneme identity, IPA, alias, OTO timing, or inventory edits are produced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .native_motif_link_v10 import group_split, load_v9
from .native_nuisance_v12 import validate_source

SCHEMA = 'phonoweave.equal_budget_continuation_proxy.v1.3'
SEED = 20261113
BUDGETS = (1, 2, 3, 4, 6)
MODES = ('preceding_0', 'preceding_1', 'preceding_both')
NULL_TRIALS = 64
MIN_VAL_GAIN = 0.05
MIN_GROUPS_BETTER = 0.6
MAX_NULL_P = 0.05


@dataclass(frozen=True)
class Clip:
    bank: str
    group: str
    wav: str
    sha: str
    stages: np.ndarray  # (3,48), unanchored acoustic stages; not phones


def build_clips(rows: list[dict], X: np.ndarray) -> tuple[list[Clip], dict]:
    """Require a complete, source-hash-consistent 0/1/2 sequence in one WAV."""
    validate_source(rows, X)
    entries: dict[tuple[str,str,str], dict[int,np.ndarray]] = defaultdict(dict)
    hashes: dict[tuple[str,str,str], str] = {}
    for row, vector in zip(rows, X):
        key=(str(row['bank']),str(row['clip_group']),str(row['wav_archive_entry']))
        sha=str(row['source_pcm_sha256']).lower()
        if key in hashes and hashes[key]!=sha:raise ValueError('conflicting PCM hashes within one WAV')
        hashes[key]=sha
        ordinal=int(row['region_index'])
        if ordinal in entries[key]:raise ValueError('duplicate stage index in WAV')
        entries[key][ordinal]=np.asarray(vector,dtype=np.float64)
    out=[];skipped=0
    for key in sorted(entries):
        stages=entries[key]
        if sorted(stages)!=[0,1,2]:
            skipped+=1
            continue
        out.append(Clip(key[0],key[1],key[2],hashes[key],np.stack([stages[i] for i in range(3)])))
    if not out:raise ValueError('no three-stage unanchored source clips')
    # Do not treat multiple copies of identical audio as independent.
    digests=defaultdict(set)
    for c in out:digests[c.sha].add(c.group)
    if any(len(g)>1 for g in digests.values()):raise ValueError('same PCM reused across source groups')
    return out,{'complete_three_stage_clips':len(out),'incomplete_clips_skipped':skipped}


def freeze_partitions(clips:list[Clip]) -> dict[str,np.ndarray]:
    # Reuse exact v10/v11/v12 source-family split, never split one group by bank.
    proxy=[{'clip_group':c.group} for c in clips]
    train,val,audit=group_split(proxy)
    splits={'train':train,'validation':val,'audit':audit}
    groups={k:{c.group for c,yes in zip(clips,mask) if yes} for k,mask in splits.items()}
    if any(groups[a] & groups[b] for a,b in [('train','validation'),('train','audit'),('validation','audit')]):
        raise AssertionError('source-family leakage')
    return splits


def stage_normalize(clips:list[Clip],train:np.ndarray) -> tuple[np.ndarray,dict]:
    if train.sum()<20:raise ValueError('insufficient training samples')
    X=np.stack([c.stages for c in clips]).astype(np.float64)
    if X.shape!=(len(clips),3,48) or not np.isfinite(X).all():
        raise ValueError('nonfinite stage features')
    mu=X[train].mean(axis=0)
    std=X[train].std(axis=0)
    floor=np.maximum(.04, .20*std.max(axis=1,keepdims=True))
    scale=np.maximum(std,floor)
    Z=np.clip((X-mu)/scale,-6,6)
    return Z,{'fitted_on_train_only':True,'stage_independent_scaling':True,
              'clip_count':len(clips),'per_stage_min_scale':np.min(scale,axis=1).round(5).tolist()}


def donor_pool(clips:list[Clip], train:np.ndarray, bank:str, excluded_group:str|None=None)->np.ndarray:
    ids=[i for i,c in enumerate(clips) if train[i] and c.bank==bank and c.group!=excluded_group]
    if len(ids)<8:raise ValueError('insufficient independent same-bank training donors: '+bank)
    if len({clips[i].group for i in ids})<8:raise ValueError('duplicate source families cannot serve as donor replication')
    return np.array(ids,dtype=int)


def rank_donors(Z:np.ndarray,clips:list[Clip],train:np.ndarray,idx:int,
                mode:str,k:int,context_override:np.ndarray|None=None)->np.ndarray:
    """Donor selection *cannot* access Z[idx,2]; target only in score().

    Baseline and context arms use exactly k donors from the same eligible pool.
    `mode=pooled` chooses training stage-2 medoid-nearest donors; context arms
    choose from heldout stages 0/1 (the unrealistically available oracle context).
    """
    if k not in BUDGETS or mode not in (*MODES,'pooled'):
        raise ValueError('unsupported budget or context mode')
    if train[idx]:raise ValueError('target cannot be a training donor')
    c=clips[idx]
    ids=donor_pool(clips,train,c.bank,excluded_group=c.group)
    if mode=='pooled':
        center=Z[ids,2].mean(axis=0)
        d=np.sum((Z[ids,2]-center)**2,axis=1)
    else:
        stages={'preceding_0':(0,),'preceding_1':(1,),'preceding_both':(0,1)}[mode]
        q=np.asarray(context_override if context_override is not None else Z[idx,:2],dtype=np.float64)
        if q.shape!=(2,48):raise ValueError('context override needs two 48D preceding stages')
        d=np.mean([np.sum((Z[ids,j]-q[j])**2,axis=1) for j in stages],axis=0)
    order=sorted(range(len(ids)),key=lambda t:(float(d[t]),clips[int(ids[t])].group,clips[int(ids[t])].wav))
    return ids[order[:k]]


def _grouped_losses(clips:list[Clip],Z:np.ndarray,train:np.ndarray,mask:np.ndarray,
                    mode:str,k:int,shuffled_context:dict[int,np.ndarray]|None=None)->tuple[dict[str,float],list[dict]]:
    agg=defaultdict(list);records=[]
    for i in np.flatnonzero(mask):
        i=int(i);c=clips[i]
        ids=rank_donors(Z,clips,train,i,mode,k,context_override=None if shuffled_context is None else shuffled_context.get(i))
        assert all(clips[d].group!=c.group and train[d] for d in ids)
        prediction=np.mean(Z[ids,2],axis=0)
        error=float(np.mean((prediction-Z[i,2])**2))
        agg[c.group].append(error)
        records.append({'source_group':c.group,'bank':c.bank,'source_pcm_sha256':c.sha,
                        'donors':[clips[int(d)].wav for d in ids],
                        'donor_source_group_count':len({clips[int(d)].group for d in ids}),
                        'loss':error})
    return {g:float(np.mean(vals)) for g,vals in agg.items()},records


def score(clips:list[Clip],Z:np.ndarray,train:np.ndarray,mask:np.ndarray,mode:str,k:int,
          shuffled_context:dict[int,np.ndarray]|None=None)->dict:
    by_group,records=_grouped_losses(clips,Z,train,mask,mode,k,shuffled_context)
    if len(by_group)<3:raise ValueError('insufficient independently held source groups')
    return {'group_mean_mse':float(np.mean(list(by_group.values()))),
            'group_losses':by_group,'regions_scored':len(records),
            'source_groups_scored':len(by_group),'records':records}


def permutation_context(clips:list[Clip],Z:np.ndarray,mask:np.ndarray,seed:int)->dict[int,np.ndarray]:
    """Shuffle only input contexts, within bank, across held source families."""
    rng=np.random.default_rng(seed);lookup={}
    for bank in sorted({c.bank for c in clips}):
        ids=np.array([i for i,c in enumerate(clips) if mask[i] and c.bank==bank],dtype=int)
        if len(ids)<3:continue
        perm=rng.permutation(ids)
        # A non-zero cyclic shift of a permutation is a strict derangement.
        shift=int(rng.integers(1,len(perm)))
        for i,j in zip(perm,np.roll(perm,shift)):
            assert int(i)!=int(j)
            lookup[int(i)]=Z[int(j),:2].copy()
    return lookup


def select_on_validation(clips:list[Clip],Z:np.ndarray,splits:dict[str,np.ndarray],
                         null_trials:int=NULL_TRIALS)->dict:
    train,val=splits['train'],splits['validation']
    baselines={k:score(clips,Z,train,val,'pooled',k) for k in BUDGETS}
    candidates=[]
    for k in BUDGETS:
        base=baselines[k]
        for mode in MODES:
            alt=score(clips,Z,train,val,mode,k)
            gain=1-alt['group_mean_mse']/max(base['group_mean_mse'],1e-12)
            keys=sorted(base['group_losses'])
            fraction=float(np.mean([alt['group_losses'][g]<base['group_losses'][g] for g in keys]))
            candidates.append({'mode':mode,'budget':k,'validation_group_mean_mse':alt['group_mean_mse'],
                               'baseline_group_mean_mse':base['group_mean_mse'],
                               'relative_gain':float(gain),'groups_better_fraction':fraction})
    best=max(candidates,key=lambda r:(r['relative_gain'],-r['budget'],r['mode']))
    null=[]
    for rep in range(null_trials):
        permutation=permutation_context(clips,Z,val,SEED+rep)
        shuffled=[]
        for c in candidates:
            alt=score(clips,Z,train,val,c['mode'],c['budget'],permutation)
            base=baselines[c['budget']]
            shuffled.append(1-alt['group_mean_mse']/max(base['group_mean_mse'],1e-12))
        null.append(max(shuffled))
    p=(1+sum(v>=best['relative_gain']-1e-12 for v in null))/(1+len(null))
    passes=bool(best['relative_gain']>=MIN_VAL_GAIN and
                best['groups_better_fraction']>=MIN_GROUPS_BETTER and p<=MAX_NULL_P)
    return {'candidate':best,'development_screen_pass':passes,
            'max_across_modes_and_budgets_permutation_p_exploratory':p,
            'permutation_null_trials':null_trials,'candidates':candidates,
            'selection_used_audit':False}


def _paired_summary(a:dict,b:dict)->dict:
    keys=sorted(set(a['group_losses'])&set(b['group_losses']))
    base=np.array([a['group_losses'][g] for g in keys],float)
    alternative=np.array([b['group_losses'][g] for g in keys],float)
    if len(keys)<3:raise ValueError('not enough paired groups')
    gain=1-alternative.mean()/max(base.mean(),1e-12)
    # This interval is descriptive; source groups, not banks, are resampled.
    rng=np.random.default_rng(SEED)
    boots=[]
    for _ in range(1000):
        picks=rng.integers(len(keys),size=len(keys))
        boots.append(1-alternative[picks].mean()/max(base[picks].mean(),1e-12))
    return {'paired_source_groups':len(keys),'matched_observations':a['regions_scored'],
            'pooled_group_mean_mse':round(float(base.mean()),6),
            'context_group_mean_mse':round(float(alternative.mean()),6),
            'relative_improvement':round(float(gain),6),
            'groups_where_context_better':sum(bool(x<y) for x,y in zip(alternative,base)),
            'source_group_resampled_95pct_descriptive_interval':list(np.quantile(boots,[.025,.975]).round(6)),
            'not_a_synthesis_quality_score':True}


def verify_v12(source:dict,summary_path:Path|None)->dict:
    if summary_path is None:return {'v12_source_identity_verified':False}
    info=json.loads(Path(summary_path).read_text(encoding='utf8'))
    if info.get('schema')!='phonoweave.paired_bank_context_decomposition.v1.2':
        raise ValueError('incorrect v12 schema')
    if info.get('source_archive_sha256')!=source['archive_sha256']:
        raise ValueError('v12 source SHA256 mismatch')
    return {'v12_source_identity_verified':True,'v12_bank_correction_imported_or_applied':False}


def run(v9:Path,output:Path,v12_summary:Path|None=None,null_trials:int=NULL_TRIALS)->dict:
    v9=Path(v9);output=Path(output)
    if output.exists():raise FileExistsError('refusing to overwrite previous research run')
    if not (1<=null_trials<=512):raise ValueError('null_trials out of range')
    rows,X,source=load_v9(v9)
    provenance=verify_v12(source,v12_summary)
    clips,clip_summary=build_clips(rows,X)
    splits=freeze_partitions(clips)
    Z,norm=stage_normalize(clips,splits['train'])
    chosen=select_on_validation(clips,Z,splits,null_trials=null_trials)
    mode=chosen['candidate']['mode'];budget=chosen['candidate']['budget']
    base=score(clips,Z,splits['train'],splits['audit'],'pooled',budget)
    proposed=score(clips,Z,splits['train'],splits['audit'],mode,budget)
    audit=_paired_summary(base,proposed)
    report={'schema':SCHEMA,'source_archive_sha256':source['archive_sha256'],
            'provenance':'retrospective_OTO_curated_600ms_excerpt_source_AUDIO_features_only',
            'clip_inventory':clip_summary,'source_family_count':len({c.group for c in clips}),
            'bank_count':len({c.bank for c in clips}),
            'source_groups_by_split':{key:len({c.group for c,yes in zip(clips,mask) if yes}) for key,mask in splits.items()},
            'stage_normalization':norm,'v12_check':provenance,
            'development':{k:v for k,v in chosen.items() if k!='candidates'},
            'validation_model_candidates':chosen['candidates'],
            'audit_descriptive':audit,
            'source_bank_and_group_identifiers_not_phoneme_ground_truth':True,
            'query_uses_real_target_clip_previous_stages_not_available_to_a_text_only_synthesizer':True,
            'equal_donor_budget_and_train_only_donors':True,
            'no_actual_rendered_synthesis_or_listener_test':True,
            'verified_phone_identity_or_boundary':False,
            'automatic_unit_split_merge_authorized':False,
            'candidate_fails_audit_direction_check':bool(audit['relative_improvement']<=0),
            'automatic_candidate_deployment_authorized':False,
            'audit_previously_explored_in_prior_studies_not_confirmatory':True}
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    with (output/'audit_donor_comparison.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['source_group','bank','source_pcm_sha256','pooled_donor_wavs',
                                           'context_donor_wavs','pooled_spectral_mse','context_spectral_mse'])
        writer.writeheader()
        by_key={(r['source_group'],r['bank']):r for r in proposed['records']}
        for row in base['records']:
            alt=by_key[(row['source_group'],row['bank'])]
            writer.writerow({'source_group':row['source_group'],'bank':row['bank'],
                             'source_pcm_sha256':row['source_pcm_sha256'],
                             'pooled_donor_wavs':'|'.join(row['donors']),
                             'context_donor_wavs':'|'.join(alt['donors']),
                             'pooled_spectral_mse':round(row['loss'],8),
                             'context_spectral_mse':round(alt['loss'],8)})
    return report


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v9-results',type=Path,required=True)
    parser.add_argument('--v12-summary',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--null-trials',type=int,default=NULL_TRIALS)
    args=parser.parse_args(argv)
    report=run(args.v9_results,args.output,args.v12_summary,args.null_trials)
    print(json.dumps({'development':report['development'],'audit_descriptive':report['audit_descriptive']},
                     indent=2,ensure_ascii=False))
    return 0

if __name__=='__main__':
    raise SystemExit(main())