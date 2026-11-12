"""Leakage-aware IPA-independent temporal acoustic representation diagnostic.

Reads original OTO source PCM matched against a frozen v2 batch index. Treats
those regions as UNVERIFIED sound candidates, not phoneme boundaries.
Only source-WAV-template-disjoint development data select a representation.
No alias, OTO, transcript, voicebank PCM, or unit decision is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import wave
import zipfile

import numpy as np
from scipy.fft import dct, rfft
from scipy.signal import resample_poly

from .orthography_native_units import MEL

SCHEMA = 'phonoweave.user_native_temporal_eval.v3'
# Predeclared: model does not get to move OTO anchor in response to test labels.
SLICES_MS = {'far':(-340.,-260.), 'early':(-270.,-190.),
             'mid':(-200.,-120.), 'late':(-130.,-50.),
             'edge':(-80.,0.)}
MODES = ('original_70ms','early_context','wide_mean','wide_trajectory',
         'late_context','full_trajectory','stable_region')


def _checksum(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_v2(data_dir: Path):
    data_dir=Path(data_dir)
    with (data_dir/'source_index.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    feats=json.loads((data_dir/'feature_vectors.json').read_text(encoding='utf-8'))
    if len(rows)!=len(feats) or not rows:raise ValueError('v2 candidate/features length mismatch')
    for row,ft in zip(rows,feats):
        if (row['bank'],row['source_wav'],int(row['oto_line']))!=(ft['bank'],ft['source_wav'],int(ft['oto_line'])):
            raise ValueError('v2 row/feature provenance mismatch')
        if any(token in row['source_wav'] for token in ('/','\\','..')) or not row['source_wav'].lower().endswith('.wav'):
            raise ValueError('unsafe source WAV name in frozen v2 index')
        row['features_original']=ft['features']
        if len(row['features_original'])!=12:raise ValueError('expected original 12D feature')
    keys=[(r['bank'],r['source_wav'],r['oto_line']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate v2 candidate')
    return rows


def _wav_bytes(raw):
    with wave.open(io.BytesIO(raw),'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getcomptype()!='NONE':
            raise ValueError('source must be mono PCM16')
        rate=w.getframerate()
        if rate not in (44100,48000,96000):raise ValueError('unsupported source rate')
        frames=w.readframes(w.getnframes())
    return np.frombuffer(frames,dtype='<i2').astype(np.float32)/32768,rate,sha256(frames).hexdigest()


def extract_temporal(rows:list[dict],manifest:dict):
    """Compute all candidate views in batches per source WAV, SHA bound to v2."""
    lookup={b['style']+'_'+b['pitch']:b for b in manifest['banks']}
    by_bank=defaultdict(lambda:defaultdict(list))
    for ix,row in enumerate(rows):by_bank[row['bank']][row['source_wav']].append(ix)
    n=len(rows);jnames=list(SLICES_MS);nwin=len(jnames)
    views={name:np.zeros((n,12),dtype=np.float32) for name in jnames}
    energies=np.full((n,nwin),-120.,dtype=np.float32)
    variance=np.full((n,nwin),1.e9,dtype=np.float32)
    valid=np.zeros((n,nwin),dtype=bool)
    mel=MEL.astype(np.float32,copy=False)
    hann=np.hanning(400).astype(np.float32)
    for bank in sorted(by_bank):
        b=lookup.get(bank)
        if b is None:raise ValueError('missing frozen bank '+bank)
        archive=Path(b['archive']);archive_sha=_checksum(archive)
        if any(rows[ix]['source_zip_sha256']!=archive_sha for source in by_bank[bank].values() for ix in source):
            raise ValueError('source ZIP SHA-256 changed for '+bank)
        with zipfile.ZipFile(archive) as z:
            raw_oto=z.read(b['root']+'oto.ini')
            try:oto_text=raw_oto.decode('utf-8-sig')
            except UnicodeDecodeError:oto_text=raw_oto.decode('cp932')
            source_lines=oto_text.splitlines()
            for wav,indices in sorted(by_bank[bank].items()):
                # Source hashes alone do not authenticate label text. The
                # claimed user label must round-trip to this exact OTO row.
                for ix in indices:
                    item=rows[ix];line_no=int(item['oto_line'])
                    if line_no<1 or line_no>len(source_lines):
                        raise ValueError('OTO row number outside original source')
                    original=source_lines[line_no-1]
                    if '=' not in original:raise ValueError('invalid original OTO row')
                    original_wav,tail=original.split('=',1)
                    fields=tail.rsplit(',',5)
                    if len(fields)!=6:raise ValueError('invalid original OTO time fields')
                    alias=fields[0]
                    if (original_wav!=wav or alias!=item['source_alias']
                            or not alias.startswith(b['prefix'])
                            or alias[len(b['prefix']):].split()!=[item['symbol'],item['context']]):
                        raise ValueError('source OTO alias/label does not match original')
                    times=[float(q) for q in fields[1:]]
                    anchor=times[0]+(times[3]+times[4])/2
                    if abs(float(item['start_ms'])-(anchor-125))>.12:
                        raise ValueError('source OTO anchor does not match original window')
                pcm,rate,pcmsha=_wav_bytes(z.read(b['root']+wav))
                if any(rows[ix]['source_pcm_sha256']!=pcmsha for ix in indices):
                    raise ValueError('source PCM SHA-256 changed: '+bank+'/'+wav)
                g=math.gcd(rate,16000)
                signal=resample_poly(pcm,16000//g,rate//g).astype(np.float32)
                pre=np.empty_like(signal)
                pre[0]=signal[0];pre[1:]=signal[1:]-.97*signal[:-1]
                # Each candidate contributes six 25 ms frames per 80 ms window.
                frames=[];meta=[]
                for ix in indices:
                    row=rows[ix]
                    anchor=float(row['start_ms'])+125.
                    for j,name in enumerate(jnames):
                        lo,hi=SLICES_MS[name]
                        l=round((anchor+lo)*16);r=round((anchor+hi)*16)
                        if l<0 or r>len(signal) or r-l<1040:continue
                        segment=signal[l:r]
                        rms=float(np.sqrt(np.mean(segment*segment)))
                        energies[ix,j]=20*np.log10(max(rms,1e-10))
                        if rms<0.002:continue
                        xs=np.lib.stride_tricks.sliding_window_view(pre[l:r],400)[::160]
                        if len(xs)<2:continue
                        frames.append(xs.copy());meta.append((ix,j,len(xs)))
                if not frames:continue
                flat=np.concatenate(frames,axis=0)
                flat=(flat-flat.mean(axis=1,keepdims=True))*hann[None,:]
                power=np.abs(rfft(flat,n=1024,axis=1))**2
                cep=dct(np.log(np.maximum(power.astype(np.float32)@mel.T,1e-9)),type=2,norm='ortho',axis=1)[:,1:13]
                pos=0
                for ix,j,count in meta:
                    sub=cep[pos:pos+count];pos+=count
                    center=np.median(sub,axis=0)
                    center=center/max(1.,float(np.linalg.norm(center)))
                    x=sub/np.maximum(1.,np.linalg.norm(sub,axis=1,keepdims=True))
                    views[jnames[j]][ix]=center
                    variance[ix,j]=float(np.median(np.linalg.norm(x-center[None,:],axis=1)))
                    valid[ix,j]=True
    return views,energies,variance,valid


def construct_modes(rows:list[dict],views:dict,energies:np.ndarray,variance:np.ndarray,valid:np.ndarray):
    base=np.asarray([r['features_original'] for r in rows],dtype=float)
    # No supervised temporal selection. Fixed combinations known before evaluation.
    def concat(*names):
        return np.concatenate([views[k] for k in names],axis=1)/np.sqrt(len(names))
    def mean(*names):
        x=np.mean([views[k] for k in names],axis=0)
        return x/np.maximum(1,np.linalg.norm(x,axis=1,keepdims=True))
    # Select a 80ms continuous, non-silent, locally stable region WITHOUT labels.
    # This is not a guaranteed realization or phone core.
    subset=[0,1,2,3] # exclude edge near next phone
    stable_score=variance[:,subset].copy()
    stable_score+=np.maximum(-35-energies[:,subset],0)*0.04
    stable_score[~valid[:,subset]]=1e9
    chosen=np.argmin(stable_score,axis=1)
    stable=np.stack([views[j] for j in list(SLICES_MS)],axis=1)[np.arange(len(rows)),chosen]
    stable[~np.any(valid[:,subset],axis=1)]=base[~np.any(valid[:,subset],axis=1)]
    return {'original_70ms':base,
            'early_context':concat('far','early','mid'),
            'wide_mean':mean('early','mid','late'),
            'wide_trajectory':concat('early','mid','late'),
            'late_context':concat('mid','late','edge'),
            'full_trajectory':concat('far','early','mid','late','edge'),
            'stable_region':stable},chosen


def normalize_bank(X:np.ndarray,bank_ids:np.ndarray):
    """Unlabeled bank centering, including evaluation bank; transductive by design."""
    out=X.copy()
    for b in sorted(set(bank_ids)):
        idx=bank_ids==b;out[idx]-=out[idx].mean(axis=0)
    return out


def classify_centroid(X:np.ndarray,y:np.ndarray,train:np.ndarray,test:np.ndarray):
    """Same nearest-mean policy as v2; labels never used to build held-out mean."""
    names=np.unique(y[train]);centroids=np.stack([X[train][y[train]==v].mean(axis=0) for v in names])
    dist=np.sum((X[test,None,:]-centroids[None,:,:])**2,axis=2)
    return names[np.argmin(dist,axis=1)]


def center_using_training_only(X:np.ndarray,bank_ids:np.ndarray,train:np.ndarray):
    """Fit a bank mean ONLY on its training samples, never on test examples.

    For a wholly unseen bank, fall back to training-set global mean. This
    is a conservative, inductive model and explicitly exposes style shift.
    """
    if not np.any(train):raise ValueError('empty training partition')
    centered=X.copy()
    overall=X[train].mean(axis=0)
    for bank in sorted(set(bank_ids)):
        in_bank=(bank_ids==bank)
        bank_train=in_bank & train
        mean=X[bank_train].mean(axis=0) if np.any(bank_train) else overall
        centered[in_bank]-=mean
    return centered


def score_templates(X,y,groups,banks,train_groups,test_groups):
    """Train-only normalization and fixed centroid labels per group exclusion."""
    train=np.isin(groups,train_groups);tested=np.isin(groups,test_groups)
    if np.any(train&tested):raise ValueError('source template leakage')
    eligible=tested & np.isin(y,np.unique(y[train]));test=np.flatnonzero(eligible)
    if not test.size:raise ValueError('no eligible held-out observations')
    Z=center_using_training_only(X,banks,train)
    predicted=classify_centroid(Z,y,np.flatnonzero(train),test)
    true=y[test];correct=predicted==true
    all_labels=sorted(set(y[test]))
    per_label={label:(int(np.sum(correct[true==label])),int(np.sum(true==label))) for label in all_labels}
    macro=float(np.mean([a/b for a,b in per_label.values()]))
    per_group={group:{'correct':int(np.sum(correct[groups[test]==group])),
                      'tested':int(np.sum(groups[test]==group))} for group in test_groups if np.any(groups[test]==group)}
    return {'correct':int(np.sum(correct)),'tested':int(len(test)),
            'accuracy':round(float(np.mean(correct)),5),'macro_recall':round(macro,5),
            'per_group':per_group,'per_label':{k:{'correct':a,'tested':b} for k,(a,b) in per_label.items()}}


def _folded_score(X,y,groups,banks,available_groups):
    """Pooled results across group-held-out folds, with per-label macro recall."""
    all_groups=sorted(set(available_groups));count=Counter();total=Counter();by_group={}
    for held in all_groups:
        scores=score_templates(X,y,groups,banks,[g for g in all_groups if g!=held],[held])
        by_group[held]=scores['per_group'][held]
        for name,c in scores['per_label'].items():
            count[name]+=c['correct'];total[name]+=c['tested']
    n=sum(total.values());c=sum(count.values())
    return {'correct':c,'tested':n,'accuracy':round(c/n,5) if n else None,
            'macro_recall':round(float(np.mean([count[k]/total[k] for k in total])),5) if n else None,
            'per_group':by_group}


def dev_selection(Xs:dict,rows:list[dict],fixed_test_groups:set[str]):
    y=np.asarray([r['symbol'] for r in rows]);groups=np.asarray([r['source_wav'] for r in rows])
    banks=np.asarray([r['bank'] for r in rows]);all_groups=sorted(set(groups))
    dev_groups=[x for x in all_groups if x not in fixed_test_groups]
    if len(dev_groups)<5 or len(fixed_test_groups)<3:raise ValueError('too few independent groups')
    dev={name:_folded_score(Xs[name],y,groups,banks,dev_groups) for name in MODES}
    # Predeclared primary endpoint: pooled macro recall; accuracy breaks ties.
    selected=sorted(MODES,key=lambda k:(-dev[k]['macro_recall'],-dev[k]['accuracy'],MODES.index(k)))[0]
    # Evaluate only once on final frozen 8 WAV-template groups.
    final={k:score_templates(Xs[k],y,groups,banks,dev_groups,sorted(fixed_test_groups))
           for k in ('original_70ms',selected)}
    return {'schema':SCHEMA,'candidate_modes':list(MODES),'dev_groups':dev_groups,
            'frozen_test_groups':sorted(fixed_test_groups),'dev_scores':dev,
            'selected_mode':selected,'heldout':final,
            'source_template_grouping':'exact source WAV basename across every bank',
            'normalization':'training-only per-bank centering; wholly unseen banks use pooled train center',
            'selection_rule':'highest development pooled macro recall, then development accuracy; frozen test never used for selection',
            'status':'DESCRIPTIVE_EXISTING_USER_LABELS_NOT_PHONE_BOUNDARIES_OR_SPLIT_DECISIONS'}


def robustness_checks(rows:list[dict],modes:dict,selected:str,report:dict):
    y=np.asarray([r['symbol'] for r in rows]);banks=np.asarray([r['bank'] for r in rows]);
    def score_split(group_name):
        groups=np.asarray([r[group_name] for r in rows])
        unique=sorted(set(groups))
        return {name:_folded_score(modes[name],y,groups,banks,unique)
                for name in ('original_70ms',selected)}
    checks={'source_wav_template':score_split('source_wav'),
            'following_context':score_split('context'),
            'bank':score_split('bank'), 'voice_style':score_split('style')}
    gs=report['frozen_test_groups']
    baseline=report['heldout']['original_70ms']['per_group']
    improved=report['heldout'][selected]['per_group']
    count0=np.array([baseline[g]['correct'] for g in gs]);count1=np.array([improved[g]['correct'] for g in gs])
    sizes=np.array([baseline[g]['tested'] for g in gs]);rng=np.random.default_rng(20261110)
    draws=rng.integers(0,len(gs),size=(25000,len(gs)))
    diffs=(count1[draws]-count0[draws]).sum(axis=1)/sizes[draws].sum(axis=1)
    checks['frozen_test_group_bootstrap']={'ci95_absolute_accuracy_change':np.quantile(diffs,[.025,.975]).tolist(),
        'positive_groups':int(np.sum(count1>count0)),'negative_groups':int(np.sum(count1<count0)),
        'tied_groups':int(np.sum(count1==count0)),'resample_unit':'source WAV template',
        'limitation':'not independent singers, sessions or production systems'}
    return checks


def frozen_test_templates(rows:list[dict]):
    groups=sorted({r['source_wav'] for r in rows})
    # Deterministic selection is independent of feature/label/ground-truth scores.
    order=sorted(groups,key=lambda name:sha256(('phonoweave-v3-holdout|'+name).encode()).hexdigest())
    n=max(5,round(.32*len(groups)))
    return set(order[:n])


def run(batch_dir:Path,manifest:dict,output:Path):
    if output.exists():raise FileExistsError('refusing overwrite output')
    rows=load_v2(batch_dir)
    views,energies,variance,valid=extract_temporal(rows,manifest)
    all_modes,chosen=construct_modes(rows,views,energies,variance,valid)
    result=dev_selection(all_modes,rows,frozen_test_templates(rows))
    checks=robustness_checks(rows,all_modes,result['selected_mode'],result)
    result['dataset']={'rows':len(rows),'source_templates':len(set(r['source_wav'] for r in rows)),
                       'bank_count':len(set(r['bank'] for r in rows)),
                       'user_label_count':len(set(r['symbol'] for r in rows)),
                       'temporal_windows_ms':SLICES_MS,'window_valid_counts':{name:int(np.sum(valid[:,i])) for i,name in enumerate(SLICES_MS)},
                       'stable_region_counts':dict(Counter(list(SLICES_MS)[i] for i in chosen))}
    result['dataset']['v2_source_index_sha256']=_checksum(Path(batch_dir)/'source_index.csv')
    result['dataset']['v2_feature_vectors_sha256']=_checksum(Path(batch_dir)/'feature_vectors.json')
    result['no_ipa_required']=True
    result['no_new_user_aliases_created']=True
    result['all_true_phone_boundaries_confirmed']=0
    output.mkdir(parents=True)
    (output/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'robustness.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Save compact cached features for reproduction from exact rows. Pure numerical data.
    np.savez_compressed(output/'temporal_features.npz',**{f'feat_{key}':v for key,v in all_modes.items()})
    (output/'feature_cache_sha256.txt').write_text(_checksum(output/'temporal_features.npz')+'  temporal_features.npz\n',encoding='ascii')
    with (output/'holdout_by_source_wav.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(('source_wav','partition'))
        for group in sorted(set(r['source_wav'] for r in rows)):
            w.writerow((group,'final_holdout' if group in result['frozen_test_groups'] else 'development'))
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description='User-native temporal context test on v2 frozen candidate inventory')
    ap.add_argument('--batch-result',required=True,type=Path)
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--archive-root',type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args(argv)
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    if args.archive_root:
        for b in manifest['banks']:b['archive']=str(args.archive_root/Path(b['archive']).name)
    result=run(args.batch_result,manifest,args.output)
    print(json.dumps({'selected_mode':result['selected_mode'],
                      'development':{k:result['dev_scores'][result['selected_mode']][k] for k in ('accuracy','macro_recall','tested')},
                      'heldout':{k:{kk:v[kk] for kk in ('accuracy','macro_recall','correct','tested')} for k,v in result['heldout'].items()}},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
