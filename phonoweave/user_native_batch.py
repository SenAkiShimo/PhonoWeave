"""Batch discovery of opaque user-defined acoustic unit surfaces.

Never normalizes input to IPA, expands alias labels into linguistic phones, or
infers a real boundary from OTO. These are source-linked provisional windows.
Classifier evaluations are diagnostics of existing author aliases, not phone
segmentation, split/merge efficacy, or new-unit approval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from collections import defaultdict, Counter
import wave
import zipfile

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from .orthography_native_units import acoustic_vector

SCHEMA = 'phonoweave.user_native_batch_input.v2'
OUTPUT = 'phonoweave.user_native_batch_evidence.v2'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_oto(raw: bytes, prefix: str) -> list[dict]:
    try: txt=raw.decode('utf-8-sig')
    except UnicodeDecodeError: txt=raw.decode('cp932')
    out=[]
    for line_no,line in enumerate(txt.splitlines(),1):
        if not line.strip(): continue
        if '=' not in line:raise ValueError(f'bad OTO line {line_no}')
        src,tail=line.split('=',1)
        fields=tail.rsplit(',',5)
        if len(fields)!=6:raise ValueError(f'bad OTO time fields line {line_no}')
        alias=fields[0]
        if not alias.startswith(prefix):continue
        try: values=[float(v) for v in fields[1:]]
        except ValueError:raise ValueError(f'bad OTO time values line {line_no}')
        if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite OTO')
        body=alias[len(prefix):]
        pieces=body.split()
        # Only two-token voicebank labels, NOT claimed to be two phonemes.
        if len(pieces)!=2 or pieces[0]=='-':continue
        if not src or '/' in src or '\\' in src or '..' in src or not src.lower().endswith('.wav'):
            raise ValueError('unsafe source WAV')
        out.append({'oto_line':line_no,'source_wav':src,'oto_alias':alias,'body':body,
                    'left':pieces[0],'right':pieces[1],'times':values})
    return out


def _get_wave(raw: bytes) -> tuple[np.ndarray,int,str]:
    with wave.open(io.BytesIO(raw),'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getcomptype()!='NONE':
            raise ValueError('PCM16 mono only')
        rate=w.getframerate()
        if rate not in (44100,48000,96000):raise ValueError('unsupported rate')
        data=w.readframes(w.getnframes())
    return np.frombuffer(data,dtype='<i2').astype(np.float64)/32768.,rate,hashlib.sha256(data).hexdigest()


def _validate_config(config: dict):
    if config.get('schema')!=SCHEMA:raise ValueError('unsupported schema')
    banks=config.get('banks')
    if not isinstance(banks,list) or not banks:raise ValueError('no banks')
    seen=set()
    for b in banks:
        if not all(isinstance(b.get(k),str) and b[k] for k in ('style','pitch','archive','root','prefix')):
            raise ValueError('bank metadata missing')
        bid=(b['style'],b['pitch'])
        if bid in seen:raise ValueError('duplicate bank')
        seen.add(bid)
        root=b['root']
        if root.startswith('/') or '..' in root.split('/') or not root.endswith('/'):
            raise ValueError('unsafe ZIP root')
    window=config.get('oto_relative_ms',[-125,-55])
    if not isinstance(window,list) or len(window)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in window):
        raise ValueError('invalid window')
    if not (-500<=window[0]<window[1]<=100 and window[1]-window[0]>=45):
        raise ValueError('unsafe candidate window')
    if not isinstance(config.get('language_id'),str) or not config['language_id'].strip():
        raise ValueError('language identifier required')
    return window


def extract_batch(config: dict) -> tuple[list[dict],list[dict],list[dict]]:
    window=_validate_config(config)
    selected=config.get('include_left_symbols')
    if selected is not None and (not isinstance(selected,list) or len(selected)!=len(set(selected)) or not all(isinstance(s,str) and s for s in selected)):
        raise ValueError('invalid user symbol whitelist')
    selected=set(selected) if selected is not None else None
    rows=[];blocked=[];banks=[]
    for b in config['banks']:
        archive=Path(b['archive']).expanduser().resolve(strict=True)
        style,pitch=b['style'],b['pitch']
        bank=style+'_'+pitch
        with zipfile.ZipFile(archive) as z:
            zset=set(z.namelist())
            path=b['root']+'oto.ini'
            if path not in zset:raise ValueError('OTO missing '+bank)
            oto=z.read(path)
            oto_all_rows=sum(bool(line.strip()) for line in oto.decode('utf-8-sig',errors='replace').splitlines())
            aliases=parse_oto(oto,b['prefix'])
            grouped=defaultdict(list)
            for row in aliases:
                if selected is None or row['left'] in selected:
                    grouped[row['source_wav']].append(row)
            accepted=0
            for src,records in sorted(grouped.items()):
                path=b['root']+src
                if path not in zset:
                    for row in records:blocked.append({'bank':bank,'alias':row['oto_alias'],'reason':'missing_source_wav'})
                    continue
                pcm,rate,pcm_hash=_get_wave(z.read(path))
                for row in records:
                    o,cons,cut,pre,overlap=row['times']
                    anchor=o+(pre+overlap)/2
                    l=round((anchor+window[0])*rate/1000)
                    r=round((anchor+window[1])*rate/1000)
                    if l<0 or r>len(pcm) or r-l<round(rate*.045):
                        blocked.append({'bank':bank,'alias':row['oto_alias'],'reason':'window_out_of_bounds'})
                        continue
                    result=acoustic_vector(pcm[l:r],rate)
                    if result is None:
                        blocked.append({'bank':bank,'alias':row['oto_alias'],'reason':'nonvoiced_or_short_provisional_window'})
                        continue
                    feature,quality=result
                    rows.append({'bank':bank,'style':style,'pitch':pitch,'symbol':row['left'],
                        'context':row['right'],'source_wav':src,'oto_line':row['oto_line'],
                        'source_alias':row['oto_alias'],'source_pcm_sha256':pcm_hash,
                        'start_ms':round(l*1000/rate,3),'end_ms':round(r*1000/rate,3),
                        'rms_dbfs':quality['rms_dbfs'],'features':feature.tolist(),
                        'source_zip':archive.name,'source_zip_sha256':None})
                    accepted+=1
            digest=sha256_file(archive)
            for row in rows:
                if row['bank']==bank:row['source_zip_sha256']=digest
            banks.append({'bank':bank,'style':style,'pitch':pitch,'archive_basename':archive.name,
                          'archive_sha256':digest,'oto_sha256':hashlib.sha256(oto).hexdigest(),
                          'total_oto_rows':oto_all_rows,
                          'eligible_alias_rows':sum(map(len,grouped.values())),'accepted_windows':accepted})
    if len({(r['bank'],r['source_wav'],r['oto_line']) for r in rows})!=len(rows):
        raise ValueError('duplicate physical OTO source rows')
    return rows,blocked,banks


def _centroid_classify(features: np.ndarray,labels:np.ndarray,train_idx:np.ndarray,test_idx:np.ndarray):
    names=sorted(set(labels[train_idx].tolist()))
    means=np.stack([features[train_idx][labels[train_idx]==k].mean(axis=0) for k in names])
    diff=features[test_idx,None,:]-means[None,:,:]
    nearest=np.argmin((diff*diff).sum(axis=2),axis=1)
    predicted=np.asarray(names)[nearest]
    correct=predicted==labels[test_idx]
    return predicted,correct


def _score(rows: list[dict], features: np.ndarray, kind: str):
    labels=np.asarray([r['symbol'] for r in rows]);
    if kind=='source_wav_template': groups=np.asarray([r['source_wav'] for r in rows])
    elif kind=='bank':groups=np.asarray([r['bank'] for r in rows])
    elif kind=='style':groups=np.asarray([r['style'] for r in rows])
    elif kind=='following_context':groups=np.asarray([r['context'] for r in rows])
    else: raise ValueError('bad split')
    names=sorted(set(labels));conf=Counter(); total=Counter(); correct=Counter(); holdouts=[]
    for group in sorted(set(groups)):
        test=np.flatnonzero(groups==group); train=np.flatnonzero(groups!=group)
        if not len(test) or not len(train):continue
        available=set(labels[train]); eligible=test[np.isin(labels[test],list(available))]
        if not len(eligible):continue
        pred,ok=_centroid_classify(features,labels,train,eligible)
        for idx,p,is_ok in zip(eligible,pred,ok):
            label=labels[idx];total[label]+=1;correct[label]+=int(is_ok)
            if not is_ok:conf[(label,p)]+=1
        holdouts.append({'group':group,'n_test':len(eligible),'correct':int(ok.sum())})
    n=sum(total.values());c=sum(correct.values())
    return {'split':kind,'accuracy':round(c/n,4) if n else None,'correct':c,'tested':n,
            'macro_recall':round(float(np.mean([correct[k]/total[k] for k in names if total[k]])),4) if n else None,
            'per_label':{k:{'tested':total[k],'correct':correct[k],
                            'recall':round(correct[k]/total[k],4) if total[k] else None} for k in names},
            'top_confusions':[{'label':a,'predicted':b,'count':v} for (a,b),v in conf.most_common(30)],
            'holdouts':holdouts}


def _silhouette(distance:np.ndarray,c:np.ndarray) -> float:
    unique=sorted(set(c.tolist()));scores=[]
    for i,group in enumerate(c):
        same=np.flatnonzero(c==group)
        if len(same)==1:scores.append(0.);continue
        a=distance[i,same[same!=i]].mean()
        b=min(distance[i,c==other].mean() for other in unique if other!=group)
        scores.append((b-a)/max(a,b,1e-12))
    return float(np.mean(scores))


def evaluate_batch(rows:list[dict],blocked:list[dict],banks:list[dict],config:dict)->dict:
    if not rows: raise ValueError('no acoustic rows to analyze')
    features=np.asarray([r['features'] for r in rows],dtype=float)
    bankkeys=np.asarray([r['bank'] for r in rows])
    # Unlabeled mean uses the evaluation bank: explicitly transductive analysis.
    centered=features.copy()
    for group in sorted(set(bankkeys)):
        idx=np.flatnonzero(bankkeys==group)
        centered[idx]-=centered[idx].mean(axis=0)
    labels=sorted({r['symbol'] for r in rows})
    context=defaultdict(set);source=defaultdict(set);banksets=defaultdict(set);counts=Counter()
    for r in rows:
        label=r['symbol']; counts[label]+=1;context[label].add(r['context'])
        source[label].add(r['source_wav']);banksets[label].add(r['bank'])
    scores={}
    for split in ('source_wav_template','bank','style','following_context'):
        scores[split]=_score(rows,centered,split)
    # Use per-(bank,label) means for manageable unsupervised audit, no audio labels enter clustering.
    grouped=defaultdict(list)
    for idx,r in enumerate(rows):grouped[(r['bank'],r['symbol'])].append(idx)
    avg=np.stack([centered[v].mean(axis=0) for v in grouped.values()])
    cluster=[]
    if len(avg)>=5:
        dist=squareform(pdist(avg,'euclidean'))
        h=linkage(avg,method='ward')
        for k in range(2,min(10,len(avg)-1)+1):
            cs=fcluster(h,k,criterion='maxclust')
            if len(set(cs))<2:continue
            cluster.append({'k':k,'silhouette':round(_silhouette(dist,cs),4)})
    # Honest nearest-neighbor labels based on across-bank means. Neither name nor IPA considered.
    proto={}
    for label in labels:
        ids=[i for i,r in enumerate(rows) if r['symbol']==label]
        proto[label]=centered[ids].mean(axis=0)
    nearest=[]
    for a in labels:
        others=sorted([(round(float(np.linalg.norm(proto[a]-proto[b])),5),b) for b in labels if b!=a])
        nearest.append({'symbol':a,'closest_other_user_symbols':[{'symbol':b,'distance':d} for d,b in others[:3]]})
    weakest=sorted(scores['source_wav_template']['per_label'].items(),
                   key=lambda item:(item[1]['recall'] is None, item[1]['recall'] or 0, item[0]))
    queue=[{'user_surface':name,'test_samples':val['tested'],
            'source_wav_template_holdout_recall':val['recall'],
            'reason':'low_cross_source_identifiability_review_only' if val['recall'] is not None and val['recall']<0.5 else 'comparison_baseline_review_only',
            'no_new_unit_authorized':True} for name,val in weakest]
    results={'schema':OUTPUT,'language_id':config['language_id'],'input_representation':'opaque_user_symbols_from_original_OTO_surfaces',
             'source_relation':'same_voicebank_diffSinger_rendered_styles_pitches_author_confirmed',
             'full_phone_boundaries_confirmed':0,
             'candidate_window_rule':{'anchor':'oto offset + (preutterance + overlap)/2',
                                      'relative_ms':config.get('oto_relative_ms',[-125,-55]),
                                      'status':'UNVERIFIED_OTO_RELATIVE_AUDIO_WINDOW'},
             'counts':{'banks':len(banks),'total_oto_rows':sum(b['total_oto_rows'] for b in banks),
                       'eligible_two_token_surface_rows':sum(b['eligible_alias_rows'] for b in banks),
                       'other_oto_rows_not_scanned':sum(b['total_oto_rows']-b['eligible_alias_rows'] for b in banks),
                       'source_audio_files_unique_within_bank':len(set((r['bank'],r['source_wav']) for r in rows)),
                       'user_surface_labels':len(labels),'distinct_following_context_surfaces':len(set(r['context'] for r in rows)),
                       'accepted_windows':len(rows),'abstained_windows':len(blocked)},
             'source_banks':banks,'review_queue':queue,
             'per_label':[{'symbol':label,'samples':counts[label],'contexts':len(context[label]),
                           'source_wav_templates':len(source[label]),'banks':len(banksets[label]),
                           'candidate_status':'descriptive_existing_surface_only'} for label in labels],
             'evaluation':scores,'nearest_other_existing_labels':nearest,
             'label_aggregated_clustering_audit':{'unit':'per-bank_per-user-surface_centroids',
                          'label_information_used_for_aggregation':True,
                          'k_scores':cluster,'k_selection_status':'diagnostic_only_no_unsupervised_discovery'},
             'unit_decision':'NO_AUTO_SPLIT_OR_MERGE',
             'proposed_new_aliases':[],
             'source_modified':False,
             'limitations':[
                 'These are author OTO label surfaces; they may not denote phonemes; left-side cannot be assumed to be a single phone.',
                 'OTO-relative windows are not verified vowel cores or phone boundaries for every context.',
                 'Unlabeled subbank centering uses statistics from the held-out group (transductive) for noise removal.',
                 'Different DiffSinger pitch subbanks are not independent takes; source-WAV-name groups do not guarantee independence.',
                 'Supervised accuracy only measures recovery of existing user alias symbols, NOT discovery of new phones.',
                 'Low within-WAV and cross-context contrast may reflect contextual coarticulation, not a need to merge units.',
                 'Aggregated clustering uses known user surface labels to form centroids; this is NOT label-free unit discovery.',
                 'Cluster count and nearest neighbors do not authorize synthesis inventory changes.',
             ]}
    return results


def run(config: dict,out:Path):
    if out.exists():raise FileExistsError('refusing overwrite: '+str(out))
    rows,blocked,banks=extract_batch(config)
    report=evaluate_batch(rows,blocked,banks,config)
    out.mkdir(parents=True)
    (out/'evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    fields=['bank','style','pitch','symbol','context','source_wav','oto_line','source_alias',
            'source_zip','source_zip_sha256','source_pcm_sha256','start_ms','end_ms','rms_dbfs']
    with (out/'source_index.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
    with (out/'abstentions.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['bank','alias','reason']);w.writeheader();w.writerows(blocked)
    # Features retained for future exact reproducibility, no audio embedded.
    with (out/'review_queue.csv').open('w',encoding='utf8',newline='') as f:
        fields=['user_surface','test_samples','source_wav_template_holdout_recall',
                'reason','no_new_unit_authorized']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(report['review_queue'])
    (out/'feature_vectors.json').write_text(json.dumps([{'bank':r['bank'],'source_wav':r['source_wav'],
        'oto_line':r['oto_line'],'features':r['features']} for r in rows],ensure_ascii=False)+'\n',encoding='utf8')
    return report


def main(argv=None):
    ap=argparse.ArgumentParser(description='No-IPA user-native whole-OTO batch acoustic screen')
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--archive-root',help='Resolve ZIP basenames in this folder instead of manifest absolute paths')
    a=ap.parse_args(argv)
    cfg=json.loads(Path(a.manifest).read_text(encoding='utf8'))
    if a.archive_root:
        for bank in cfg['banks']:
            bank['archive']=str(Path(a.archive_root)/Path(bank['archive']).name)
    report=run(cfg,Path(a.output))
    print(json.dumps({'counts':report['counts'],
                      'whole_source_wav_holdout':{k:report['evaluation']['source_wav_template'][k] for k in ('accuracy','tested','macro_recall')},
                      'unit_decision':report['unit_decision']},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
