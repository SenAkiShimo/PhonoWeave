"""Language-script-native, read-only acoustic contrast discovery.

Sound-unit names are opaque user-defined strings, NOT IPA. This module tests
whether AUTHOR-ALREADY-LABELED acoustic candidates generalize across held-out
pitch and following context, and never treats an OTO time or synthetic pitch
as phonetic ground truth. It does not mint/approve a new phone alias.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import wave
import zipfile

import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment

SCHEMA = "phonoweave.user_native_acoustic_contrast.v1"
OUT_SCHEMA = "phonoweave.user_native_acoustic_evidence.v1"


def _checksum(path: Path) -> str:
    h = sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _mel(f: np.ndarray) -> np.ndarray:
    return 2595 * np.log10(1 + f / 700)


def _melbank(sr: int = 16000, nfft: int = 1024, bands: int = 30) -> np.ndarray:
    frequencies = np.linspace(0, sr / 2, nfft//2 + 1)
    cuts = 700 * (10 ** (np.linspace(_mel(np.array([180]))[0],
        _mel(np.array([5500]))[0], bands+2)/2595) - 1)
    mat = np.zeros((bands, len(frequencies)))
    for i in range(bands):
        l,c,r = cuts[i:i+3]
        mat[i] = np.maximum(0, np.minimum((frequencies-l)/(c-l), (r-frequencies)/(r-c)))
        mat[i] /= max(np.sum(mat[i]), 1e-12)
    return mat


MEL = _melbank()


def acoustic_vector(pcm: np.ndarray, rate: int) -> tuple[np.ndarray, dict] | None:
    """Compute contrast features on a conservative provisional core.

    Crucially this is NOT a predicted onset/offset. It summarizes an explicitly
    selected OTO-relative comparison window. LPC/cepstral shape still retains
    pitch and source-condition confounds; held-out checks are mandatory.
    """
    if len(pcm) < round(.040*rate):
        return None
    x = np.asarray(pcm, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x*x)))
    if not math.isfinite(rms) or rms < 0.002:
        return None
    # Resampling to a common rate makes features portable across WAV formats.
    from math import gcd
    g = gcd(rate, 16000)
    x = resample_poly(x, 16000//g, rate//g)
    x = np.concatenate(([x[0]], x[1:]-.97*x[:-1]))
    nfft = 1024
    win = 400    # 25 ms
    hop = 160    # 10 ms
    frames=[]
    for pos in range(0, len(x)-win+1, hop):
        frame = x[pos:pos+win]
        power = np.abs(np.fft.rfft((frame-frame.mean())*np.hanning(win), nfft))**2
        logmel = np.log(MEL @ power + 1e-9)
        coef = dct(logmel, type=2, norm='ortho')
        frames.append(coef[1:13])  # remove absolute energy coefficient
    if len(frames) < 2:
        return None
    feature = np.median(np.stack(frames), axis=0)
    # Avoid look-ahead: normalize using the entire fixed feature vector only.
    feature /= max(1.,np.linalg.norm(feature))
    return feature, {'rms_dbfs':round(20*np.log10(rms),3),
                     'feature_name':'mel_dct_1_12_l2_unit', 'num_frames':len(frames)}


def _read_wav(raw: bytes) -> tuple[np.ndarray,int,str]:
    with wave.open(io.BytesIO(raw),'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getcomptype()!='NONE':
            raise ValueError('expected mono PCM16 WAV')
        rate=w.getframerate()
        if rate not in (44100,48000,96000):
            raise ValueError('unsupported sample rate')
        pcm=w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype='<i2').astype(np.float64)/32768,rate,sha256(pcm).hexdigest()


def _parse_oto(raw: bytes, prefix: str):
    try: txt=raw.decode('utf-8-sig')
    except UnicodeDecodeError: txt=raw.decode('cp932')
    rows=defaultdict(list)
    for lineno,line in enumerate(txt.splitlines(),1):
        if not line.strip(): continue
        wav,tail=line.split('=',1)
        alias,*fields=tail.rsplit(',',5)
        if len(fields)!=5: raise ValueError(f'invalid oto row {lineno}')
        vals=list(map(float, fields))
        if not all(map(math.isfinite, vals)):
            raise ValueError('nonfinite oto time')
        if not alias.startswith(prefix):continue
        rows[(wav, alias[len(prefix):])].append((lineno,alias,vals))
    return rows


def extract(config: dict) -> tuple[list[dict],list[dict]]:
    if config.get('schema')!=SCHEMA: raise ValueError('unsupported input schema')
    language=config.get('language_id')
    base=config.get('user_base_symbol')
    groups=config.get('user_realizations')
    contexts=config.get('contexts')
    banks=config.get('banks')
    if not all(isinstance(s,str) and s for s in (language,base)):
        raise ValueError('language_id and user_base_symbol required')
    if not isinstance(groups,list) or len(groups)<1 or len(set(groups))!=len(groups) or not all(isinstance(s,str) and s for s in groups):
        raise ValueError('at least one unique user label required')
    if not isinstance(contexts,list) or len(contexts)<2 or len(set(contexts))!=len(contexts):
        raise ValueError('at least two different following contexts required')
    if not isinstance(banks,list) or not banks:raise ValueError('banks required')
    win=config.get('oto_relative_ms',[-125,-55])
    if not (isinstance(win,list) and len(win)==2 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in win)
            and -500 <= win[0] < win[1] <= 100 and win[1]-win[0]>=45):
        raise ValueError('invalid candidate window')
    specs=config.get('observations')
    if specs is None:
        # Back-compatible convenience for CVVC "left right" OTO aliases.
        # The core interface is the explicit, language-agnostic observations.
        source_map=config.get('source_wavs_by_context',{})
        specs=[]
        for ctx in contexts:
            for symbol in groups:
                source=source_map.get(ctx)
                if isinstance(source,dict):source=source.get(symbol)
                specs.append({'user_symbol':symbol,'context_id':ctx,
                              'source_wav':source,'oto_alias_body':f'{symbol} {ctx}'})
    if not isinstance(specs,list) or not specs:
        raise ValueError('nonempty explicit observation list required')
    for spec in specs:
        if not isinstance(spec,dict) or spec.get('user_symbol') not in groups or spec.get('context_id') not in contexts:
            raise ValueError('observation has unknown symbol/context')
        source=spec.get('source_wav')
        if not isinstance(source,str) or '/' in source or '\\' in source or '..' in source or not source.endswith('.wav'):
            raise ValueError('unsafe source WAV')
        if not isinstance(spec.get('oto_alias_body'),str) or not spec['oto_alias_body']:
            raise ValueError('explicit original alias body required')
        if 'oto_line' in spec and (not isinstance(spec['oto_line'],int) or spec['oto_line']<=0):
            raise ValueError('invalid source oto line')
    records=[];abstentions=[]
    seen=set()
    source_records=set()
    for b in banks:
        style,pitch=b['style'],b['pitch']
        bid=f'{style}_{pitch}'
        if bid in seen:raise ValueError('duplicate subbank')
        seen.add(bid)
        archive=Path(b['archive']).expanduser().resolve(strict=True)
        root=b['root'];prefix=b['prefix']
        if '..' in Path(root).parts or root.startswith('/') or not root.endswith('/'):
            raise ValueError('unsafe subbank root')
        archive_sha=_checksum(archive)
        with zipfile.ZipFile(archive) as z:
            rows=_parse_oto(z.read(root+'oto.ini'),prefix)
            sources={}
            for spec in specs:
                symbol=spec['user_symbol'];ctx=spec['context_id'];source=spec['source_wav']
                if source not in sources:
                    sources[source]=_read_wav(z.read(root+source))
                pcm,rate,pcmsha=sources[source]
                key=(source,spec['oto_alias_body'])
                matches=rows.get(key,[])
                if 'oto_line' in spec:
                    matches=[x for x in matches if x[0]==spec['oto_line']]
                if len(matches)!=1:
                    abstentions.append({'bank':bid,'symbol':symbol,'context':ctx,'reason':'missing_or_ambiguous_oto',
                                        'matched_records':len(matches),'source_wav':source,
                                        'oto_alias_body':spec['oto_alias_body']})
                    continue
                lineno,alias,timing=matches[0]
                identity=(bid,source,lineno)
                if identity in source_records:
                    raise ValueError('duplicate source acoustic row')
                source_records.add(identity)
                offset,consonant,cutoff,preutterance,overlap=timing
                anchor=offset+(preutterance+overlap)/2
                l=max(0,round((anchor+win[0])*rate/1000))
                r=min(len(pcm),round((anchor+win[1])*rate/1000))
                if l>=r or r-l<round(rate*.045):
                    abstentions.append({'bank':bid,'symbol':symbol,'context':ctx,'reason':'window_out_of_bounds'})
                    continue
                feature=acoustic_vector(pcm[l:r],rate)
                if feature is None:
                    abstentions.append({'bank':bid,'symbol':symbol,'context':ctx,'reason':'insufficient_voicing'})
                    continue
                vector,meta=feature
                records.append({'style':style,'pitch':pitch,'bank':bid,'symbol':symbol,'context':ctx,
                                'source_zip':archive.name,'source_zip_sha256':archive_sha,
                                'source_wav':source,'source_pcm_sha256':pcmsha,
                                'original_alias':alias,'oto_alias_body':spec['oto_alias_body'],
                                'oto_line':lineno,
                                'candidate_start_ms':round(l*1000/rate,3),
                                'candidate_end_ms':round(r*1000/rate,3),
                                'candidate_status':'unverified_oto_relative_acoustic_window',
                                **meta,'features':[float(v) for v in vector]})
    return records,abstentions


def _classify(train:list[dict],test:list[dict],labels:list[str]) -> list[dict]:
    if any(not any(t['symbol']==symbol for t in train) for symbol in labels):return []
    medians={symbol:np.median(np.asarray([r['features'] for r in train if r['symbol']==symbol]),axis=0)
             for symbol in labels}
    out=[]
    for r in test:
        feature=np.asarray(r['features'])
        d={symbol:float(np.linalg.norm(feature-medians[symbol])) for symbol in labels}
        nearest=min(labels,key=lambda symbol:d[symbol])
        out.append({'bank':r['bank'],'style':r['style'],'pitch':r['pitch'],
                    'context':r['context'],'actual_user_symbol':r['symbol'],
                    'nearest_user_symbol':nearest,'matched':nearest==r['symbol'],
                    'nearest_distance':round(d[nearest],5)})
    return out


def unsupervised_cluster_audit(records: list[dict], labels: list[str]) -> dict:
    """Unsupervised discovery, with user labels used *only after* fitting.

    A blind, per-subbank common-mode center reduces style/pitch spectral
    offsets without knowing individual phone categories. k is tested from
    2..6, not assumed from number of user labels; an uncertain k abstains.
    """
    if len(records)<12:return {'status':'not_enough_samples'}
    X=np.asarray([r['features'] for r in records],dtype=np.float64)
    # Raw clustering control: how much do style/pitch offsets dominate?
    raw=X.copy()
    raw_agreement=None
    if len(labels)>1:
        raw_labels=fcluster(linkage(raw, method='ward'),len(labels),criterion='maxclust')
        raw_groups=sorted(set(int(v) for v in raw_labels))
        raw_tab=np.asarray([[sum(raw_labels[i]==g and records[i]['symbol']==sym
                                 for i in range(len(records))) for sym in labels]
                            for g in raw_groups],dtype=int)
        ri,ci=linear_sum_assignment(-raw_tab)
        raw_agreement=round(int(raw_tab[ri,ci].sum())/len(records),4)
    by_bank=defaultdict(list)
    for idx,row in enumerate(records):by_bank[row['bank']].append(idx)
    for ids in by_bank.values():
        X[ids]-=X[ids].mean(axis=0)
    h=linkage(X,method='ward')
    distances=squareform(pdist(X,'euclidean'))
    scored=[]
    for k in range(2,min(6,len(records)-1)+1):
        c=fcluster(h,k,criterion='maxclust')
        if len(set(c))!=k:continue
        scores=[]
        for i,group in enumerate(c):
            members=np.where(c==group)[0]
            if len(members)<2:
                scores.append(0.)
                continue
            same=members[members!=i]
            a=float(distances[i,same].mean())
            b=min(float(distances[i,c==other].mean()) for other in set(c) if other!=group)
            scores.append((b-a)/max(a,b,1e-12))
        scored.append({'k':k,'silhouette':round(float(np.mean(scores)),5)})
    if len(labels) > 1:
        c=fcluster(h,len(labels),criterion='maxclust')
        classes=sorted(set(int(x) for x in c))
        contingency=np.asarray([[sum(c[i]==cl and records[i]['symbol']==label
                                   for i in range(len(records))) for label in labels]
                                for cl in classes],dtype=int)
        rows,cols=linear_sum_assignment(-contingency)
        agree=int(contingency[rows,cols].sum())
        # Only for a *post-hoc test*, never used to form clusters.
        align=[{'cluster':int(classes[i]),'closest_user_symbol':labels[j],
                'count':int(contingency[i,j])} for i,j in zip(rows,cols)]
        agreement=round(agree/len(records),4)
    else:
        align=[];agree=None;agreement=None
    return {'status':'exploratory_acoustic_clusters_only',
            'normalization':'subtract_per_subbank_unlabeled_mean',
            'algorithm':'ward_hierarchical_euclidean',
            'silhouette_by_k':scored,
            'best_k_by_silhouette':max(scored,key=lambda item:item['silhouette'])['k'] if scored else None,
            'k_equal_to_number_of_author_labels':len(labels),
            'posthoc_accuracy_given_author_label_count':agreement,
            'unadjusted_posthoc_accuracy_for_same_k':raw_agreement,
            'posthoc_count_correct':agree,
            'cluster_label_mapping_for_diagnostics_only':align,
            'k_selection_uncertainty':True,
            'limitation':'Choosing number of clusters from input labels is not independent discovery; raw pitch/style shifts dominate without normalization.'}


def evaluate(records:list[dict],config:dict,abstentions:list[dict]|None=None)->dict:
    """Predeclared two controls: donor context transfer and whole-pitch holdout.

    Neither is an independent human-repeat test for model-rendered subbanks.
    No phone splits or merges get approved by this function.
    """
    labels=config['user_realizations'];contexts=config['contexts']
    if len(set((r['bank'],r['source_wav'],r['oto_line']) for r in records))!=len(records):
        raise ValueError('duplicate acoustic item')
    by_bank=defaultdict(list)
    for r in records:by_bank[r['bank']].append(r)
    directional=[]
    if len(labels)>1:
        for bank,group in sorted(by_bank.items()):
            for a in contexts:
                for b in contexts:
                    if a==b:continue
                    directional+=_classify([x for x in group if x['context']==a],
                                           [x for x in group if x['context']==b],labels)
    # Train on one *voice style*, test on the other (same voicebank).
    style_holdout=[]
    pitch_holdout=[]
    if len(labels)>1:
        for style in sorted(set(r['style'] for r in records)):
            train=[x for x in records if x['style']!=style]
            test=[x for x in records if x['style']==style]
            style_holdout+=_classify(train,test,labels)
        # Hold out *all* samples from one pitch; no train-test shared WAV.
        for bank,group in sorted(by_bank.items()):
            style=group[0]['style']
            train=[x for x in records if x['style']==style and x['bank']!=bank]
            pitch_holdout+=_classify(train,group,labels)
    def score(a):return {'correct':sum(x['matched'] for x in a),'total':len(a),
                         'accuracy':round(sum(x['matched'] for x in a)/len(a),4) if a else None}
    pairwise=[]
    for bank,group in sorted(by_bank.items()):
        for ctx in contexts:
            g={r['symbol']:np.asarray(r['features']) for r in group if r['context']==ctx}
            for i,a in enumerate(labels):
                for b in labels[i+1:]:
                    if a in g and b in g:
                        pairwise.append({'bank':bank,'context':ctx,'a':a,'b':b,
                                         'feature_distance':round(float(np.linalg.norm(g[a]-g[b])),5)})
    samples_per_label=Counter(r['symbol'] for r in records)
    state='insufficient_independent_sources_for_auto_alias_change'
    return {'schema':OUT_SCHEMA,'language_id':config['language_id'],
            'original_user_base_symbol':config['user_base_symbol'],
            'user_realization_labels':labels,
            'pitch_context_origin':'author_confirmed_diffSinger_renders_not_independent_human_singers',
            'candidate_windows':len(records),'abstentions':abstentions or [],
            'per_label_count':dict(samples_per_label),
            'cross_context_top1':score(directional),
            'leave_pitch_out_top1':score(pitch_holdout),
            'leave_style_out_top1':score(style_holdout),
            'unsupervised_cluster_audit':unsupervised_cluster_audit(records,labels),
            'chance_top1':round(1/len(labels),4) if len(labels)>1 else None,
            'cross_context_predictions':directional,
            'leave_pitch_out_predictions':pitch_holdout,
            'leave_style_out_predictions':style_holdout,
            'within_bank_context_distances':pairwise,
            'unit_decision':state,
            'single_label_exploratory_mode':len(labels)==1,
            'recommended_new_user_aliases':[],
            'proposed_user_labels':{},
            'source_audio_modified':False,
            'limitations':[
                'These features do not estimate phonetic onset/offset. OTO anchor only supplies an unverified window.',
                'DiffSinger pitch/style renders are not independent human takes; groups share model and templates.',
                'Labels are user-defined and known before testing, so results assess separability, not unsupervised discovery.',
                'No split/share or user alias mutation without separate acoustic/held-out synthesis evidence.',
            ]}


def main(argv=None)->int:
    ap=argparse.ArgumentParser(description='IPA-free user-native acoustic unit contrast, exploratory')
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args(argv)
    cfg=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    out=Path(args.output)
    if out.exists():raise ValueError('refusing overwrite of existing output')
    samples,abstain=extract(cfg)
    result=evaluate(samples,cfg,abstain)
    out.mkdir(parents=True)
    (out/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'source_samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Compact, spreadsheet-friendly source audit without embedded feature arrays.
    cols=['style','pitch','bank','symbol','context','source_zip','source_zip_sha256',
          'source_wav','source_pcm_sha256','original_alias','oto_alias_body','oto_line',
          'candidate_start_ms','candidate_end_ms','candidate_status','rms_dbfs']
    with (out/'sample_index.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader()
        writer.writerows({col:row[col] for col in cols} for row in samples)
    (out/'review_candidates.json').write_text(json.dumps({
        'schema':'phonoweave.user_native_unit_review.v1',
        'language_id':cfg['language_id'],'base_user_symbol':cfg['user_base_symbol'],
        'existing_user_labels':cfg['user_realizations'],
        'review_status':'research_candidate_only',
        'approved_new_labels':[],
        'exportable_pronunciation_mapping':False,
        'evidence_file':'evidence.json',
        'manual_review_question':'Are these stable realizations valuable enough to keep separately in synthesis?',
        'notes':'Labels are opaque; no IPA names, no OTO edits, no automatic split/merge.'
    },ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'candidates':len(samples),'abstentions':len(abstain),
                      'cross_context':result['cross_context_top1'],
                      'pitch_holdout':result['leave_pitch_out_top1'],
                      'unit_decision':result['unit_decision']},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
