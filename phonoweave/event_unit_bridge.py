"""IPA-independent candidate linking: acoustic-only event proposals to source-trained user labels.

Training references are frozen v3 OTO-derived samples. Inference takes ONLY v4
acoustic event time and source audio; OTO timestamps are not used to generate
inference proposals or features. Posthoc evaluation may read the frozen v2 index.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math,wave,zipfile
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.fft import dct,rfft
from scipy.signal import resample_poly

MODE='late_context'
SLICES=(-200.,-120.),(-130.,-50.),(-80.,0.)
# Frozen split from v3, independent of event matches
TEST_GROUPS={'a.wav','ai.wav','ang.wav','e0.wav','en0.wav','eriin.wav','uvvn.wav','vcsection2.wav'}
MIN_RMS=.002

def readcsv(data):return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))

def load_data(v3,v4,manifest):
    with zipfile.ZipFile(v3) as z:
        rows=readcsv(z.read('v2_frozen_inputs/source_index.csv'))
        arrays=np.load(io.BytesIO(z.read('results/temporal_features.npz')))
        X=arrays['feat_'+MODE].astype(np.float32)
        report=json.loads(z.read('results/evaluation.json'))
    assert len(rows)==len(X) and set(report['frozen_test_groups'])==TEST_GROUPS and report['selected_mode']==MODE
    with zipfile.ZipFile(manifest) as z:
        config=json.loads(z.read('runs/lynneqing_bulk_manifest_local_example.json'))
    with zipfile.ZipFile(v4) as z:
        events={b:readcsv(z.read(f'results/{b}/events.csv')) for b in [p['style']+'_'+p['pitch'] for p in config['banks']]}
        sources={b:readcsv(z.read(f'results/{b}/sources.csv')) for b in events}
        segs={b:readcsv(z.read(f'results/{b}/candidate_segments.csv')) for b in events}
    return rows,X,config,events,sources,segs

def pcm_bytes(raw):
    with wave.open(io.BytesIO(raw),'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getcomptype()!='NONE':raise ValueError('needs mono PCM16')
        rate=w.getframerate(); frames=w.readframes(w.getnframes())
    if rate not in (44100,48000,96000):raise ValueError('unexpected sample rate')
    return np.frombuffer(frames,dtype='<i2').astype(np.float32)/32768,rate,hashlib.sha256(frames).hexdigest()

def extract_at_ends(signal,rate,times,mel):
    """Computes exact frozen 36-dimensional late-context feature at AUDIO-ONLY times.

    No OTO or phones passed; invalid / silent windows are abstained.
    """
    n=len(times);result=np.zeros((n,36),dtype=np.float32);valid=np.ones(n,dtype=bool)
    if not n:return result,valid
    g=math.gcd(rate,16000);audio=resample_poly(signal,16000//g,rate//g).astype(np.float32)
    pre=np.empty_like(audio);pre[0]=audio[0];pre[1:]=audio[1:]-.97*audio[:-1]
    hann=np.hanning(400).astype(np.float32)
    mel=mel.astype(np.float32,copy=False)
    # Batch each feature slice, not 45k separate FFT calls
    for j,(lo,hi) in enumerate(SLICES):
        l=np.rint((times+lo)*16).astype(int);r=np.rint((times+hi)*16).astype(int)
        ok=(l>=0)&(r<=len(audio))&(r-l>=1040)
        valid&=ok
        ix=np.flatnonzero(ok)
        if not len(ix):continue
        good=[];frames=[]
        for index in ix:
            seg=audio[l[index]:r[index]]
            if np.sqrt(np.mean(seg*seg))<MIN_RMS:valid[index]=False;continue
            x=np.lib.stride_tricks.sliding_window_view(pre[l[index]:r[index]],400)[::160]
            if len(x)<2:valid[index]=False;continue
            good.append(index);frames.append(x.copy())
        if not frames:continue
        from scipy.fft import rfft,dct
        # chunks to limit memory
        for offset in range(0,len(good),256):
            subset=frames[offset:offset+256];count=[len(x) for x in subset]
            flat=np.concatenate(subset,axis=0)
            flat=(flat-flat.mean(axis=1,keepdims=True))*hann
            pow=np.abs(rfft(flat,n=1024,axis=1))**2
            cep=dct(np.log(np.maximum(pow.astype(np.float32)@mel.T,1e-9)),type=2,norm='ortho',axis=1)[:,1:13]
            p=0
            for i,c in enumerate(count):
                sub=cep[p:p+c];p+=c
                center=np.median(sub,axis=0)
                center=center/max(1.,float(np.linalg.norm(center)))
                result[good[offset+i],j*12:j*12+12]=center
    result/=np.sqrt(3)
    return result,valid

def model(rows,X):
    labels=np.array([r['symbol'] for r in rows]);banks=np.array([r['bank'] for r in rows]);groups=np.array([r['source_wav'] for r in rows])
    tr=~np.isin(groups,list(TEST_GROUPS));bankmeans={}
    pooled=X[tr].mean(axis=0)
    for b in sorted(set(banks)):
        idx=tr&(banks==b)
        bankmeans[b]=X[idx].mean(axis=0) if np.any(idx) else pooled
    y=sorted(set(labels[tr]));Y=np.stack([(X[tr&(labels==c)]-np.stack([bankmeans[b] for b in banks[tr&(labels==c)]])).mean(axis=0) for c in y])
    return y,Y,bankmeans,pooled

def label_features(X,bank,y,Y,bankmeans,pooled):
    C=X-bankmeans.get(bank,pooled)
    ds=np.sum((C[:,None,:]-Y[None,:,:])**2,axis=2)
    ii=np.argsort(ds,axis=1)[:,:3]
    scores=np.take_along_axis(ds,ii,axis=1)
    margin=(scores[:,1]-scores[:,0])/np.maximum(scores[:,1],1e-7)
    return [[y[k] for k in row] for row in ii],scores,margin

def run(v3,v4,manifest,out):
    out=Path(out)
    if out.exists():raise FileExistsError('output exists')
    out.mkdir(parents=True)
    rows,X,config,events,sources,segs=load_data(v3,v4,manifest)
    from .orthography_native_units import MEL
    mel=MEL
    labels,Y,bankmeans,pooled=model(rows,X)
    # Freeze the source identity from training archive independently of events.
    expected_pcm={(r['bank'],r['source_wav']):r['source_pcm_sha256'] for r in rows}
    expected_archives={r['bank']:r['source_zip_sha256'] for r in rows}
    predictions=[];segment_predictions=[]
    for bank in sorted(events):
        spec=next(x for x in config['banks'] if x['style']+'_'+x['pitch']==bank)
        ar=Path(spec['archive']);root=spec['root']
        if hashlib.sha256(ar.read_bytes()).hexdigest()!=expected_archives[bank]:
            raise ValueError('archive hash different from frozen v3 training source: '+bank)
        index={r['source_wav']:r for r in sources[bank]}
        by_ev=defaultdict(list);by_seg=defaultdict(list)
        for e in events[bank]:
            if e['kind']=='internal_acoustic_change':by_ev[e['wav']].append(e)
        for q in segs[bank]:by_seg[q['wav']].append(q)
        with zipfile.ZipFile(ar) as z:
            for wav in sorted(index):
                rec=index[wav];signal,rate,digest=pcm_bytes(z.read(root+wav))
                if digest!=rec['source_pcm_sha256']:raise ValueError('PCM hash changed '+bank+'/'+wav)
                if (bank,wav) in expected_pcm and digest!=expected_pcm[(bank,wav)]:
                    raise ValueError('v3 labeled source PCM hash disagrees with v4 raw source')
                ev=sorted(by_ev[wav],key=lambda e:float(e['time_ms']))
                # no label timestamps or OTO rows used during inference
                times=np.asarray([float(e['time_ms']) for e in ev])
                feats,valid=extract_at_ends(signal,rate,times,mel)
                pred,score,margin=label_features(feats,bank,labels,Y,bankmeans,pooled)
                event_lookup={}
                for i,e in enumerate(ev):
                    ms=float(e['time_ms']); q={
                        'bank':bank,'source_wav':wav,'source_pcm_sha256':digest,'time_ms':round(ms,3),
                        'event_num':int(e['event_num']),'kind':'internal_acoustic_change','spectral_strength':e['peak_strength'],
                        'candidate_unit':pred[i][0] if valid[i] else '',
                        'alternatives':'|'.join(pred[i][:3]) if valid[i] else '',
                        'margin':round(float(margin[i]),6) if valid[i] else '',
                        'status':'candidate_UNVERIFIED' if valid[i] else 'abstain_insufficient_acoustic_context',
                        'source_split':'heldout_source_template' if wav in TEST_GROUPS else 'development_source_template'}
                    predictions.append(q)
                    event_lookup[round(ms,3)]=q
                for seg in by_seg[wav]:
                    a,b=float(seg['start_ms']),float(seg['end_ms'])
                    dur=b-a
                    item=event_lookup.get(round(b,3))
                    allowed=50 <= dur <= 500
                    if allowed and item is not None and item['status']=='candidate_UNVERIFIED':
                        guess=item['candidate_unit'];alternatives=item['alternatives'];status='candidate_UNVERIFIED_not_a_phone'
                    else:
                        guess=alternatives=''
                        status=('abstain_long_or_short_region' if not allowed
                                else 'abstain_no_valid_end_event')
                    segment_predictions.append({'bank':bank,'source_wav':wav,'source_pcm_sha256':digest,'segment_num':int(seg['segment_num']),'start_ms':round(a,3),'end_ms':round(b,3),'duration_ms':round(dur,3),'candidate_unit':guess,'alternatives':alternatives,'status':status,'interpretation':'segment_candidate_not_an_established_phone'})
    # v2 source-index evaluated only after inference completed
    evals=[]
    matched=Counter();total=Counter()
    bypred=defaultdict(list)
    for r in predictions:bypred[(r['bank'],r['source_wav'])].append(r)
    for r in rows:
        if r['source_wav'] not in TEST_GROUPS:continue
        bank,wav=r['bank'],r['source_wav']
        anchor=float(r['start_ms'])+125
        # 100ms fixed posthoc tolerance around OTO midpoint, not a gold phone boundary
        nearby=[p for p in bypred[bank,wav] if abs(p['time_ms']-anchor)<=100 and p['status']=='candidate_UNVERIFIED']
        closest=min(nearby,key=lambda p:abs(p['time_ms']-anchor)) if nearby else None
        # summary at per-OTO level: posthoc proximity does NOT prove correct phone segmentation
        best=[p for p in nearby if p['candidate_unit']==r['symbol']]
        evals.append({'bank':bank,'source_wav':wav,'oto_line':r['oto_line'],'reference_symbol':r['symbol'],'reference_context':r['context'],
            'oto_reference_anchor_ms':round(anchor,3),'any_event_within_100ms':bool(nearby),'event_count_100ms':len(nearby),
            'closest_event_ms':closest['time_ms'] if closest else '',
            'closest_prediction':closest['candidate_unit'] if closest else '',
            'closest_correct':bool(closest and closest['candidate_unit']==r['symbol']),
            'any_correct_prediction_nearby':bool(best),
            'analysis_only_not_true_phone_boundaries':True})
    fields=list(predictions[0])
    with (out/'event_unit_candidates.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(predictions)
    with (out/'candidate_segment_unit_hints.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(segment_predictions[0]));w.writeheader();w.writerows(segment_predictions)
    with (out/'heldout_posthoc_oto_diagnostics.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(evals[0]));w.writeheader();w.writerows(evals)
    test=[r for r in evals];
    report={'schema':'phonoweave.event_to_native_unit_bridge.v0.5','note':'Unanchored v4 event proposals + supervised v3 user-label prototypes, not verified phone segmentation',
            'training_samples':sum(r['source_wav'] not in TEST_GROUPS for r in rows),
            'frozen_heldout_templates':sorted(TEST_GROUPS),'heldout_oto_probe_count':len(test),
            'total_unanchored_events':len(predictions),'feature_valid_events':sum(r['status']=='candidate_UNVERIFIED' for r in predictions),
            'abstained_events':sum(r['status']!='candidate_UNVERIFIED' for r in predictions),
            'total_unanchored_segments':len(segment_predictions),
            'segments_with_candidate_hint':sum(r['status']=='candidate_UNVERIFIED_not_a_phone' for r in segment_predictions),
            'heldout_any_event_near_oto_fraction':round(sum(r['any_event_within_100ms'] for r in test)/len(test),5),
            'heldout_closest_label_correct_fraction_all':round(sum(r['closest_correct'] for r in test)/len(test),5),
            'heldout_any_correct_nearby_fraction_all':round(sum(r['any_correct_prediction_nearby'] for r in test)/len(test),5),
            'heldout_candidate_precision_proxy':None,
            'caution':'OTO anchor locations only used for this posthoc diagnostic. A correct label near an OTO anchor is not a verified phone boundary or a novel unit discovery.',
            'ipa_used':False,'source_aliases_modified':False,'auto_split_merge':False}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--v3',required=True);p.add_argument('--v4',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); print(json.dumps(run(a.v3,a.v4,a.manifest,a.output),indent=2))

def main(argv=None):
    p=argparse.ArgumentParser(prog='phonoweave link-user-units-to-events')
    p.add_argument('--v3',required=True);p.add_argument('--v4',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    print(json.dumps(run(a.v3,a.v4,a.manifest,a.output),ensure_ascii=False,indent=2));return 0
