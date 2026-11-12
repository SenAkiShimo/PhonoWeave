"""IPA-free, OTO-free acoustic-event proposals from whole original WAVs.

These are *acoustic change points* and activity on/offsets, NOT inferred
phoneme identities or verified phoneme boundaries. The proposal path never
reads OTO or human annotations. Separate post-hoc review may read either.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import wave
import zipfile

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter1d
from scipy.signal import find_peaks, peak_widths, resample_poly, stft

SCHEMA = "phonoweave.oto_free_acoustic_events.v0.1"
HOP_MS = 10.0
MIN_SAMPLE_RATE = 16000


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_pcm(raw: bytes) -> tuple[np.ndarray, int, str]:
    with wave.open(io.BytesIO(raw), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getcomptype() != "NONE":
            raise ValueError("only uncompressed mono PCM16 source audio is allowed")
        sr = w.getframerate()
        if sr < MIN_SAMPLE_RATE or sr > 96000:
            raise ValueError("unsupported sample rate")
        data = w.readframes(w.getnframes())
    signal = np.frombuffer(data, dtype="<i2").astype(np.float32)/32768.0
    return signal, sr, sha(data)


def melbank(sr: int, nfft: int = 512, nbands: int = 24):
    to_mel = lambda hz: 2595*np.log10(1+hz/700)
    from_mel = lambda m: 700*(10**(m/2595)-1)
    edges = from_mel(np.linspace(to_mel(80),to_mel(min(7600,sr/2-100)),nbands+2))
    hz = np.fft.rfftfreq(nfft, 1/sr)
    filters = np.zeros((nbands,len(hz)),dtype=np.float32)
    for i in range(nbands):
        a,b,c=edges[i:i+3]
        filters[i]=np.maximum(0,np.minimum((hz-a)/max(1,b-a),(c-hz)/max(1,c-b)))
        filters[i]/=max(1e-9,float(filters[i].sum()))
    return filters


def _median_filter_activity(rms: np.ndarray) -> np.ndarray:
    # Equal fixed threshold for the two flanks, independent of labels and OTO.
    active=gaussian_filter1d(rms,1.0)>0.0020
    # Suppress bursts shorter than 30ms; fill holes shorter than 30ms.
    return binary_closing(binary_opening(active,structure=np.ones(3)),structure=np.ones(3))


def find_events(signal: np.ndarray, sample_rate: int) -> dict:
    """Return conservative label-free events for an entire continuous signal.

    Output time is in ORIGINAL source WAV coordinates; no source OTO is read.
    An event is a *proposal*, and its half-prominence width is NOT a credible
    interval or a gold phoneme boundary.
    """
    if signal.ndim!=1 or len(signal)<round(sample_rate*.12):
        return {"events":[],"frames":0,"active_seconds":0.,"status":"too_short"}
    if not np.all(np.isfinite(signal)):
        raise ValueError("nonfinite source audio")
    if sample_rate != 16000:
        from math import gcd
        g=gcd(sample_rate,16000)
        signal=resample_poly(signal,16000//g,sample_rate//g).astype(np.float32)
    sr=16000
    # 25ms FFT hop 10ms, no padding: correct absolute times at frame centers.
    _, frames_t, Z=stft(signal,fs=sr,window="hann",nperseg=400,noverlap=240,
                        nfft=512,boundary=None,padded=False,scaling="spectrum")
    if len(frames_t)<14:
        return {"events":[],"frames":len(frames_t),"active_seconds":0.,"status":"too_short"}
    power=np.maximum(np.abs(Z).astype(np.float32)**2,1.e-12)
    bandpower=(melbank(sr)@power).T
    logmel=np.log(np.maximum(bandpower,1e-10))
    # Per-frame spectral shape: insensitive to global gain shifts.
    shape=logmel-logmel.mean(axis=1,keepdims=True)
    shape=gaussian_filter1d(shape,1.1,axis=0)
    # STFT power is spread across many bins for fricatives but concentrated
    # in a few bins for vowels. MEAN-bin power falsely calls unvoiced
    # frication "silence". Sum spectral power (Parseval-consistent).
    rms=np.sqrt(np.maximum(2.0*np.sum(power,axis=0),0.0)).astype(np.float32)
    activity=_median_filter_activity(rms)
    # 40-ms flank means with a 10-ms exclusion at the tested instant.
    # This compares actual acoustic frames with each other, NOT OTO anchors.
    half=4
    previous=np.zeros_like(shape)
    following=np.zeros_like(shape)
    for lag in range(1,half+1):
        previous[half+1:-half-1]+=shape[half+1-lag:-half-1-lag]
        following[half+1:-half-1]+=shape[half+1+lag:-half-1+lag]
    previous/=half;following/=half
    change=np.sqrt(np.mean((following-previous)**2,axis=1))
    change[:half+2]=0;change[-half-2:]=0
    change=gaussian_filter1d(change,1.0)
    # An internal event requires sustained sound on BOTH sides (70 ms) to
    # exclude naive silence-only segmentation.
    flank=np.convolve(activity.astype(np.int32),np.ones(15,dtype=np.int32),mode='same')==15
    flank[:8]=False;flank[-8:]=False
    eligible=flank & activity
    eligible_scores=change[eligible]
    internal=[]
    if len(eligible_scores)>=12:
        median=float(np.median(eligible_scores))
        mad=float(np.median(np.abs(eligible_scores-median)))
        threshold=max(0.25,median+1.8*max(mad,0.02),float(np.quantile(eligible_scores,.77)))
        peaks,properties=find_peaks(change,distance=10,prominence=.18,height=threshold)
        if len(peaks):
            widths=peak_widths(change,peaks,rel_height=.5)
            for ii,p in enumerate(peaks):
                if not eligible[p]:continue
                # the measured peak width is *acoustic resolution only*;
                # clamp for visual display, never label as confidence interval.
                halfspan=max(10.,min(75.,float(widths[0][ii])*HOP_MS/2))
                t_ms=round(float(frames_t[p])*1000,2)
                internal.append({"time_ms":t_ms,"kind":"internal_acoustic_change",
                    "peak_strength":round(float(change[p]),5),
                    "half_prominence_width_ms":round(2*halfspan,2),
                    "display_start_ms":round(max(0,t_ms-halfspan),2),
                    "display_end_ms":round(min(len(signal)/sr*1000,t_ms+halfspan),2),
                    "status":"unverified_acoustic_event"})
    # Quiet-to-active and active-to-quiet transitions are separately annotated,
    # never disguised as phone transitions.
    transitions=np.flatnonzero(np.diff(activity.astype(np.int8))!=0)+1
    activity_events=[]
    for p in transitions:
        if p<3 or p>=len(activity)-3:continue
        if activity[p] and np.mean(activity[p:p+7])<.7:continue
        if not activity[p] and np.mean(activity[p-7:p])<.7:continue
        activity_events.append({"time_ms":round(float(frames_t[p])*1000,2),
             "kind":"activity_onset" if activity[p] else "activity_offset",
             "status":"signal_activity_only"})
    events=sorted(internal+activity_events,key=lambda x:(x['time_ms'],x['kind']))
    return {"events":events,"frames":len(frames_t),
        "active_seconds":round(float(activity.sum()*.010),3),
        "status":"measured","internal_count":len(internal),
        "activity_edges":len(activity_events)}


def propose_intervals(events:list[dict], duration_ms:float)->list[dict]:
    """Partition active regions by unsupervised internal change points.

    A partition is an unknown acoustic region, never assigned a phoneme
    identity. Activity transitions are separate from internal change points.
    """
    if duration_ms<=0:raise ValueError('duration must be positive')
    ordered=sorted(events,key=lambda e:(float(e['time_ms']),e['kind']))
    if any(float(e['time_ms'])<0 or float(e['time_ms'])>duration_ms for e in ordered):
        raise ValueError('out of range acoustic event')
    intervals=[];started=None;groups=[]
    for e in ordered:
        t=float(e['time_ms'])
        if e['kind']=='activity_onset':
            if started is None:started=t
        elif e['kind']=='activity_offset':
            if started is None:started=0.0  # active at recording start
            if t>started:groups.append((started,t))
            started=None
    if started is not None and duration_ms>started:
        groups.append((started,duration_ms))
    for start,end in groups:
        changes=sorted(float(e['time_ms']) for e in ordered
            if e['kind']=='internal_acoustic_change' and start+45<=float(e['time_ms'])<=end-45)
        boundaries=[start]
        for p in changes:
            if p-boundaries[-1]>=60 and end-p>=45:boundaries.append(p)
        boundaries.append(end)
        for a,b in zip(boundaries,boundaries[1:]):
            if b-a<25:continue
            intervals.append({'start_ms':round(a,2),'end_ms':round(b,2),
                'kind':'unclassified_acoustic_region',
                'status':'candidate_partition_NOT_verified_phone'})
    return intervals


def textgrid(events:list[dict],duration_s:float,segments:list[dict]|None=None)->str:
    """Praat text point tier: no phone-label or boundary assertion."""
    def quote(x):return '"'+str(x).replace('"','""')+'"'
    out=['File type = "ooTextFile"','Object class = "TextGrid"','',
         'xmin = 0',f'xmax = {duration_s:.6f}','tiers? <exists>','size = 2','item []:',
         '    item [1]:','        class = "TextTier"',
         '        name = "acoustic_proposals_NOT_phonemes"',
         '        xmin = 0',f'        xmax = {duration_s:.6f}',
         f'        points: size = {len(events)}']
    for j,e in enumerate(events,1):
        out += [f'        points [{j}]:',f'            number = {e["time_ms"]/1000:.6f}',
                f'            mark = {quote(e["kind"])}']
    segments=segments if segments is not None else propose_intervals(events,duration_s*1000)
    spans=[];cursor=0.
    for row in sorted(segments,key=lambda v:v['start_ms']):
        a=row['start_ms']/1000;b=row['end_ms']/1000
        if a<cursor-0.00001:raise ValueError('overlapping proposed regions')
        if a>cursor+0.000001:spans.append((cursor,a,''))
        spans.append((a,b,'unclassified_acoustic_region'))
        cursor=b
    if cursor<duration_s:spans.append((cursor,duration_s,''))
    out += ['    item [2]:','        class = "IntervalTier"',
            '        name = "unverified_acoustic_partitions"',
            '        xmin = 0',f'        xmax = {duration_s:.6f}',
            f'        intervals: size = {len(spans)}']
    for j,(a,b,name) in enumerate(spans,1):
        out += [f'        intervals [{j}]:',f'            xmin = {a:.6f}',
                f'            xmax = {b:.6f}',f'            text = {quote(name)}']
    return '\n'.join(out)+'\n'


def _safe_member(root:str,name:str)->str:
    if not root.endswith('/') or root.startswith('/') or '..' in root.split('/'):
        raise ValueError('invalid ZIP root')
    rel=name[len(root):]
    if not name.startswith(root) or '/' in rel or '\\' in rel or '..' in rel:
        raise ValueError('unsafe WAV member')
    return rel


def scan_archive(archive:Path,root:str,output:Path)->dict:
    """Read original continuous WAV, no OTO reads, no target labels."""
    if output.exists():raise FileExistsError('refuse to overwrite an existing output directory')
    archive=Path(archive).resolve(strict=True)
    output.mkdir(parents=True)
    all_rows=[];sources=[];all_segments=[]
    with zipfile.ZipFile(archive) as z:
        wavs=sorted(n for n in z.namelist() if n.startswith(root) and n.lower().endswith('.wav') and '/' not in n[len(root):])
        if not wavs:raise ValueError('no WAVs in ZIP root')
        (output/'textgrids').mkdir()
        for wav in wavs:
            basename=_safe_member(root,wav)
            pcm,sr,pcm_sha=read_pcm(z.read(wav))
            result=find_events(pcm,sr)
            length_ms=round(len(pcm)*1000/sr,3)
            events=result['events']
            segments=propose_intervals(events,length_ms)
            for i,seg in enumerate(segments,1):
                all_segments.append({'wav':basename,'segment_num':i,
                    'source_pcm_sha256':pcm_sha,'sample_rate':sr,**seg})
            for i,e in enumerate(events,1):
                all_rows.append({'wav':basename,'event_num':i,'source_pcm_sha256':pcm_sha,
                                 'sample_rate':sr,'duration_ms':length_ms,**e})
            (output/'textgrids'/f'{Path(basename).stem}.TextGrid').write_text(
                textgrid(events,length_ms/1000,segments),encoding='utf8')
            sources.append({'source_wav':basename,'source_pcm_sha256':pcm_sha,
                'duration_ms':length_ms,'total_events':len(events),
                'internal_events':result.get('internal_count',0),
                'activity_edges':result.get('activity_edges',0),
                'candidate_segments':len(segments),
                'activity_seconds':result['active_seconds'],'status':result['status']})
    with (output/'events.csv').open('w',newline='',encoding='utf8') as f:
        fields=['wav','event_num','source_pcm_sha256','sample_rate','duration_ms','time_ms','kind','peak_strength','half_prominence_width_ms','display_start_ms','display_end_ms','status']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(all_rows)
    with (output/'candidate_segments.csv').open('w',newline='',encoding='utf8') as f:
        cols=['wav','segment_num','source_pcm_sha256','sample_rate',
              'start_ms','end_ms','kind','status']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(all_segments)
    with (output/'sources.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(sources[0]));w.writeheader();w.writerows(sources)
    payload={'schema':SCHEMA,'archive_name':archive.name,'archive_sha256':_sha_file(archive),
       'source_wavs':len(sources),'internal_acoustic_events':sum(x['internal_events'] for x in sources),
       'activity_edges':sum(x['activity_edges'] for x in sources),
       'candidate_segments':len(all_segments),
       'duration_s':round(sum(x['duration_ms'] for x in sources)/1000,3),
       'oto_files_read':0,'language_specific_phone_models':0,
       'verified_phoneme_boundaries':0,'status':'acoustic_event_proposals_ONLY'}
    (output/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf8')
    return payload


def _sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    result=scan_archive(a.archive,a.root,a.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
