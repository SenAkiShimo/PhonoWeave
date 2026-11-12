"""IPA/OTO-time-independent acoustic motifs directly from PCM WAV waveforms.

Audio-only segmentation and log-mel spectra. No dictionaries, language IDs,
user-unit labels, pitch conventions, or OTO timestamps are passed into the
feature extractor or clusterer. Input WAV sets may nevertheless have been
CURATED using OTO: provenance of archive selection must be described honestly.
A motif is NOT an identified phone; no aliases or voicebanks are edited.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import wave
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .native_acoustic_subtypes_v9 import kmeans2, assign, adjusted_rand_index

SCHEMA = 'phonoweave.audio_only_acoustic_motifs.v0.9'
MIN_RMS = 0.003
MEL_BANDS = 24


def pcm16(raw: bytes) -> tuple[np.ndarray, int, str]:
    with wave.open(io.BytesIO(raw)) as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getcomptype()!='NONE':
            raise ValueError('only mono uncompressed PCM16 supported')
        rate=w.getframerate()
        if not (8000 <= rate <= 96000):
            raise ValueError('unsupported sample rate')
        frames=w.readframes(w.getnframes())
    return (np.frombuffer(frames,dtype='<i2').astype(np.float32)/32768.,
            rate,hashlib.sha256(frames).hexdigest())


def mel_bank(rate: int, nfft: int, bands: int = MEL_BANDS) -> np.ndarray:
    mel=lambda hz:2595.*np.log10(1.+hz/700.)
    inv=lambda m:700.*(10.**(m/2595.)-1.)
    hz=inv(np.linspace(mel(80.),mel(min(rate/2.-1.,7600.)),bands+2))
    frequencies=np.fft.rfftfreq(nfft,1/rate)
    m=np.zeros((len(frequencies),bands),dtype=np.float64)
    for k in range(bands):
        l,c,r=hz[k:k+3]
        m[:,k]=np.maximum(0,np.minimum((frequencies-l)/max(c-l,1.e-6),
                                        (r-frequencies)/max(r-c,1.e-6)))
        m[:,k]/=max(m[:,k].sum(),1.e-9)
    return m


def audio_features(x: np.ndarray, rate:int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """25ms Hann FFT, 10ms hop, log-mel (not a phone recognition model)."""
    win=max(128,round(rate*0.025));hop=max(64,round(rate*0.010))
    if len(x)<win+hop:
        return (np.zeros((0,MEL_BANDS)),np.zeros(0),np.zeros(0))
    frames=np.lib.stride_tricks.sliding_window_view(x,win)[::hop]
    nfft=1<<(win-1).bit_length()
    power=np.abs(np.fft.rfft(frames*np.hanning(win),n=nfft,axis=1))**2
    mel=power@mel_bank(rate,nfft)
    logmel=np.log1p(mel*10000.)
    rms=np.sqrt((frames.astype(np.float64)**2).mean(axis=1))
    times=(np.arange(len(frames))*hop+win/2)/rate*1000.
    return logmel,rms,times


def propose_segments(logmel:np.ndarray,rms:np.ndarray,times:np.ndarray) -> list[tuple[int,int]]:
    """Activity + spectral-novelty boundaries without linguistic side information."""
    n=len(rms)
    if n<8 or np.max(rms)<MIN_RMS:return []
    activity=rms>max(MIN_RMS,0.18*float(np.quantile(rms,0.85)))
    valid=np.flatnonzero(activity)
    if len(valid)<8:return []
    a=max(0,int(valid[0]));b=min(n,int(valid[-1])+1)
    if b-a<8:return []
    mag=np.abs(np.diff(logmel,axis=0))
    novelty=np.r_[0.,np.mean(mag,axis=1)]
    smooth=np.convolve(novelty,[0.25,0.50,0.25],mode='same')
    segment=smooth[a+4:b-4]
    if len(segment)==0:return [(a,b)]
    med=float(np.median(segment));mad=float(np.median(np.abs(segment-med)))
    threshold=max(med+2*mad, float(np.quantile(segment,0.75)))
    peaks=[]
    for i in range(a+4,b-4):
        if smooth[i]>threshold and smooth[i]>=smooth[i-1] and smooth[i]>smooth[i+1]:
            if not peaks or i-peaks[-1]>=8:
                peaks.append(i)
            elif smooth[i]>smooth[peaks[-1]]:
                peaks[-1]=i
    cuts=[a]
    for i in peaks:
        if i-cuts[-1]>=8 and b-i>=8:cuts.append(i)
    cuts.append(b)
    return [(l,r) for l,r in zip(cuts,cuts[1:]) if r-l>=8]


def embed_segment(logmel:np.ndarray,a:int,b:int) -> np.ndarray:
    """Spectral distribution and evolution (48D), not just event timing."""
    y=logmel[a:b].astype(np.float64)
    center=y.mean(axis=0)
    # Retain spectral shape, suppress uniform loudness shifts.
    shape=center-center.mean()
    t=max(1,len(y)//3)
    evolution=y[-t:].mean(axis=0)-y[:t].mean(axis=0)
    return np.r_[shape,evolution].astype(np.float32)


def read_clips(path: str | Path, suffix: str = 'source_context_600ms.wav'):
    path=Path(path)
    if not path.is_file() or path.suffix.lower()!='.zip':
        raise ValueError('input must be an existing ZIP with PCM16 WAVs')
    with zipfile.ZipFile(path) as z:
        paths=sorted(n for n in z.namelist() if n.endswith(suffix) and not n.endswith('/'))
        if not paths:raise ValueError('no matching WAV clips in archive')
        for name in paths:
            if name.startswith('/') or '..' in Path(name).parts:raise ValueError('unsafe archive entry')
            # Archive paths are NEVER passed as clustering features.
            x,rate,digest=pcm16(z.read(name))
            yield name,x,rate,digest


def extract_from_audio_zip(source: str | Path):
    manifest=[]; embeddings=[]
    for name,x,rate,digest in read_clips(source):
        logmel,rms,times=audio_features(x,rate)
        spans=propose_segments(logmel,rms,times)
        path=Path(name)
        parts=path.parts
        bank=next((p for p in parts if p.startswith(('lu_','mang_'))), 'unidentified_bank')
        # This is an audit/splitting key only. The folder may reflect OTO-curated source choice.
        # Strip only the opaque OTO line suffix for grouping matching clips;
        # it is never parsed as timing or supplied to acoustic algorithms.
        group=parts[-2].split("_OTO",1)[0] if len(parts)>2 else name
        for ordinal,(a,b) in enumerate(spans):
            if float(np.mean(rms[a:b]))<MIN_RMS:continue
            embeddings.append(embed_segment(logmel,a,b))
            manifest.append({'wav_archive_entry':name,'bank':bank,'clip_group':group,
                             'source_pcm_sha256':digest,'region_index':ordinal,
                             'start_ms':round(float(times[a]),3),
                             'end_ms':round(float(times[b-1]),3),
                             'status':'unanchored_spectral_region_NOT_a_phone'})
    return manifest,np.stack(embeddings) if embeddings else np.empty((0,48),dtype=np.float32)


def discover_audio_motifs(manifest:list[dict],X:np.ndarray) -> tuple[list[dict],dict]:
    if len(X)!=len(manifest) or X.shape[1:]!=(48,) or not np.isfinite(X).all():
        raise ValueError('invalid waveform-derived feature table')
    groups=sorted({m['clip_group'] for m in manifest})
    if len(groups)<5 or len(X)<30:
        return [], {'status':'abstain_too_few_audio_regions_or_unique_clips',
                    'regions':len(X),'distinct_clip_groups':len(groups)}
    # Deterministic group-level split, excluding exact audio clip groups from fit.
    testgroups={g for g in groups if int(hashlib.sha256(g.encode()).hexdigest()[:6],16)%5==0}
    if len(testgroups)<2:
        testgroups=set(groups[::5])
    dev=np.array([m['clip_group'] not in testgroups for m in manifest],bool)
    if dev.sum()<20 or (~dev).sum()<10:
        return [],{'status':'abstain_insufficient_train_or_heldout_regions','regions':len(X)}
    # No bank/label/OTO metadata enter the normalization or clustering.
    mean=X[dev].mean(axis=0)
    sd=np.std(X[dev],axis=0)
    sd=np.maximum(sd,max(.04,.2*float(np.max(sd))))
    Y=np.clip((X-mean)/sd,-6.,6.)
    try:centers,_,loss=kmeans2(Y[dev],SEED_FROM_AUDIO)
    except ValueError:return [],{'status':'abstain_constant_spectral_vectors','regions':len(X)}
    assigned=assign(Y,centers)
    for r,k,held in zip(manifest,assigned,~dev):
        r['motif_id']='acoustic_motif_'+str(int(k))
        r['split']='heldout_audio_clip_group' if held else 'development_audio_clip_group'
    support={k:len({r['clip_group'] for r in manifest if r.get('motif_id')=='acoustic_motif_'+str(k)}) for k in (0,1)}
    banks={k:len({r['bank'] for r in manifest if r.get('motif_id')=='acoustic_motif_'+str(k)}) for k in (0,1)}
    motif_descriptives={}
    for k in (0,1):
        cluster_rows=[r for r in manifest if r.get('motif_id')=='acoustic_motif_'+str(k)]
        mids=np.array([r['region_index'] for r in cluster_rows])
        lengths=np.array([r['end_ms']-r['start_ms'] for r in cluster_rows])
        motif_descriptives[f'acoustic_motif_{k}']={
            'median_region_length_ms':round(float(np.median(lengths)),2),
            'mean_region_order_index_within_clip':round(float(np.mean(mids)),3),
            'originating_clip_groups':support[k],
            'banks_observed':banks[k],
        }
    styles=sorted({r['bank'].split('_')[0] for r in manifest})
    style_ari=None
    if len(styles)==2:
        trained=[]
        for ix,style in enumerate(styles):
            style_mask=np.array([r['bank'].split('_')[0]==style for r in manifest],bool) & dev
            if style_mask.sum()>=20:
                try:trained.append(kmeans2(Y[style_mask],SEED_FROM_AUDIO+ix+1)[0])
                except ValueError:pass
        if len(trained)==2:
            style_ari=round(adjusted_rand_index(assign(Y[~dev],trained[0]),assign(Y[~dev],trained[1])),4)
    report={'status':'exploratory_audio_only_motifs_NOT_phones', 'regions':len(X),
            'distinct_clip_groups':len(groups),'dev_regions':int(dev.sum()),
            'heldout_regions':int((~dev).sum()), 'motif_count':2,
            'cross_style_heldout_assignment_ari':style_ari,
            'motif_counts':dict(Counter(r['motif_id'] for r in manifest)),
            'clip_group_support':support, 'bank_support':banks,
            'motif_descriptive_statistics':motif_descriptives,
            'mel_bands':MEL_BANDS,'feature_dim':48,'language_input_used':False,
            'oto_time_input_used':False,'directory_metadata_in_features':False,
            'input_archive_was_oto_curated':bool(any('all_sh_phone_cores/' in r['wav_archive_entry'] for r in manifest)),
            'clip_groups_normalize_oto_line_suffix_for_split_only':True,
            'heldout_is_not_new_speaker':True,
            'no_phoneme_or_synthesis_unit_decisions':True,
            'heldout_is_not_independent_source_speaker':True,
            'feature_note':'24 log-mel spectral shape + 24 early/late spectral change'}
    return manifest,report

SEED_FROM_AUDIO=20261111


def run(source: str|Path, output:str|Path) -> dict:
    output=Path(output)
    if output.exists():raise FileExistsError('output exists; refusing overwrite')
    manifest,embeddings=extract_from_audio_zip(source)
    rows,report=discover_audio_motifs(manifest,embeddings)
    report['schema']=SCHEMA
    report['archive_sha256']=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    report['source_selection_provenance']='user_supplied_audio_zip; selection may be OTO-derived even though local segmentation does not read OTO'
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Store only acoustic vectors and structural metadata (never raw audio).
    np.savez_compressed(output/'raw_spectral_embeddings.npz',vectors=embeddings)
    (output/'motif_regions.json').write_text(json.dumps({'schema':SCHEMA,'regions':rows or manifest},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(prog='python -m phonoweave.native_wav_motifs_v9')
    p.add_argument('--audio-zip',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv)
    print(json.dumps(run(a.audio_zip,a.output),ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
