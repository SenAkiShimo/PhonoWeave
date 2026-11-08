"""Read-only cross-subbank VC audit on existing UTAU ZIP archives.

A style is not assumed to be an independent speaker. A pitch subbank is not
assumed to be an independently recorded take. The user has confirmed this
particular seven-subbank LynneQing set was directly generated with DiffSinger.
An optional source-provenance declaration distinguishes generated samples
from human recordings, without inferring model checkpoint or shared prompts.
Manual D4 transition intervals are transferred to other banks as *unverified
candidates* only. Nothing here is a split/share or synthesis efficacy decision.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import statistics
import wave
import zipfile

import numpy as np

SCHEMA = 'phonoweave.existing_multisubbank_vc_audit.v1'
TARGETS = (("i", "m", "vcsection2.wav"), ("ir", "m", "vcsection2.wav"),
           ("i0", "m", "vcsection2.wav"), ("i", "sh", "vcsection3.wav"),
           ("ir", "sh", "ingi0ir.wav"), ("i0", "sh", "vcsection3.wav"))
# User-drawn D4 clip intervals relative to 240 ms OTO midpoint; do not transfer
# their verified status to other audio. These are a *diagnostic comparison* only.
D4_REVIEWED_INTERVALS = {
    ('i','m'): (236.6, 290.2), ('ir','m'): (207.0,259.7),
    ('i0','m'): (211.8,254.5), ('i','sh'): (219.8,263.8),
    ('ir','sh'): (234.8,282.7), ('i0','sh'): (226.1,289.5),
}
EDGES = np.geomspace(150, 8000, 13)

@dataclass(frozen=True)
class Bank:
    style: str
    pitch: str
    archive: Path
    root: str
    prefix: str
    reference: bool = False

    @property
    def id(self) -> str:
        return f'{self.style}_{self.pitch}'


def make_bank(entry: dict) -> Bank:
    required=('style','pitch','archive','root','prefix')
    if not all(isinstance(entry.get(k),str) and entry[k] for k in required):
        raise ValueError('bank needs nonempty style,pitch,archive,root,prefix')
    root=entry['root']
    if not root.endswith('/') or '..' in Path(root).parts or root.startswith('/'):
        raise ValueError('invalid ZIP subbank directory')
    return Bank(entry['style'],entry['pitch'],Path(entry['archive']).expanduser().resolve(strict=True),root,entry['prefix'],entry.get('reference',False) is True)


def parse_oto(data: bytes, prefix: str) -> list[dict]:
    try:
        txt=data.decode('utf-8-sig')
    except UnicodeDecodeError:
        txt=data.decode('cp932')
    result=[]
    for line_no,line in enumerate(txt.splitlines(),1):
        if not line.strip(): continue
        try:
            wav,tail=line.split('=',1)
            alias,*fields=tail.rsplit(',',5)
            if len(fields)!=5: raise ValueError('5 OTO time fields required')
            values=tuple(float(v) for v in fields)
            if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite timing')
            if not alias.startswith(prefix):
                # Extra malformed rows must be surfaced, not silently counted.
                result.append(dict(line=line_no,wav=wav,alias=alias,body=None,values=values,invalid_prefix=True))
                continue
            result.append(dict(line=line_no,wav=wav,alias=alias,
                body=alias[len(prefix):],values=values,invalid_prefix=False))
        except (ValueError,IndexError) as exc:
            raise ValueError(f'invalid OTO row {line_no}: {exc}') from exc
    return result


def wav_header(z:zipfile.ZipFile, name:str)->dict:
    with z.open(name) as handle:
        first=handle.read(512)
    try:
        with wave.open(io.BytesIO(first),'rb') as w:
            rate,channels,width,frames=w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getnframes()
    except (wave.Error,EOFError) as exc:
        raise ValueError('invalid WAV header: '+name) from exc
    if channels!=1 or width!=2 or rate not in (44100,48000,96000):
        raise ValueError('unsupported WAV format: '+name)
    return {'rate':rate,'duration_ms':round(frames*1000/rate,4),'frames':frames}


def wav_data(z:zipfile.ZipFile, name:str)->tuple[np.ndarray,int,str]:
    raw=z.read(name)
    with wave.open(io.BytesIO(raw),'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getcomptype()!='NONE':
            raise ValueError('only mono PCM16 accepted for pilot')
        rate=w.getframerate()
        data=w.readframes(w.getnframes())
    return np.frombuffer(data,dtype='<i2').astype(np.float64)/32768.0,rate,sha256(data).hexdigest()


def sample_metrics(x:np.ndarray,rate:int,center_ms:float)->dict|None:
    n=round(.016*rate)
    start=round(center_ms/1000*rate)-n//2
    if start<0 or start+n>len(x):return None
    samples=x[start:start+n]
    rms=float(np.sqrt(np.mean(samples**2)))
    signal=(samples-np.mean(samples))*np.hanning(n)
    ps=np.abs(np.fft.rfft(signal))**2
    freq=np.fft.rfftfreq(n,1/rate)
    usable=(freq>=150)&(freq<=8000)
    total=float(ps[usable].sum())
    if total<1e-20:return None
    high=float(ps[usable&(freq>=2200)].sum())/total
    centroid=float((freq[usable]*ps[usable]).sum())/total
    bands=[]
    for low,up in zip(EDGES[:-1], EDGES[1:]):
        bands.append(float(np.log10(float(np.mean(ps[(freq>=low)&(freq<up)]))+1e-10)))
    bands=np.asarray(bands)
    bands-=bands.mean()
    return {'hf_fraction':float(high),'centroid_hz':centroid,'rms_dbfs':20*math.log10(max(rms,1e-12)), 'shape':bands.tolist()}


def _one_features(x,rate,anchor,manual_rel):
    start,end=manual_rel
    # Relative to fixed OTO 240ms reference. D4 user annotated these intervals;
    # other banks get a *transferred* interval requiring independent review.
    start=anchor+(start-240)
    end=anchor+(end-240)
    seq=[sample_metrics(x,rate,start+frac*(end-start)) for frac in (.15,.325,.5,.675,.85)]
    if any(v is None for v in seq): return None
    trajectory=[]
    for relative in range(-80,102,2):
        metric=sample_metrics(x,rate,anchor+relative)
        if metric is None:continue
        trajectory.append((relative,metric))
    def window(a,b,key):
        v=[row[1][key] for row in trajectory if a<=row[0]<=b and row[1]['rms_dbfs']>-53]
        return float(np.median(v)) if v else None
    early,late=window(-60,-20,'hf_fraction'),window(40,80,'hf_fraction')
    mean_shape=np.asarray([v['shape'] for v in seq])
    return {
      'candidate_interval_ms':[round(start,2),round(end,2)],
      'hf_early':round(early,5) if early is not None else None,
      'hf_late':round(late,5) if late is not None else None,
      'hf_rise':round(late-early,5) if early is not None and late is not None else None,
      'hf_timecourse':[[t,round(v['hf_fraction'],5)] for t,v in trajectory],
      'spectral_shape_five_positions':np.round(mean_shape,5).tolist(),
      'rms_dbfs_five_positions':[round(v['rms_dbfs'],2) for v in seq],
    }


def build_audit(config: dict)->dict:
    if config.get('schema')!='phonoweave.multisubbank_input.v1':raise ValueError('unsupported input schema')
    source_provenance = config.get('source_provenance', {'generation_method':'unknown','attestation':'not_provided'})
    if (not isinstance(source_provenance,dict) or
        source_provenance.get('generation_method') not in ('unknown','diffsinger_direct_render','human_recorded') or
        source_provenance.get('attestation') not in ('not_provided','author_confirmed')):
        raise ValueError('unsupported or missing source provenance declaration')
    if (source_provenance['generation_method']=='diffsinger_direct_render'
        and source_provenance['attestation']!='author_confirmed'):
        raise ValueError('DiffSinger generation claim requires author confirmation')
    banks=[make_bank(item) for item in config.get('banks',[])]
    if not banks or len({b.id for b in banks})!=len(banks):raise ValueError('missing or duplicate banks')
    # A single DiffSinger voicebank can expose multiple authored styles.
    # Do not equate a shared voicebank identity with a known checkpoint or
    # independent speakers. Preserve the author's claim as metadata only.
    relation=source_provenance.get('voice_style_relation','unknown')
    if relation not in ('unknown','same_voicebank_different_styles','different_voicebanks'):
        raise ValueError('unsupported voice-style relationship')
    if relation=='same_voicebank_different_styles':
        bank_id=source_provenance.get('voicebank_id')
        style_ids=source_provenance.get('voice_styles')
        if (source_provenance['attestation']!='author_confirmed'
            or not isinstance(bank_id,str) or not bank_id.strip()
            or not isinstance(style_ids,list)
            or len(style_ids)!=len(set(style_ids))
            or set(style_ids)!={b.style for b in banks}):
            raise ValueError('shared voicebank claim requires author confirmation, '
                             'nonblank voicebank id and exact voice-style list')
    if sum(b.reference for b in banks)!=1:raise ValueError('exactly one D4 reviewed reference required')
    if not next(b for b in banks if b.reference).id=='mang_D4':
        raise ValueError('only user-confirmed mang_D4 reference annotation supported')
    metadata=[];raw_records={};cliprows=[];wavdur=defaultdict(dict);allclips={}
    for bank in banks:
        with zipfile.ZipFile(bank.archive) as z:
            files={n for n in z.namelist()}
            name=bank.root+'oto.ini'
            if name not in files:raise ValueError(f'{bank.id}: missing expected OTO')
            oto_bytes=z.read(name)
            rows=parse_oto(oto_bytes,bank.prefix)
            good=[r for r in rows if not r['invalid_prefix']]
            wav_names=sorted(n for n in files if n.startswith(bank.root) and n.endswith('.wav') and '/' not in n[len(bank.root):])
            wav_base={n[len(bank.root):] for n in wav_names}
            # Read WAV headers from each subbank. No audio changes or extraction.
            for name in wav_names:
                wavdur[name[len(bank.root):]][bank.id]=wav_header(z,name)['duration_ms']
            metadata.append({'bank':bank.id,'voice_style':bank.style,'pitch':bank.pitch,
                'archive_basename':bank.archive.name,'archive_sha256':_sha_file(bank.archive),
                'oto_sha256':sha256(oto_bytes).hexdigest(),'oto_rows':len(rows),
                'invalid_prefix_rows':len(rows)-len(good),'wav_count':len(wav_names),
                'reference_human_intervals':bank.reference})
            raw_records[bank.id]=good
            by_key=defaultdict(list)
            for r in good:
                by_key[(r['wav'],r['body'])].append(r)
            # Only WAVs actually referenced by the frozen 6-target QC pilot.
            needed={x[2] for x in TARGETS}
            decoded={x:wav_data(z,bank.root+x) for x in needed if x in wav_base}
            for left,right,wav in TARGETS:
                key=(wav,f'{left} {right}')
                candidates=by_key[key]
                if len(candidates)!=1:
                    cliprows.append({'bank':bank.id,'voice_style':bank.style,'pitch':bank.pitch,'left':left,'right':right,
                        'status':'missing_or_duplicate_source_oto','matched_oto_rows':len(candidates)})
                    continue
                row=candidates[0]
                if wav not in decoded:raise ValueError('missing target WAV')
                audio,rate,pcmsha=decoded[wav]
                offset,consonant,cutoff,preutterance,overlap=row['values']
                anchor=offset+(preutterance+overlap)/2
                if anchor<100 or anchor+160>len(audio)*1000/rate:
                    raise ValueError('anchor outside WAV for '+bank.id+left+right)
                metrics=_one_features(audio,rate,anchor,D4_REVIEWED_INTERVALS[(left,right)])
                if metrics is None:raise ValueError('unmeasurable candidate window')
                idrow={'bank':bank.id,'voice_style':bank.style,'pitch':bank.pitch,'left':left,'right':right,
                    'source_wav':wav,'oto_line':row['line'],'original_alias':row['alias'],'original_oto_anchor_ms':round(anchor,2),
                    'source_pcm_sha256':pcmsha,'status':'human_reviewed_D4' if bank.reference else 'transferred_window_unverified',
                    **metrics}
                cliprows.append(idrow)
                allclips[(bank.id,left,right)]=idrow
    refid=next(b.id for b in banks if b.reference)
    ref=raw_records[refid]
    baseline=Counter((r['wav'],r['body'],r['values']) for r in ref)
    for meta in metadata:
        rows=raw_records[meta['bank']]
        counts=Counter((r['wav'],r['body'],r['values']) for r in rows)
        ali=Counter((r['wav'],r['body']) for r in rows)
        baseline_ali=Counter((r['wav'],r['body']) for r in ref)
        meta['exact_oto_rows_shared_with_D4']=sum((counts&baseline).values())
        meta['fraction_exact_oto_rows_vs_D4']=round(meta['exact_oto_rows_shared_with_D4']/max(1,len(rows)),6)
        meta['alias_and_wav_rows_shared_with_D4']=sum((ali&baseline_ali).values())
    spreads=[]
    for w,times in wavdur.items():
        if len(times)==len(banks):
            spreads.append({'wav':w,'duration_spread_ms':round(max(times.values())-min(times.values()),3),
                'duration_by_bank_ms':times})
    bygroup=defaultdict(list)
    for c in cliprows:
        if c.get('status') in ('human_reviewed_D4','transferred_window_unverified'):
            bygroup[(c['bank'],c['right'])].append(c)
    pairwise=[]
    for (bank,right),group in sorted(bygroup.items()):
        mapping={c['left']:c for c in group}
        if len(mapping)!=3:continue
        for a,b in [('i','ir'),('i','i0'),('ir','i0')]:
            x=np.asarray(mapping[a]['spectral_shape_five_positions'])
            y=np.asarray(mapping[b]['spectral_shape_five_positions'])
            dist=float(np.sqrt(np.mean((x-y)**2)))
            pairwise.append({'bank':bank,'style':mapping[a]['voice_style'],'pitch':mapping[a]['pitch'],
                             'right':right,'left_a':a,'left_b':b,'distance_5x12_logpower':round(dist,5),
                             'status':'diagnostic_only_unreviewed_cross_bank_boundaries'})
    # Exact same decoded audio blob for targeted source WAV is flag, but distinct
    # hashes cannot PROVE independent takes, since transforms alter hashes.
    hashes=defaultdict(list)
    for (bid,l,r),row in allclips.items():hashes[(row['source_wav'],row['source_pcm_sha256'])].append(bid)
    duplicates=[{'source_wav':wav,'banks':sorted(set(ids))} for (wav,sha),ids in hashes.items() if len(set(ids))>1]
    summary={
      'voice_styles':sorted({b.style for b in banks}),
      'subbanks':len(banks),
      'wav_count_excluding_demo':sum(x['wav_count'] for x in metadata),
      'oto_rows':sum(x['oto_rows'] for x in metadata),
      'target_oto_intervals':len(allclips),
      'source_wav_duration_spread_ms_median':round(float(np.median([x['duration_spread_ms'] for x in spreads])),3) if spreads else None,
      'source_wav_duration_spread_ms_max':round(float(max(x['duration_spread_ms'] for x in spreads)),3) if spreads else None,
      'target_pcm_hash_cross_bank_duplicates':duplicates,
      'all_subbanks_cover_all_six_targets':len(allclips)==len(banks)*6,
    }
    return {
       'schema':SCHEMA,'status':'existing_voicebank_descriptive_only_not_independent_replicates',
       'summary':summary,'subbanks':metadata,
       'duration_alignment':spreads,'targets':cliprows,'pairwise':pairwise,
       'limitations':[
         'The four Mang and three Lu pitches are two styles of the same author-confirmed LynneQing voicebank, not two independent singer populations or seven independent speakers (for the LynneQing manifest).',
         'If author-confirmed DiffSinger rendered, these are synthetic model outputs, not human singing sessions or independent human repetitions. Checkpoint, prompt and inference settings are not determined from the WAVs.',
         'Most OTO timings are identical across pitch; transferred D4 manually marked intervals are NOT verified boundaries in the other six banks.',
         'Matched fixed-length recordings across subbanks may reflect a recording template, synchronized capture, reuse or audio processing; different PCM hashes cannot prove independent takes.',
         'Spectral distances retain pitch/timbre and alias boundary alignment confounds, so they cannot establish share/split utility.',
         'No held-out independent session, real resampler render, pitch-adjusted control or perception test.',
         'No original WAV, alias, OTO or existing bank is altered.',
       ],
       'source_provenance':source_provenance,
       'independent_recording_status':('not_human_recording_repeats_synthetic_output'
           if source_provenance['generation_method']=='diffsinger_direct_render'
           else 'unknown'),
       'evaluation_population':('diffsinger_generated_voicebank_audio'
           if source_provenance['generation_method']=='diffsinger_direct_render'
           else 'undetermined'),
       'approved_split_or_share_decisions':0,
       'speaker_generalization':'not_evaluated','actual_synthesis_gain':'not_evaluated'
    }


def _sha_file(p:Path)->str:
    digest=sha256()
    with p.open('rb') as f:
        for buf in iter(lambda:f.read(4*1024*1024),b''):digest.update(buf)
    return digest.hexdigest()


def write_reports(result:dict,out:Path)->None:
    out=out.resolve()
    if out.exists():raise FileExistsError('report output exists')
    out.mkdir(parents=True)
    with (out/'summary.json').open('w',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    for name,rows in [('subbank_inventory.csv',result['subbanks']),('pairwise_acoustic_shapes.csv',result['pairwise'])]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            if rows:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    targets=result['targets']; cols=['bank','voice_style','pitch','left','right','source_wav','oto_line','original_alias','original_oto_anchor_ms','status','hf_early','hf_late','hf_rise','candidate_interval_ms']
    with (out/'vc_target_features.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in targets:w.writerow({k:r.get(k) for k in cols})
    with (out/'wav_duration_spread.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['wav','duration_spread_ms']);w.writeheader()
        for r in result['duration_alignment']:w.writerow({k:r[k] for k in ['wav','duration_spread_ms']})


def main(argv=None):
    p=argparse.ArgumentParser(description='Read-only multi-pitch VC audit')
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv)
    try:
        result=build_audit(json.loads(a.manifest.read_text('utf-8')))
        write_reports(result,a.output)
    except (ValueError,OSError,zipfile.BadZipFile) as exc:p.error(str(exc))
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2))
    print('Independent-take status unknown; no acoustic split/share claim.')
    return 0

if __name__=='__main__':raise SystemExit(main())
