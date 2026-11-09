"""Read-only phone-core and transition segmentation of source-linked UTAU ZIPs.

An author-reviewed *transition interval* is NOT a point phoneme boundary.
This prototype extracts conservative vowel/nucleus and consonant cores,
preserves the uncertain transition, and abstains on nasal onset. Automatic
/sh/ frication onset is acoustic evidence, not a certified phonetic boundary.
No alias/OTO/recording scheme edits, and no automatic split/merge decisions.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
import statistics
import wave
import zipfile
import numpy as np

SCHEMA='phonoweave.phone_core_segments.v0.1'
REVIEW_COLUMNS=('right_surface','left_surface','original_alias','source_wav','oto_line',
    'boundary_onset_ms_in_excerpt','boundary_offset_ms_in_excerpt','target_sound_present')
TARGETS={
 ('i','m'):'vcsection2.wav', ('ir','m'):'vcsection2.wav', ('i0','m'):'vcsection2.wav',
 ('i','sh'):'vcsection3.wav', ('ir','sh'):'ingi0ir.wav', ('i0','sh'):'vcsection3.wav',
}
LABEL_NAMES={'i':'li_nucleus_i','ir':'shi_ri_nucleus_ir','i0':'zi_ci_nucleus_i0'}

@dataclass(frozen=True)
class Bank:
    style:str
    pitch:str
    archive:Path
    root:str
    prefix:str
    @property
    def uid(self): return f'{self.style}_{self.pitch}'


def _sha_file(path:Path)->str:
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _bank(row):
    if not all(isinstance(row.get(k),str) and row[k] for k in ('style','pitch','archive','root','prefix')):
        raise ValueError('bank descriptor incomplete')
    if row['root'].startswith('/') or not row['root'].endswith('/') or '..' in Path(row['root']).parts:
        raise ValueError('invalid archive root')
    return Bank(row['style'],row['pitch'],Path(row['archive']).expanduser().resolve(strict=True),row['root'],row['prefix'])


def _oto_rows(data:bytes)->list[dict]:
    try: src=data.decode('utf-8-sig')
    except UnicodeDecodeError: src=data.decode('cp932')
    out=[]
    for n,line in enumerate(src.splitlines(),1):
        if not line.strip(): continue
        try:
            wav,rem=line.split('=',1)
            alias,offset,consonant,cutoff,preutterance,overlap=rem.rsplit(',',5)
            times=[float(v) for v in (offset,consonant,cutoff,preutterance,overlap)]
            if not all(math.isfinite(t) for t in times):raise ValueError('nonfinite OTO value')
        except (ValueError,IndexError) as e:
            raise ValueError(f'malformed OTO at line {n}') from e
        out.append(dict(line=n,wav=wav,alias=alias,oto_values=times,
             anchor_ms=times[0]+(times[3]+times[4])/2))
    return out


def _pcm_from_zip(z:zipfile.ZipFile,path:str):
    raw=z.read(path)
    with wave.open(io.BytesIO(raw),'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getcomptype()!='NONE':
            raise ValueError(f'expected mono PCM16: {path}')
        rate=w.getframerate();rawpcm=w.readframes(w.getnframes())
    return np.frombuffer(rawpcm,dtype='<i2').astype(np.float64)/32768.,rate,sha256(raw).hexdigest()


def _crop(x,rate,start_ms,end_ms):
    a=round(start_ms*rate/1000); b=round(end_ms*rate/1000)
    if not 0<=a<b<=len(x):raise ValueError('invalid excerpt coordinates')
    return x[a:b].copy()


def _write_wav(path:Path,x,rate):
    # Do not resample, filter, normalize or alter samples in the source core.
    quant=np.round(np.clip(x,-1.,1.)*32768).astype('<i2')
    with path.open('xb') as f:
        with wave.open(f,'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate)
            w.writeframes(quant.tobytes())
    return sha256(path.read_bytes()).hexdigest()


def _frames(x:np.ndarray,rate:int):
    """20 ms Hann frames, 2ms hop; onset timing only to a frame resolution."""
    n=round(.020*rate);hop=round(.002*rate)
    windows=np.lib.stride_tricks.sliding_window_view(x,n)[::hop]
    centers=(np.arange(len(windows))*hop+n/2)*1000/rate
    win=np.hanning(n)
    y=(windows-windows.mean(axis=1,keepdims=True))*win
    ps=np.abs(np.fft.rfft(y,axis=1))**2
    freqs=np.fft.rfftfreq(n,1/rate)
    base=ps[:,(freqs>=150)&(freqs<=8000)].sum(axis=1)
    high=ps[:,(freqs>=2200)&(freqs<=8000)].sum(axis=1)
    hf=high/np.maximum(base,1e-18)
    db=20*np.log10(np.maximum(np.sqrt((windows**2).mean(axis=1)),1e-12))
    return centers,hf,db


def frication_event(x,rate):
    """First persistent high-band transition near the 240ms OTO anchor.

    Returns acoustic onset for /sh/ only, NEVER a phoneme boundary estimate.
    Skip absent/quiet/transient excursions; no trained model is implied.
    """
    t,hf,db=_frames(x,rate)
    baseline=(t>=175)&(t<=218)&(db>-53)
    if np.sum(baseline)<5:return None
    baseline_hf=float(np.median(hf[baseline]))
    if baseline_hf>=.28:return None
    eligible=(t>=212)&(t<=350)
    indices=np.flatnonzero(eligible)
    for idx in indices:
        if idx+5>=len(hf):break
        if (np.all(hf[idx:idx+5]>=.50) and
            np.all(db[idx:idx+5]>-53) and
            float(np.median(hf[idx:idx+5]))-baseline_hf>.42):
            return dict(event_center_ms=round(float(t[idx]),2),
               baseline_hf=round(baseline_hf,4),
               onset_hf=round(float(hf[idx]),4),
               evidence='persistent_highband_frication_not_phoneme_boundary')
    return None


def _review_rows(path:Path, ref_bank:str):
    rows={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f)
        if not set(REVIEW_COLUMNS).issubset(rd.fieldnames or []):raise ValueError('review columns incomplete')
        for item in rd:
            k=(item['left_surface'],item['right_surface'])
            if k in rows or k not in TARGETS:raise ValueError('duplicate/unexpected review target')
            if item['source_wav']!=TARGETS[k] or item['target_sound_present']!='yes':
                raise ValueError('review source mismatch or sound absent')
            if item['original_alias']!=f'{ref_bank}{k[0]} {k[1]}':
                raise ValueError('review alias identity mismatch')
            try:
                a=float(item['boundary_onset_ms_in_excerpt']);b=float(item['boundary_offset_ms_in_excerpt'])
                if not 100<a<b<430:raise ValueError('review interval outside local pilot')
            except (ValueError,TypeError) as e:raise ValueError('invalid author interval') from e
            rows[k]=(a,b,int(item['oto_line']))
    if set(rows)!=set(TARGETS):raise ValueError('exact six human-reviewed targets required')
    return rows


def _textgrid(path,core_left,transition,core_right,left,right):
    """Three tiers: two sparse conservative cores + one uncertain transition.

    Only the 600ms  local crop is represented; blank intervals are not silence.
    """
    def intervals(a,b,label):
        if not (0<=a<b<=600): raise ValueError('TextGrid interval invalid')
        return [(0,a,''),(a,b,label),(b,600,'')]
    def tier(name,entries):
        data=[f'    item [{tier.i}]:', '        class = "IntervalTier"',
              f'        name = "{name}"','        xmin = 0','        xmax = 0.6',
              f'        intervals: size = {len(entries)}']
        for j,(a,b,t) in enumerate(entries,1):
            if b<=a:continue
            data.extend([f'        intervals [{j}]:', f'            xmin = {a/1000:.6f}',
                         f'            xmax = {b/1000:.6f}',f'            text = "{t}"'])
        return data
    content=['File type = "ooTextFile"','Object class = "TextGrid"','',
             'xmin = 0','xmax = 0.6','tiers? <exists>','size = 3','item []:']
    for i,(name,bounds,label) in enumerate((('left_nucleus_core',core_left,left),
             ('transition_uncertain',transition,'uncertain_transition'),
             ('right_consonant_core',core_right,right)),1):
        tier.i=i;content+=tier(name,intervals(*bounds,label))
    path.write_text('\n'.join(content)+'\n',encoding='utf-8')


def _ranges(left,right):
    """Conservative cores inside a local 600ms anchor-centered excerpt.

    Guard 12ms around transition; cores are 40ms each, not full phone spans.
    """
    if not 130<=left<right<=430 or right-left<18:return None
    a=(round(left-57,2),round(left-17,2))
    b=(round(left,2),round(right,2))
    c=(round(right+18,2),round(right+58,2))
    if a[0]<0 or c[1]>600:return None
    return a,b,c


def _load_review_pack(review_pack:Path):
    """Pin hand-reviewed transitions to the exact WAV bytes the author heard."""
    with zipfile.ZipFile(review_pack) as z:
        manifest=json.loads(z.read('review_manifest.json'))
        items={}
        for group in manifest['review_groups']:
            for row in group['clip_records']:
                key=(row['left_surface'],group['right_surface'])
                if key in items:raise ValueError('duplicate reference-pack clip')
                content=z.read(row['excerpt_filename'])
                if sha256(content).hexdigest()!=row['excerpt_sha256']:
                    raise ValueError('review-pack excerpt hash mismatch')
                items[key]=(row,content)
    if set(items)!=set(TARGETS):raise ValueError('review-pack six targets incomplete')
    return items,manifest['source']


def scan_right_sh_phone_cores(banks:list[Bank],root:Path,archive_hashes:dict) -> dict:
    """Scan every existing OTO `... sh` alias for an acoustic fricative core.

    No claim is made about the identity or boundary of the left phone. /sh/
    is a source-provided alias assumption plus high-frequency evidence, not
    confirmed independent phonetic annotation. Source contexts may share WAVs.
    """
    dest=root/'all_sh_phone_cores'
    dest.mkdir(parents=True,exist_ok=False)
    index=[]
    for bank in banks:
        with zipfile.ZipFile(bank.archive) as z:
            otodata=z.read(bank.root+'oto.ini')
            osh=sha256(otodata).hexdigest()
            rows=_oto_rows(otodata)
            selected=[r for r in rows if r['alias'].startswith(bank.prefix)
                      and r['alias'].endswith(' sh')
                      and len(r['alias'][len(bank.prefix):].split())==2]
            cache={}
            for row in selected:
                wav=row['wav'];left=row['alias'][len(bank.prefix):-3]
                if not left or not re.fullmatch(r'[A-Za-z0-9]+',left):
                    # Do not interpret punctuation/hidden tokens as a phone.
                    continue
                if wav not in cache:
                    cache[wav]=_pcm_from_zip(z,bank.root+wav)
                audio,rate,wavsha=cache[wav]
                anchor=row['anchor_ms'];start=anchor-240
                out=dict(bank=bank.uid,style=bank.style,pitch=bank.pitch,
                    right_alias='sh',left_surface_unverified=left,
                    original_alias=row['alias'],source_wav=wav,oto_line=row['line'],
                    oto_anchor_source_ms=round(anchor,3),
                    excerpt_source_start_ms=round(start,3),
                    source_wav_sha256=wavsha,
                    archive_sha256=archive_hashes[bank.archive],oto_sha256=osh,
                    status='abstain_no_acoustic_core')
                if start<0 or start+600>len(audio)*1000/rate:
                    index.append(out);continue
                excerpt=_crop(audio,rate,start,start+600)
                event=frication_event(excerpt,rate)
                if event is None:
                    index.append(out);continue
                e=event['event_center_ms']
                # Allow a guard after detected high-band rise to avoid treating
                # a transition onset/partial frame as stable phone material.
                begin,end=e+25,e+65
                if end>580:
                    index.append(out);continue
                ts,hf,db=_frames(excerpt,rate)
                mask=(ts>=begin)&(ts<=end)
                median_hf=float(np.median(hf[mask]));median_db=float(np.median(db[mask]))
                if median_hf<.6 or median_db<=-53:
                    index.append(out);continue
                location=dest/bank.uid/f'{left}_to_sh_OTO{row["line"]}'
                location.mkdir(parents=True,exist_ok=False)
                # Keep a playable 600ms context, so a reviewer can reject false cores.
                _write_wav(location/'source_context_600ms.wav',excerpt,rate)
                core=_crop(excerpt,rate,begin,end)
                core_hash=_write_wav(location/'sh_candidate_core_40ms.wav',core,rate)
                out.update(status='acoustic_frication_core_candidate_not_approved_phone',
                    frication_event_ms_in_excerpt=e,
                    core_interval_ms_in_excerpt=[round(begin,2),round(end,2)],
                    core_interval_ms_in_source=[round(start+begin,3),round(start+end,3)],
                    core_hf_fraction_median=round(median_hf,4),
                    core_rms_dbfs_median=round(median_db,2),
                    core_wav_sha256=core_hash,
                    phonetic_boundary_verified=False)
                (location/'provenance.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                index.append(out)
    index.sort(key=lambda x:(x['style'],x['pitch'],x['oto_line']))
    columns=['bank','style','pitch','left_surface_unverified','original_alias','source_wav',
       'oto_line','status','frication_event_ms_in_excerpt',
       'core_interval_ms_in_excerpt','core_interval_ms_in_source',
       'core_hf_fraction_median','source_wav_sha256']
    with (dest/'sh_phone_core_index.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in index:w.writerow({k:r.get(k) for k in columns})
    result={'schema':'phonoweave.right_sh_phone_core_scan.v0.1',
        'oto_sh_alias_candidates':len(index),
        'acoustic_sh_cores_extracted':sum(r['status'].startswith('acoustic_') for r in index),
        'abstentions':sum(r['status'].startswith('abstain_') for r in index),
        'phonetic_boundaries_certified':0,'no_split_or_merge':True,
        'per_bank':{},
        'limitations':[
          'High-frequency frication evidence is not a validated exact phoneme boundary.',
          'Left surface label is kept uninterpreted; right /sh/ is an author-format alias hypothesis.',
          'Multiple OTO rows may share one parent WAV and are not independent source recordings.',
          'These are source WAV excerpts, not a full singing synthesis or source scheme.',
        ]}
    for bank in banks:
        rows=[r for r in index if r['bank']==bank.uid]
        result['per_bank'][bank.uid]={'aliases':len(rows),
            'acoustic_cores':sum(r['status'].startswith('acoustic_') for r in rows),
            'distinct_source_wavs':len({r['source_wav'] for r in rows})}
    (dest/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


def _run(config:dict,review_csv:Path,output:Path,*,review_pack:Path):
    if config.get('schema')!='phonoweave.multisubbank_input.v1':raise ValueError('unexpected input manifest')
    banks=[_bank(b) for b in config['banks']]
    if len({b.uid for b in banks})!=len(banks):raise ValueError('duplicate banks')
    reference=[b for b in banks if b.uid=='mang_D4']
    if len(reference)!=1:raise ValueError('mang_D4 reference required')
    reviewed=_review_rows(review_csv,reference[0].prefix)
    review_items,review_source=_load_review_pack(review_pack)
    if _sha_file(reference[0].archive)!=review_source['archive_sha256']:
        raise ValueError('original D4 source ZIP hash differs from human-review package')
    ref_interval_events={}
    archive_hashes={}
    for bank in banks:
        if bank.archive not in archive_hashes:
            archive_hashes[bank.archive]=_sha_file(bank.archive)
    reports=[];excerpts=[]
    # No folder is created until all input data identities and timing are inspected.
    cached={}
    for bank in banks:
        with zipfile.ZipFile(bank.archive) as z:
            oto_path=bank.root+'oto.ini'
            allnames=set(z.namelist())
            if oto_path not in allnames:raise ValueError('missing OTO')
            oto_bytes=z.read(oto_path)
            oto_sha=sha256(oto_bytes).hexdigest()
            if bank.uid=='mang_D4' and oto_sha!=review_source['oto_sha256']:
                raise ValueError('D4 OTO changed since the author reviewed it')
            rows=_oto_rows(oto_bytes)
            selection={}
            for k,wav in TARGETS.items():
                alias=bank.prefix+k[0]+' '+k[1]
                matched=[r for r in rows if r['wav']==wav and r['alias']==alias]
                if len(matched)!=1:raise ValueError(f'{bank.uid}/{alias}: no unique OTO')
                selection[k]=matched[0]
            decoded={}
            for wav in set(TARGETS.values()):
                if bank.root+wav not in allnames:raise ValueError(f'{bank.uid}: missing source {wav}')
                decoded[wav]=_pcm_from_zip(z,bank.root+wav)
            for k in TARGETS:
                left,right=k; r=selection[k];audio,rate,source_sha=decoded[r['wav']]
                anchor=r['anchor_ms'];start=anchor-240;end=start+600
                if start<0 or end>len(audio)*1000/rate:raise ValueError('excerpt outside source WAV')
                # Decode from the original bank WAV rather than copying a reviewer fixture.
                clip=_crop(audio,rate,start,end)
                review_match=None
                if bank.uid=='mang_D4':
                    refrow,refwav=review_items[k]
                    if (refrow['oto_line']!=r['line'] or
                        refrow['original_alias']!=r['alias'] or
                        refrow['source_wav']!=r['wav'] or
                        refrow['source_wav_sha256']!=source_sha):
                        raise ValueError('D4 reviewed reference identity mismatch')
                    buf=io.BytesIO()
                    with wave.open(buf,'wb') as w:
                        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate)
                        w.writeframes(np.round(clip*32768).astype('<i2').tobytes())
                    if buf.getvalue()!=refwav:
                        raise ValueError('D4 excerpt differs from author-reviewed audio')
                    review_match=True
                event=frication_event(clip,rate) if right=='sh' else None
                if bank.uid=='mang_D4':
                    lo,hi,rev_line=reviewed[k]
                    if rev_line!=r['line']:raise ValueError('author OTO source line changed')
                    provenance='human_reviewed_transition_band'
                    if event and not lo-10<=event['event_center_ms']<=hi+10:
                        raise ValueError(f'author transition and acoustic event disagree: {k}')
                    if event: ref_interval_events[k]=(lo,hi,event['event_center_ms'])
                elif right=='sh' and event and k in ref_interval_events:
                    lo0,hi0,ev0=ref_interval_events[k]
                    shift=event['event_center_ms']-ev0
                    if abs(shift)<=75:
                        lo=lo0+shift;hi=hi0+shift
                        provenance='auto_frication_event_shifted_from_human_reference_unverified'
                    else:
                        lo=hi=None;provenance='abstain_acoustic_shift_outside_reference'
                else:
                    lo0,hi0,_=reviewed[k]
                    lo,hi=lo0,hi0
                    provenance=('nasal_transition_transferred_from_oto_unverified'
                                if right=='m' else 'abstain_no_acoustic_frication_event')
                    if right=='sh':lo=hi=None
                bounds=_ranges(lo,hi) if lo is not None else None
                # No phoneme core if the transition cannot be reasonably placed.
                if bounds:
                    cleft,transition,cright=bounds
                    _,hf,db=_frames(clip,rate)
                    ts,vals,levels=_frames(clip,rate)
                    x=(ts>=cleft[0])&(ts<=cleft[1]);y=(ts>=cright[0])&(ts<=cright[1])
                    lv=float(np.median(vals[x]));rv=float(np.median(vals[y]));loud_left=float(np.median(levels[x]));loud_right=float(np.median(levels[y]))
                    left_supported=bool(lv<.35 and loud_left>-55)
                    right_supported=bool(rv>.5 and loud_right>-53) if right=='sh' else False
                    # Every output is a candidate core. Acoustic support is not a reviewer approval.
                    right_state=('acoustically_supported_frication_core_not_boundaries' if right_supported else
                        ('nasal_core_unverified' if right=='m' else 'frication_core_not_acoustically_supported'))
                    left_state='vowel_like_core_supported' if left_supported else 'left_nucleus_core_unverified'
                else:
                    left_state=right_state='abstained';lv=rv=loud_left=loud_right=None
                meta={
                    'bank':bank.uid,'voice_style':bank.style,'pitch':bank.pitch,'left_alias':left,
                    'left_class_semantics':LABEL_NAMES[left],'right_phone':right,'original_alias':r['alias'],
                    'original_oto_line':r['line'],'source_wav':r['wav'],
                    'source_wav_sha256':source_sha,'archive_name':bank.archive.name,
                    'source_archive_sha256':archive_hashes[bank.archive],
                    'oto_ini_sha256':oto_sha,'review_csv_sha256':_sha_file(review_csv),
                    'D4_review_pack_excerpt_exactly_verified':review_match,
                    'oto_anchor_source_ms':round(anchor,3),'excerpt_source_start_ms':round(start,3),
                    'acoustic_frication_event_ms_in_excerpt':event['event_center_ms'] if event else None,
                    'acoustic_event_is_phone_boundary':False,
                    'transition_interval_provenance':provenance,
                    'transition_interval_ms_in_excerpt':[round(lo,2),round(hi,2)] if lo is not None else None,
                    'phone_borders_verified':False,
                    'left_core_evidence':left_state,'right_core_evidence':right_state,
                    'left_core_hf_median':round(lv,4) if lv is not None else None,
                    'right_core_hf_median':round(rv,4) if rv is not None else None,
                    'left_core_rms_dbfs':round(loud_left,1) if loud_left is not None else None,
                    'right_core_rms_dbfs':round(loud_right,1) if loud_right is not None else None,
                    'status':'human_reviewed_transition_with_candidate_phone_cores'
                     if bank.uid=='mang_D4' else ('provisional_auto_segment' if bounds else 'abstain_no_segment'),
                    'core_bounds_ms_in_excerpt': {p:[round(a,2),round(b,2)] for p,(a,b) in zip(('left','transition','right'),bounds)} if bounds else None,
                }
                reports.append(meta)
                excerpts.append((meta,clip,rate,bounds))
    if output.exists(): raise FileExistsError('segmentation output exists (no overwrites)')
    output.mkdir(parents=True)
    try:
        for meta,clip,rate,bounds in excerpts:
            d=output/'banks'/meta['bank']/f"{meta['left_alias']}_to_{meta['right_phone']}"
            d.mkdir(parents=True,exist_ok=False)
            meta['excerpt_wav_sha256']=_write_wav(d/'excerpt_600ms.wav',clip,rate)
            if bounds:
                left,transition,right=bounds
                for filename,(a,b) in [('left_nucleus_core.wav',left),('uncertain_transition.wav',transition),('right_consonant_core.wav',right)]:
                    meta[filename.replace('.wav','_sha256')]=_write_wav(d/filename,_crop(clip,rate,a,b),rate)
                for role,(a,b) in [('left_core',left),('uncertain_transition',transition),('right_core',right)]:
                    meta[f'{role}_source_ms']=[round(meta['excerpt_source_start_ms']+a,3),round(meta['excerpt_source_start_ms']+b,3)]
                _textgrid(d/'phones.TextGrid',left,transition,right,meta['left_alias'],meta['right_phone'])
            else:
                # Abstention always retains WAV and metadata but creates no guessed phone borders.
                (d/'ABSTAIN.txt').write_text('No justified phone-core extraction; manual transition review required.\n')
            (d/'provenance.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        sh_bulk=scan_right_sh_phone_cores(banks,output,archive_hashes)
        result={
            'all_sh_phone_core_scan':{k:v for k,v in sh_bulk.items() if k!='per_bank'},
            'schema':SCHEMA,'total_targets':len(reports),'human_reviewed_bands':sum(r['status'].startswith('human_') for r in reports),
            'provisional_auto_segments':sum(r['status']=='provisional_auto_segment' for r in reports),
            'abstentions':sum(r['status']=='abstain_no_segment' for r in reports),
            'right_frication_cores_acoustically_supported':sum(r['right_core_evidence'].startswith('acoustically_supported') for r in reports),
            'phoneme_point_boundaries_confirmed':0,'whole_phone_boundaries_confirmed':0,
            'segmentation_support_scope':'6 author-defined VC classes x supplied subbanks (not full Mandarin phoneme aligner)',
            'no_auto_split_merge':True,
            'limitations':[
                'Human time labels define a transition region and are not certified exact phoneme boundaries.',
                'Left and right 40ms source segments are conservative candidate phone cores, not complete phoneme intervals.',
                'For unreviewed /sh/, a high-band frication event shifts a D4 transition template; it is NOT a model-predicted phoneme boundary.',
                'For unreviewed /m/, no acoustic onset claim is made; the transferred OTO-relative interval is a review-only candidate.',
                'This does not infer phonemes from alias strings beyond the six author-confirmed pilot classes.',
                'Speaker/source ZIPs and original OTO/aliases are read-only; no new recording scheme is auto-approved.',
            ],
            'segments':reports}
        (output/'segmentation_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        keys=['bank','voice_style','pitch','left_alias','right_phone','original_alias','original_oto_line','source_wav',
              'status','transition_interval_provenance','acoustic_frication_event_ms_in_excerpt','transition_interval_ms_in_excerpt',
              'left_core_evidence','right_core_evidence','left_core_source_ms','uncertain_transition_source_ms','right_core_source_ms']
        with (output/'segment_index.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
            for row in reports:w.writerow({k:row.get(k) for k in keys})
        (output/'README_CN.md').write_text('''# PhonoWeave：真实原始 WAV 音素核心切分 v0.1\n\n这是**真正执行了音频裁切**的研究输入，而不是对声音单位的分拆／共享建议。\n\n- `banks/<声线_音阶>/<左类别_to_右类别>/` 存有 600ms 原始音频摘录、40ms 左核候选、过渡区间、40ms 右辅音核候选和 Praat TextGrid。音频样本保持原采样率和 PCM 样本，没有滤波、变调或归一化。\n- `segment_index.csv` 与 `provenance.json` 记录从 excerpt 时间转换回原始 WAV 的绝对毫秒时间。\n- 芒 D4 的六条区间来自作者提交的 v0.7 CSV；它们代表**人工选取的过渡区间**，不代表两个精确音素边界。\n- 其余音阶的 `/sh/` 采用声学高频进入事件来调节过渡窗口（若不可信则弃判）；`/m/` 无可靠通用高频依据，因此只转移 OTO 相对模板，明确标为待复核。\n- 左右核心段为保守 40ms 样本，**不是**整音素的最终切边，也没有把整个 WAV 归类为一个音素。TextGrid 的空段表示未赋标签，而不是静音。\n- `all_sh_phone_cores/` 扩展扫描全部已有 `... sh` OTO 别名，只裁切有持久高频摩擦证据的 `/sh/` **40ms 核心候选**（无需假定左侧音素是什么）。每条保留 600ms 听辨上下文、原 WAV 绝对时间和来源校验。所有边界仍待复核。
- 首轮过渡区间只覆盖 `i / ir / i0 → m / sh` 六类；其它音素仍需按发音类型逐步扩展规则或训练人工边界模型。\n- 不得据此自动修改 OTO、alias、音素清单或推荐拆分／共享。\n''',encoding='utf-8')
    except Exception:
        # Output is a newly created research folder only; clean partial outputs.
        import shutil
        shutil.rmtree(output)
        raise
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description='Source-linked phone-core extraction, not automatic split/share')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--review-pack',type=Path,required=True,
                   help='original offline human-review ZIP containing matching audio bytes')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        data=json.loads(a.manifest.expanduser().read_text(encoding='utf-8'))
        report=_run(data,a.review.expanduser(),a.output.expanduser(),
                    review_pack=a.review_pack.expanduser())
    except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as e:
        p.error(str(e))
    print(json.dumps({k:v for k,v in report.items() if k!='segments'},ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
