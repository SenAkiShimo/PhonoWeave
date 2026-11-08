"""Descriptive acoustics for *manually reviewed* Lynne-style VC transition excerpts.

No automatic phoneme/synthesis inference. All comparisons are in-sample,
1 recorded transition per condition, and do not measure efficacy.

Usage: python phonoweave_reviewed_vc_probe.py --pack PACK --annotations CSV --output NEW_FOLDER
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from itertools import combinations, product
from pathlib import Path
import wave
import math
import os
import tempfile

import numpy as np

SCHEMA = 'phonoweave.vc_human_reviewed_acoustic_diagnostics.v0.1'
SOURCE_SCHEMA = 'phonoweave.vc_boundary_feasibility_gate.v0.1'
REQUIRED_COLUMNS = ('right_surface','left_surface','original_alias','source_wav','oto_line',
 'boundary_onset_ms_in_excerpt','boundary_offset_ms_in_excerpt',
 'target_sound_present','listening_comments','reviewer')
LEFTS=('i','ir','i0')
RIGHTS=('m','sh')
EDGES=np.geomspace(150,8000,13)

def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _pcm(content: bytes) -> tuple[np.ndarray,int]:
    with wave.open(io.BytesIO(content),'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getcomptype()!='NONE':
            raise ValueError('expected original mono PCM16 audio')
        rate=w.getframerate()
        total=w.getnframes()
        raw=w.readframes(total)
    if len(raw)!=total*2 or not 0.58<=total/rate<=0.62:
        raise ValueError('incomplete or wrong-duration source')
    return np.frombuffer(raw,dtype='<i2').astype('float64')/32768.0,rate

def load_review(pack:Path,annotations:Path):
    root=Path(pack).expanduser().resolve(strict=True)
    manifest_path=root/'review_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema')!=SOURCE_SCHEMA or manifest.get('status')!='auditory_boundary_review_required_no_split_share_evidence':
        raise ValueError('invalid frozen review manifest')
    source={}
    for g in manifest['review_groups']:
        right=g['right_surface']
        for r in g['clip_records']:
            left=r['left_surface'];key=(right,left)
            if key in source or right not in RIGHTS or left not in LEFTS:
                raise ValueError('duplicate or unexpected source identity')
            rel=f'right_{right}/{left}.wav'
            if r['excerpt_filename']!=rel or r['excerpt_oto_anchor_ms']!=240:
                raise ValueError('noncanonical source excerpt')
            path=(root/rel).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError('source file escaped root')
            content=path.read_bytes()
            if _hash(content)!=r['excerpt_sha256']:
                raise ValueError(f'source audio SHA-256 mismatch: {rel}')
            sig,rate=_pcm(content)
            source[key]={'metadata':r,'audio':sig,'rate':rate}
    expected=set(product(RIGHTS,LEFTS))
    if set(source)!=expected:raise ValueError('expected exactly six frozen source identities')
    with Path(annotations).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames is None or set(reader.fieldnames)!=set(REQUIRED_COLUMNS):
            raise ValueError('annotation columns do not match frozen schema')
        rows=list(reader)
    if len(rows)!=6:raise ValueError('need exactly six independently marked rows')
    reviewed={}
    for row in rows:
        k=(row['right_surface'],row['left_surface'])
        if k not in source or k in reviewed:raise ValueError('missing/duplicate review key')
        r=source[k]['metadata']
        if row['original_alias']!=r['original_alias'] or row['source_wav']!=r['source_wav'] or str(r['oto_line'])!=row['oto_line']:
            raise ValueError('annotation source identity mismatch')
        if row['target_sound_present'].strip().lower()!='yes' or not row['reviewer'].strip():
            raise ValueError('nonaffirmative/unreviewed annotation must not become measurement')
        try:a=float(row['boundary_onset_ms_in_excerpt']);b=float(row['boundary_offset_ms_in_excerpt'])
        except (ValueError,TypeError) as exc: raise ValueError('incomplete manual boundary') from exc
        if not all(map(math.isfinite,(a,b))) or not (20<=a<b<=580) or b-a<20:
            raise ValueError('manual boundary is outside extractable time range')
        reviewed[k]={'start_ms':a,'end_ms':b,'duration_ms':round(b-a,3),
                     'reviewer':row['reviewer'].strip(),**source[k]}
    return reviewed,{'source_review_manifest_sha256':_hash(manifest_path.read_bytes()),
                     'annotation_sha256':_hash(Path(annotations).read_bytes()),
                     'source_archive_sha256':manifest['source']['archive_sha256'],
                     'source_oto_sha256':manifest['source']['oto_sha256']}

def _frame(signal:np.ndarray,rate:int,center_ms:float):
    n=round(0.016*rate)
    start=round(center_ms*rate/1000)-n//2
    if start<0 or start+n>len(signal):raise ValueError('feature window outside audio')
    x=signal[start:start+n]
    rms=math.sqrt(float(np.mean(x*x)))
    freqs=np.fft.rfftfreq(n,1/rate)
    power=np.abs(np.fft.rfft(np.hanning(n)*(x-x.mean())))**2
    region=(freqs>=150)&(freqs<=8000)
    relevant=power[region]
    total=float(relevant.sum())
    high=float(power[(freqs>=2200)&(freqs<=8000)].sum())
    centroid=float(np.dot(freqs[region],relevant)/max(total,1e-12))
    bands=[]
    for lo,hi in zip(EDGES[:-1],EDGES[1:]):
        p=power[(freqs>=lo)&(freqs<hi)]
        bands.append(math.log10(float(np.mean(p))+1e-9))
    mean=float(np.mean(bands))
    return {'rms_dbfs':round(20*math.log10(max(rms,1e-10)),3),
            'hf_energy_fraction':round(high/max(total,1e-12),5),
            'spectral_centroid_hz':round(centroid,2),
            'normalized_log_band_shape':np.array(bands)-mean}

def _profile(r,start_delta=0,end_delta=0):
    a=r['start_ms']+start_delta;b=r['end_ms']+end_delta
    if b-a<24:raise ValueError('sensitivity shifted interval too short')
    centers=np.linspace(a+8,b-8,5)
    frames=[_frame(r['audio'],r['rate'],c) for c in centers]
    shape=np.mean(np.array([x['normalized_log_band_shape'] for x in frames]),axis=0)
    return {'mean_centroid_hz':float(np.mean([x['spectral_centroid_hz'] for x in frames])),
            'mean_hf_fraction':float(np.mean([x['hf_energy_fraction'] for x in frames])),
            'mean_rms_dbfs':float(np.mean([x['rms_dbfs'] for x in frames])),
            'early_hf_fraction':frames[0]['hf_energy_fraction'],
            'middle_hf_fraction':frames[2]['hf_energy_fraction'],
            'late_hf_fraction':frames[4]['hf_energy_fraction'],
            'shape':shape,'times_ms':centers.tolist()}

def _distance(a,b):
    return float(np.sqrt(np.mean((a-b)**2)))

def compile_probe(pack:Path,annotations:Path) -> dict:
    reviewed,prov=load_review(pack,annotations)
    stats=[];pairwise=[]
    for right in RIGHTS:
        for left in LEFTS:
            r=reviewed[(right,left)]
            z=_profile(r)
            stats.append({'right_surface':right,'left_surface':left,'original_alias':r['metadata']['original_alias'],
                          'source_wav':r['metadata']['source_wav'],'source_oto_line':r['metadata']['oto_line'],
                          'source_excerpt_sha256':r['metadata']['excerpt_sha256'],
                          'annotated_start_ms':r['start_ms'],'annotated_end_ms':r['end_ms'],
                          'annotated_duration_ms':r['duration_ms'],'mean_centroid_hz':round(z['mean_centroid_hz'],1),
                          'mean_hf_fraction':round(z['mean_hf_fraction'],4),
                          'mean_rms_dbfs':round(z['mean_rms_dbfs'],1),
                          'early_hf_fraction':z['early_hf_fraction'],
                          'middle_hf_fraction':z['middle_hf_fraction'],
                          'late_hf_fraction':z['late_hf_fraction']})
        for la,lb in combinations(LEFTS,2):
            a,b=reviewed[(right,la)],reviewed[(right,lb)]
            primary=_distance(_profile(a)['shape'],_profile(b)['shape'])
            variations=[]
            for sa,ea,sb,eb in product((-6,0,6),repeat=4):
                variations.append(_distance(_profile(a,sa,ea)['shape'], _profile(b,sb,eb)['shape']))
            pairwise.append({'right_surface':right,'left_a':la,'left_b':lb,
                'same_source_wav':a['metadata']['source_wav']==b['metadata']['source_wav'],
                'descriptive_log_band_rms_distance':round(primary,4),
                'plusminus6ms_shift_distance_min':round(min(variations),4),
                'plusminus6ms_shift_distance_median':round(float(np.median(variations)),4),
                'plusminus6ms_shift_distance_max':round(max(variations),4),
                'sensitivity_grid_size':len(variations),
                'independent_repeats_per_condition':1,
                'diagnostic_only':True})
    return {'schema':SCHEMA,'status':'in_sample_descriptive_diagnostics_only',
      'source':prov,'review':{'rows':len(stats),'manual_annotation_confirmed_by_reviewer':True,
         'independent_boundary_adjudicators':1,'independent_recording_sessions_verified':False},
      'method':{'window_ms':16,'profile_points_per_interval':5,
        'band_edges_hz':[round(float(v),2) for v in EDGES.tolist()],
        'spectral_shape':'per_frame_mean_centered_log10_band_power',
        'pair_distance':'root_mean_square_difference_of_12_band_log_shapes',
        'sensitivity': 'independent start/end shifts -6,0,+6 ms on both segments (81 variants)\n; not a statistical confidence interval'},
      'interval_features':stats,'within_consonant_pair_distances':pairwise,
      'research_decision':{'go_for_exploratory_descriptive_analysis':True,
         'go_for_synthesis_efficacy_claim':False,
         'go_for_automatic_split_share':False,
         'blockers':['one excerpt per nucleus/target pairing','no independent repeat sessions or adjudicators',
           'features include variable parts of vowel-to-consonant trajectories and contextual recording effects',
           'no actual synthesis comparison with equivalent costs']}}

def export_probe(report,output:Path,*,source_pack:Path|None=None):
    output=Path(output).expanduser().resolve()
    if source_pack is not None:
        root=Path(source_pack).expanduser().resolve(strict=True)
        if output==root or root in output.parents:
            raise ValueError("diagnostic output must be outside original source pack")
    if output.exists():raise FileExistsError('output already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pw-vc-diagnostics-',dir=output.parent) as t:
        stage=Path(t)/'report';stage.mkdir()
        (stage/'reviewed_vc_diagnostics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        for name,key in [('reviewed_interval_features.csv','interval_features'),('pairwise_acoustic_shape_distances.csv','within_consonant_pair_distances')]:
            rows=report[key]
            with (stage/name).open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
        (stage/'README.md').write_text('''# PhonoWeave: Six reviewed VC boundaries — exploratory diagnostic only

**Human-completed annotations are not independent adjudication.** A single annotated interval for each of i/ir/i0 → m/sh is not enough to determine a speaker-adaptive split or merge. The /m/ excerpts share a source WAV; /sh/ examples span different WAVs, a confound. The metrics reflect the full *annotated transition* (including vowel and consonant contributions), not isolated phones. ±6ms comparisons are alignment sensitivity only, not bootstrap confidence intervals.

Do NOT train a split classifier on these six observations; do not claim synthesis improvement. Next experiment: independent repeated target recordings in separate sessions + blinded matched-budget resynthesis, then estimate within-category variation and test boundary adjudicator agreement.
''',encoding='utf-8')
        os.rename(stage,output)

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--pack',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        result=compile_probe(a.pack,a.annotations);export_probe(result,a.output,source_pack=a.pack)
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as e:
        p.error(str(e))
    print(json.dumps({'status':result['status'],'reviewed':len(result['interval_features']),
          'pairs':len(result['within_consonant_pair_distances']),
          'automatic_split_share':result['research_decision']['go_for_automatic_split_share']},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
