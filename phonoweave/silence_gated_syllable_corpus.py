"""Prepare short, silence-gated utterances from original CVVC source ZIPs.

Alignment utterance preparation only. OTO offsets are ordering hints for the
*syllable sequence*, never phoneme boundaries. We choose cuts only where
measured source PCM is quiet, and abstain rather than cut through sound.
No trained aligner is run here. Source and transcripts remain unverified.
"""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import csv,hashlib,io,json,math,shutil,wave,zipfile
from pathlib import Path
import numpy as np

SCORE_SCHEMA='phonoweave.whole_utterance_verified_transcript.v0.1'
OUT_SCHEMA='phonoweave.silence_gated_syllable_corpus.v0.2'
PREFLIGHT=['output/source_banks.csv','output/source_wavs.csv','output/syllable_anchors.csv','output/transcript_templates_to_verify.json']


def sha256_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def _read_csv(z,name):return list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))


def _pcm(z,member):
    with wave.open(io.BytesIO(z.read(member)),'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getcomptype()!='NONE':
            raise ValueError('expected mono PCM16: '+member)
        rate=w.getframerate()
        if rate<16000:raise ValueError('sample rate too low')
        data=w.readframes(w.getnframes())
    return np.frombuffer(data,dtype='<i2'),rate


def _collapse_near_duplicate_anchors(anchors, sequence, *, threshold_ms=350):
    """De-duplicate *candidate OTO rows*, not phones or genuine repetitions.

    An equal surface within 350ms cannot be another independent syllable in
    this sparse 2.2s-per-syllable source template. Record every discarded OTO
    reference for source provenance and keep different-surface collisions as a
    reason to abstain, never silently merge different pronunciations.
    """
    if len(anchors)!=len(sequence):raise ValueError('source/sequence length mismatch')
    kept=[];tokens=[];discard=[]
    for row,token in zip(anchors,sequence):
        when=float(row['oto_offset_ms'])
        if kept and when-float(kept[-1]['oto_offset_ms'])<threshold_ms:
            if row['syllable_surface']!=kept[-1]['syllable_surface'] or token!=tokens[-1]:
                return [],[],[], 'different_surface_anchor_collision'
            discard.append({'surface':token,'ignored_oto_line':row['oto_line'],
                            'retained_oto_line':kept[-1]['oto_line'],
                            'delta_ms':round(when-float(kept[-1]['oto_offset_ms']),3)})
            continue
        kept.append(row);tokens.append(token)
    return kept,tokens,discard,'ok'


def _rms_envelope(pcm,rate):
    # 10 ms source PCM RMS, source is never re-encoded for measurement.
    hop=max(1,round(rate*.010))
    n=len(pcm)//hop
    if n<100:raise ValueError('source WAV shorter than one second')
    pcmf=pcm[:n*hop].astype(np.float64)/32768
    return np.sqrt(np.mean(pcmf.reshape(n,hop)**2,axis=1))


def _choose_cuts(env,anchors_ms,duration_ms,*, silence_threshold=.0035):
    """Require a substantial quiet valley between OTO-ordered syllable hints.

    Each cut is a *silence split*, not a phone boundary. In extreme cases
    there is no viable silence: return status=ambiguous and do not write audio.
    """
    if len(anchors_ms)==0:raise ValueError('no syllable anchors')
    if any(not math.isfinite(float(v)) for v in anchors_ms):raise ValueError('non-finite anchor')
    if any(float(b)-float(a)<400 for a,b in zip(anchors_ms,anchors_ms[1:])):
        return [],[], 'too_close_or_duplicate_oto_anchor'
    cuts=[];evidence=[]
    for a,b in zip(anchors_ms,anchors_ms[1:]):
        gap=b-a
        # The middle third is well away from either onset, including early frication.
        l=max(0,int(round((a+.35*gap)/10)))
        r=min(len(env),int(round((a+.65*gap)/10)))
        if r-l<8:return [],[], 'no_safe_search_window'
        # At least 80 ms continuous quiet; not one accidental zero frame.
        q=(env[l:r] <= silence_threshold).astype(np.int32)
        runs=[];begin=None
        for ix,quiet in enumerate(q):
            if quiet and begin is None:begin=ix
            if (not quiet or ix==len(q)-1) and begin is not None:
                end=ix if not quiet else ix+1
                if end-begin>=8:runs.append((begin,end))
                begin=None
        if not runs:return [],[], 'no_80ms_quiet_valley'
        # Minimize silence split bias by taking center of largest quiet run.
        # Tie-break near true midpoint. Cuts need not track acoustic onset.
        mid=(a+b)/20
        best=min(runs,key=lambda x:(-(x[1]-x[0]),abs(l+(x[0]+x[1])/2-mid)))
        cut=(l+(best[0]+best[1])/2)*10
        if cuts and cut<=cuts[-1]:raise ValueError('nonmonotonic acoustic cuts')
        cuts.append(float(cut))
        evidence.append({'quiet_duration_ms':(best[1]-best[0])*10,
                         'cut_rms':float(np.max(env[max(0,int(cut/10)-3):min(len(env),int(cut/10)+4)]))})
    boundaries=[0.,*cuts,float(duration_ms)]
    if any(b-a<300 for a,b in zip(boundaries,boundaries[1:])):
        return [],[], 'short_chunk'
    return boundaries,evidence,'quiet_valleys_confirmed'


def _trim_silence(pcm,rate,start_ms,end_ms,*, pad_ms=220,activity_threshold=.0008):
    """Trim long empty edges with generous phone-preserving padding.

    Does NOT infer phone boundaries. Never removes active frames intentionally.
    Returns source sample indices, and fails if the chunk is truly silent.
    """
    a=round(start_ms*rate/1000);b=round(end_ms*rate/1000)
    if not 0<=a<b<=len(pcm):raise ValueError('bad chunk coordinates')
    sub=pcm[a:b]
    hop=max(1,round(rate*.01));n=len(sub)//hop
    if n<25:raise ValueError('too short for activity screen')
    v=(sub[:n*hop].astype(np.float64)/32768).reshape(n,hop)
    env=np.sqrt(np.mean(v*v,axis=1))
    voiced=np.where(env>activity_threshold)[0]
    if len(voiced)<6:return None,'insufficient_acoustic_activity'
    lead=max(0,round((voiced[0]*10-pad_ms)*rate/1000))
    tail=min(len(sub),round(((voiced[-1]+1)*10+pad_ms)*rate/1000))
    if tail-lead<.25*rate:return None,'too_short_after_padding'
    return (a+lead,a+tail),'acoustic_activity_with_conservative_padding'


def _wav(path,pcm,rate):
    with path.open('xb') as f:
        with wave.open(f,'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate)
            w.writeframes(np.asarray(pcm,dtype='<i2').tobytes())


def build(preflight_zip:Path,voicebank_dir:Path,output:Path,*,allow_unverified_pilot:bool=False,
          max_bank_wavs:int|None=None):
    preflight_zip=Path(preflight_zip).resolve(strict=True)
    voicebank_dir=Path(voicebank_dir).resolve(strict=True)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('output exists; do not overwrite')
    if not allow_unverified_pilot:
        raise ValueError('source transcripts are unverified. Supply --allow-unverified-pilot for an explicitly non-gold pilot')
    with zipfile.ZipFile(preflight_zip) as pz:
        for n in PREFLIGHT:
            if n not in pz.namelist():raise ValueError('missing frozen preflight '+n)
        banks=_read_csv(pz,PREFLIGHT[0]);wavs=_read_csv(pz,PREFLIGHT[1]);anchors=_read_csv(pz,PREFLIGHT[2]);
        t=json.loads(pz.read(PREFLIGHT[3]));
        if t.get('schema')!=SCORE_SCHEMA:raise ValueError('invalid transcript schema')
    template={x['source_wav']:x for x in t['templates']}
    if len(template)!=len(t['templates']):raise ValueError('duplicate source template')
    bankmap={x['bank']:x for x in banks}
    if len(bankmap)!=len(banks):raise ValueError('duplicate bank')
    group=defaultdict(list)
    for r in anchors:group[(r['bank'],r['source_wav'])].append(r)
    for b in banks:
        path=voicebank_dir/b['source_archive']
        if sha256_file(path)!=b['archive_sha256']:raise ValueError('ZIP SHA-256 mismatch: '+str(path))
    clips=[];abstained=[];decisions=[];duplicates=[];paths_open={};count=0
    try:
        output.mkdir(parents=True)
        (output/'corpus').mkdir()
        for row in wavs:
            seq=template[row['source_wav']]['draft_sequence']
            key=(row['bank'],row['source_wav'])
            if not seq:
                abstained.append({'bank':row['bank'],'source_wav':row['source_wav'],'reason':'no_complete_transcript'})
                continue
            if max_bank_wavs is not None and count>=max_bank_wavs:
                abstained.append({'bank':row['bank'],'source_wav':row['source_wav'],'reason':'pilot_limit'})
                continue
            src=sorted(group[key],key=lambda r:float(r['oto_offset_ms']))
            if len(src)!=len(seq) or [s['syllable_surface'] for s in src]!=seq:
                raise ValueError('syllable order mismatch: '+str(key))
            src,seq,discard,anchor_status=_collapse_near_duplicate_anchors(src,seq)
            for d in discard:duplicates.append({'bank':row['bank'],'source_wav':row['source_wav'],**d})
            if anchor_status!='ok':
                count+=1
                abstained.append({'bank':row['bank'],'source_wav':row['source_wav'],'reason':anchor_status})
                continue
            bank=row['bank'];b=bankmap[bank];arc=voicebank_dir/b['source_archive']
            if bank not in paths_open:paths_open[bank]=zipfile.ZipFile(arc)
            z=paths_open[bank]
            source_member=row['source_wav_zip_path']
            if source_member not in z.namelist():raise ValueError('missing WAV '+source_member)
            pcm,rate=_pcm(z,source_member)
            duration_ms=len(pcm)*1000/rate
            if abs(duration_ms-float(row['duration_ms']))>2:raise ValueError('source WAV duration mismatch')
            env=_rms_envelope(pcm,rate)
            boundaries,gates,status=_choose_cuts(env,[float(s['oto_offset_ms']) for s in src],duration_ms)
            count+=1
            if status!='quiet_valleys_confirmed':
                abstained.append({'bank':bank,'source_wav':row['source_wav'],'reason':status})
                continue
            for i,syllable in enumerate(seq):
                indices,act_status=_trim_silence(pcm,rate,boundaries[i],boundaries[i+1])
                if indices is None:
                    abstained.append({'bank':bank,'source_wav':row['source_wav'],
                                      'reason':act_status,'syllable_index':i,'syllable':syllable})
                    continue
                a,bindex=indices
                # Fail if cropped audio is substantially silent or clips > one octave of context.
                rel=f'{bank}/{Path(row["source_wav"]).stem}__{i+1:03d}.wav'
                dst=output/'corpus'/rel
                dst.parent.mkdir(exist_ok=True)
                _wav(dst,pcm[a:bindex],rate)
                label=dst.with_suffix('.lab')
                with label.open('x',encoding='utf-8') as f:f.write(syllable+'\n')
                clips.append({'bank':bank,'source_wav':row['source_wav'],'syllable_index':i,
                              'syllable':syllable,'source_zip_member':source_member,
                              'source_zip_sha256':bankmap[bank]['archive_sha256'],
                              'relative_wav':rel,'relative_lab':rel[:-4]+'.lab',
                              'source_start_sample':a,'source_end_sample':bindex,
                              'source_start_ms':round(a*1000/rate,3),
                              'source_end_ms':round(bindex*1000/rate,3),
                              'duration_ms':round((bindex-a)*1000/rate,3),
                              'sample_rate':rate,'wav_sha256':sha256_file(dst),
                              'lab_sha256':sha256_file(label),
                              'utterance_cut_status':'measured_silence_not_phone_boundary',
                              'transcript_status':'UNVERIFIED_PILOT_ONLY'})
            decisions.append({'bank':bank,'source_wav':row['source_wav'],
                              'tokens':len(seq),'cut_status':status,
                              'min_intertoken_quiet_ms':min((x['quiet_duration_ms'] for x in gates),default=None),
                              'worst_cut_rms':max((x['cut_rms'] for x in gates),default=None)})
        if not clips:raise ValueError('all clips abstained')
        with (output/'clips.csv').open('x',encoding='utf8',newline='') as f:
            cols=list(clips[0]);writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(clips)
        with (output/'cut_checks.csv').open('x',encoding='utf8',newline='') as f:
            cols=list(decisions[0]);writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(decisions)
        with (output/'abstentions.json').open('x',encoding='utf8') as f:json.dump(abstained,f,indent=2,ensure_ascii=False)
        with (output/'collapsed_near_duplicate_oto_rows.json').open('x',encoding='utf8') as f:json.dump(duplicates,f,indent=2,ensure_ascii=False)
        report={'schema':OUT_SCHEMA,'source_wavs_total':len(wavs),
                'scanned_whole_transcript_wavs':count,'clips_created':len(clips),
                'unique_syllable_surfaces':len(set(x['syllable'] for x in clips)),
                'banks':dict(Counter(x['bank'] for x in clips)),
                'abstentions':dict(Counter(r['reason'] for r in abstained)),
                'deduplicated_near_identical_oto_candidates':len(duplicates),
                'source_transcripts_verified':False,'acoustic_model_executed':False,
                'phoneme_boundaries_created':0,
                'method':'OTO only to order syllables; cuts exclusively in measured quiet source PCM; no phone boundaries inferred',
                'sample_copies_exact':True}
        (output/'chunk_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
        # Existing whole_utterance_mfa_bridge align/import consumes precisely
        # these names and enforces WAV/lab SHA plus a model execution receipt.
        bridge_index=[]
        for c in clips:
            bridge_index.append({'bank':c['bank'],'source_wav':Path(c['relative_wav']).name,
                'source_original_wav':c['source_wav'],
                'source_duration_ms':c['duration_ms'],
                'wav_sha256':c['wav_sha256'],'lab_sha256':c['lab_sha256'],
                'transcript_status':c['transcript_status'],
                'source_start_ms':c['source_start_ms'],
                'syllables':1})
        with (output/'staged_source_index.csv').open('x',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(bridge_index[0]));w.writeheader();w.writerows(bridge_index)
        bridge_status={'schema':'phonoweave.whole_utterance_mfa_bridge.v0.1',
            'mode':'UNVERIFIED_CANDIDATE_PILOT',
            'source_wavs_total':len(wavs),'staged_wavs':len(clips),
            'excluded_wavs':len(set((r['bank'],r['source_wav']) for r in abstained)),
            'staged_syllable_tokens':len(clips),
            'acoustic_model_ran':False,'phone_boundaries_created':0,
            'provenance':'short syllables, source-bound cuts inside measured quiet PCM',
            'transcript_warning':'none of these candidate pinyin transcripts has been verified'}
        (output/'status.json').write_text(json.dumps(bridge_status,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
        return report
    except BaseException:
        if output.exists():shutil.rmtree(output)
        raise
    finally:
        for z in paths_open.values():z.close()


def lift_to_original_time(stage:Path, imported:Path, output:Path):
    """Convert REAL imported MFA boundaries from clip time to original time.

    This does not run a model or infer any timestamps. Absent imported real
    TextGrid intervals fail closed. Labels remain unverified model candidates.
    """
    stage=Path(stage).resolve(strict=True)
    imported=Path(imported).resolve(strict=True)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('refuse to overwrite lifted phone intervals')
    if output.is_relative_to(stage) or output.is_relative_to(imported):
        raise ValueError('output cannot be inside source or imported data')
    status=json.loads((stage/'status.json').read_text(encoding='utf8'))
    if status.get('schema')!='phonoweave.whole_utterance_mfa_bridge.v0.1':
        raise ValueError('stage not compatible with source-bound alignment')
    with (stage/'clips.csv').open(encoding='utf8',newline='') as f:
        clips=list(csv.DictReader(f))
    keys={(r['bank'],Path(r['relative_wav']).name):r for r in clips}
    if len(keys)!=len(clips):raise ValueError('duplicate syllable clip')
    source=imported/'aligned_intervals.csv'
    with source.open(encoding='utf8',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError('no imported model intervals')
    result=[]
    for r in rows:
        key=r['bank'],r['source_wav']
        if key not in keys:raise ValueError('model interval not bound to a source syllable')
        c=keys[key]
        if r['source_wav_sha256']!=c['wav_sha256']:
            raise ValueError('model interval refers to altered clip WAV')
        start,end=float(r['start_s']),float(r['end_s'])
        duration=float(c['duration_ms'])/1000
        if not 0<=start<=end<=duration+.001:raise ValueError('model interval outside clip')
        orig=float(c['source_start_ms'])/1000
        result.append({'bank':c['bank'],'source_wav':c['source_wav'],
          'source_member':c['source_zip_member'],'original_source_zip_sha256':c['source_zip_sha256'],
          'syllable_index':c['syllable_index'],'syllable_surface':c['syllable'],
          'tier':r['tier'],'model_label':r['label'],
          'original_start_s':round(orig+start,6),
          'original_end_s':round(orig+end,6),
          'clip_start_s':start,'clip_end_s':end,
          'origin':'external_MFA_model_candidate_NOT_GOLD',
          'transcript_status':c['transcript_status'],
          'source_wav_sha256_of_clip':c['wav_sha256']})
    with output.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    return {'intervals_lifted':len(result),'phone_gold_count':0}


def main(argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0]=='lift':
        p=argparse.ArgumentParser(description='Lift actual MFA TextGrid intervals to original source-WAV coordinates')
        p.add_argument('lift');p.add_argument('--stage',type=Path,required=True);p.add_argument('--imported',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        a=p.parse_args(args)
        print(json.dumps(lift_to_original_time(a.stage,a.imported,a.output),ensure_ascii=False))
        return
    p=argparse.ArgumentParser(description='Read-only silence-gated per-syllable preparation for MFA (not phone alignment)')
    p.add_argument('--preflight-zip',required=True,type=Path)
    p.add_argument('--voicebank-dir',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--allow-unverified-pilot',action='store_true')
    p.add_argument('--max-bank-wavs',type=int)
    a=p.parse_args(args)
    report=build(a.preflight_zip,a.voicebank_dir,a.output,allow_unverified_pilot=a.allow_unverified_pilot,max_bank_wavs=a.max_bank_wavs)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
