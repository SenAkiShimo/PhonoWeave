"""Source-safe, source-linked full-recording corpus builder for a Mandarin MFA baseline.

A draft alias-based syllable sequence is NOT a verified transcript. This module
never emits acoustic phone boundaries without an actual MFA TextGrid. The 56
VC-only source WAVs without a full transcript are deliberately quarantined.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import wave
import zipfile

SCHEMA = 'phonoweave.whole_utterance_mfa_bridge.v0.1'
TRANSCRIPT_SCHEMA = 'phonoweave.whole_utterance_verified_transcript.v0.1'
SUPPORTED_VERIFICATIONS = ('verified_against_source_score', 'verified_by_author')
SYLLABLE = re.compile(r'^[a-züê]+$')


def digest_file(path: Path) -> str:
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _safe(text: str) -> str:
    if not isinstance(text,str) or not text or text.startswith(('.', '/')) or '/' in text or '\\' in text or '..' in text:
        raise ValueError('unsafe basename')
    return text


def _csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)


def _load_index(preflight:Path, transcript_templates:Path):
    root=preflight.resolve(strict=True)
    wavs=list(csv.DictReader((root/'source_wavs.csv').open(encoding='utf-8',newline='')))
    banks=list(csv.DictReader((root/'source_banks.csv').open(encoding='utf-8',newline='')))
    templates=json.loads(transcript_templates.read_text(encoding='utf-8'))
    if templates.get('schema')!=TRANSCRIPT_SCHEMA:
        raise ValueError('unexpected transcript template schema')
    rows=templates.get('templates')
    if not isinstance(rows,list) or len(rows)!=len({r['source_wav'] for r in wavs}):
        raise ValueError('transcript template coverage mismatch')
    tmap={row['source_wav']:row for row in rows}
    if len(tmap)!=len(rows):raise ValueError('duplicate templates')
    bmap={r['bank']:r for r in banks}
    if len(bmap)!=len(banks) or len({(w['bank'],w['source_wav']) for w in wavs})!=len(wavs):
        raise ValueError('source inventory has duplicate bank or WAV')
    for row in rows:
        _safe(row['source_wav'])
        if set(row.get('banks',[]))!={r['bank'] for r in wavs if r['source_wav']==row['source_wav']}:
            raise ValueError('template does not match bank source inventory')
        if row.get('verification','not_verified') not in (*SUPPORTED_VERIFICATIONS,'not_verified'):
            raise ValueError('unrecognized verification status')
        sequence=row.get('draft_sequence')
        if not isinstance(sequence,list) or any(not isinstance(x,str) or not SYLLABLE.fullmatch(x) for x in sequence):
            raise ValueError('invalid syllable tokens')
    return wavs,bmap,tmap


def _verified(row:dict) -> bool:
    if row['verification'] not in SUPPORTED_VERIFICATIONS:return False
    if not row.get('draft_sequence'):return False
    evidence=row.get('verification_evidence')
    return isinstance(evidence,str) and len(evidence.strip())>=8


def build_corpus(preflight: Path, templates_path: Path, manifest_path: Path,
                 destination: Path, *, allow_unverified_pilot: bool=False,
                 max_wavs: int|None=None) -> dict:
    wavs,bankmap,templates=_load_index(preflight,templates_path)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    banks={b['style']+'_'+b['pitch']:b for b in manifest.get('banks',[])}
    if set(banks)!=set(bankmap):raise ValueError('source bank identities differ from preflight')
    if max_wavs is not None and max_wavs<=0:raise ValueError('max_wavs must be positive')
    dest=destination.expanduser().resolve()
    if dest.exists():raise FileExistsError('refuse overwrite of existing output')
    if any(dest.is_relative_to(Path(b['archive']).resolve().parent) and
           dest == Path(b['archive']).resolve() for b in banks.values()):
        raise ValueError('source-overwrite destination')
    # Verify private ZIPs against frozen preflight before reading any waveform.
    for bank,desc in banks.items():
        arc=Path(desc['archive']).resolve(strict=True)
        if digest_file(arc)!=bankmap[bank]['archive_sha256']:
            raise ValueError('source archive hash mismatch: '+bank)
    staging=[];excluded=[]
    selected=0;opened={}
    try:
        # No external model or downloaded code is involved in this stage.
        for row in wavs:
            bank=row['bank'];name=_safe(row['source_wav']); t=templates[name]
            if not t['draft_sequence']:
                excluded.append(dict(bank=bank,source_wav=name,reason='no_whole_utterance_transcription',
                                     suggestion='obtain original synthesis score or independently label the full recording'))
                continue
            approved=_verified(t)
            if not approved and not allow_unverified_pilot:
                excluded.append(dict(bank=bank,source_wav=name,reason='transcript_not_verified',
                                     suggestion='verify against original score, or use explicit --allow-unverified-pilot for diagnostics'))
                continue
            if max_wavs is not None and selected>=max_wavs:
                excluded.append(dict(bank=bank,source_wav=name,reason='pilot_max_wavs_limit',suggestion='rerun without limit'))
                continue
            desc=banks[bank];zpath=desc['root']+name
            if bank not in opened:opened[bank]=zipfile.ZipFile(desc['archive'],'r')
            z=opened[bank]
            if zpath not in z.namelist():raise ValueError('source WAV absent from archive')
            if row['source_wav_zip_path']!=zpath or row['ota_oto_sha256']!=bankmap[bank]['oto_sha256']:
                raise ValueError('wrong original source path or OTO hash')
            # This uses the input WAV unchanged. No trimming or normalization.
            staging.append((bank,name,zpath,desc['archive'],t,row,approved))
            selected+=1
        # All source checks happen before output is made. A failed extraction
        # removes the incomplete destination, preserving source input paths.
        dest.mkdir(parents=True,exist_ok=False)
        (dest/'corpus').mkdir()
        saved=[]
        for bank,name,zpath,arc,t,row,approved in staging:
            folder=dest/'corpus'/bank
            folder.mkdir(exist_ok=True)
            audio=folder/name
            with audio.open('xb') as out, opened[bank].open(zpath) as inp:
                shutil.copyfileobj(inp,out,1024*1024)
            with wave.open(str(audio),'rb') as w:
                if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getcomptype()!='NONE':
                    raise ValueError('source not mono PCM16')
                if abs(w.getnframes()*1000/w.getframerate()-float(row['duration_ms']))>1.0:
                    raise ValueError('source duration differs from preflight')
            lab=audio.with_suffix('.lab')
            with lab.open('x',encoding='utf-8') as f:
                f.write(' '.join(t['draft_sequence'])+'\n')
            saved.append(dict(bank=bank,source_wav=name,source_zip_basename=Path(arc).name,
                              source_zip_member=zpath,wav_sha256=digest_file(audio),
                              lab_sha256=digest_file(lab),syllables=len(t['draft_sequence']),
                              transcript_status='source_score_or_author_verified' if approved else 'UNVERIFIED_PILOT_ONLY',
                              eligible_for_speech_accuracy_claim=approved,
                              source_rate_hz=row['sample_rate'],source_duration_ms=row['duration_ms']))
        _csv(dest/'staged_source_index.csv',saved, ['bank','source_wav','source_zip_basename','source_zip_member','wav_sha256','lab_sha256','syllables','transcript_status','eligible_for_speech_accuracy_claim','source_rate_hz','source_duration_ms'])
        _csv(dest/'excluded_source_wavs.csv',excluded,['bank','source_wav','reason','suggestion'])
        report=dict(schema=SCHEMA,mode='UNVERIFIED_CANDIDATE_PILOT' if allow_unverified_pilot else 'SOURCE_VERIFIED_ONLY',
                    archive_integrity_verified=True,source_wavs_total=len(wavs),staged_wavs=len(saved),
                    excluded_wavs=len(excluded),staged_syllable_tokens=sum(x['syllables'] for x in saved),
                    excluded_reasons=dict(Counter(e['reason'] for e in excluded)),
                    acoustic_model_ran=False,phone_boundaries_created=0,
                    note='Lab files hold Pinyin syllable surfaces; MFA must provide its own compatible pronunciation model. These are not phoneme labels.',
                    model_caveat='Mandarin MFA trained on speech, accuracy for DiffSinger singing is unknown.')
        (dest/'status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return report
    except BaseException:
        if dest.exists():shutil.rmtree(dest)
        raise
    finally:
        for z in opened.values():z.close()


def doctor() -> dict:
    binary=shutil.which('mfa')
    if binary is None:
        return dict(mfa_installed=False,model_available=False,
                    blocker='MFA CLI and Mandarin model not installed in current environment',
                    recommended_install='conda install -c conda-forge montreal-forced-aligner; mfa model download acoustic mandarin_mfa')
    p=subprocess.run([binary,'--version'],capture_output=True,text=True,timeout=30)
    return dict(mfa_installed=p.returncode==0,mfa_version=(p.stdout or p.stderr).strip()[:140],
                model_available='not_verified',
                hint='MFA 3.4+: mfa align_hf corpus mandarin_mfa output --use_g2p --language unknown; check mfa --help for installed version')


def prepare_run_command(corpus:Path,aligned:Path, *, mode='hf') -> list[str]:
    if mode=='hf':return ['mfa','align_hf',str(corpus),'mandarin_mfa',str(aligned),'--use_g2p','--language','unknown']
    if mode=='legacy':return ['mfa','align',str(corpus),'mandarin_china_mfa','mandarin_mfa',str(aligned)]
    raise ValueError('unsupported MFA runner')


def run_mfa(stage:Path,output:Path,*, mode='hf') -> dict:
    stage=stage.resolve(strict=True)
    status=json.loads((stage/'status.json').read_text(encoding='utf-8'))
    if status.get('schema')!=SCHEMA or status['staged_wavs']==0:
        raise ValueError('no staged WAVs or incompatible prepared corpus')
    output=output.resolve()
    if output.exists() or output.is_relative_to(stage):raise FileExistsError('non-overwrite output required')
    binary=shutil.which('mfa')
    if not binary:raise RuntimeError('MFA missing: install the engine and Mandarin model; cannot fabricate phone timestamps')
    cmd=prepare_run_command(stage/'corpus',output,mode=mode)
    p=subprocess.run(cmd,check=False,text=True,capture_output=True)
    if p.returncode!=0:raise RuntimeError('MFA failed: '+(p.stderr or p.stdout)[-2000:])
    if not output.is_dir() or not list(output.rglob('*.TextGrid')):
        raise RuntimeError('MFA command exited successfully but produced no TextGrid; no phone labels recorded')
    receipt=dict(schema='phonoweave.mfa_runtime_receipt.v0.1',
                 model_executed=True,model_identifier='mandarin_mfa',
                 full_command=cmd,version=subprocess.run([binary,'--version'],capture_output=True,text=True).stdout.strip(),
                 frozen_stage_index_sha256=digest_file(stage/'staged_source_index.csv'),
                 textgrids_found=len(list(output.rglob('*.TextGrid'))),
                 model_alignment_is_not_human_ground_truth=True)
    with (output/'_phonoweave_mfa_run_receipt.json').open('x',encoding='utf-8') as f:
        json.dump(receipt,f,ensure_ascii=False,indent=2)
        f.write('\n')
    return receipt


def _parse_mfa_long_textgrid(path:Path) -> list[dict]:
    """Parse the common Praat long TextGrid IntervalTier without mistaking tier headers for phones."""
    text=path.read_text(encoding='utf-8-sig')
    parts=re.split(r'(?=\n\s*item \[\d+\]:)',text)
    result=[]
    for part in parts:
        klass=re.search(r'class\s*=\s*"([^"]+)"',part)
        name=re.search(r'name\s*=\s*"([^"]+)"',part)
        if not klass or klass.group(1)!='IntervalTier' or not name:continue
        tier=name.group(1)
        for m in re.finditer(r'intervals \[\d+\]:\s*xmin\s*=\s*([\d.eE+-]+)\s*xmax\s*=\s*([\d.eE+-]+)\s*text\s*=\s*"((?:[^"]|"")*)"',part):
            start,end=float(m.group(1)),float(m.group(2))
            if not 0<=start<=end:raise ValueError('invalid aligned interval')
            result.append(dict(tier=tier,start_s=start,end_s=end,label=m.group(3).replace('""','"')))
    if not result:raise ValueError('no aligned Praat IntervalTier intervals')
    return result


def import_textgrids(stage:Path,grids:Path,out:Path) -> dict:
    stage=stage.resolve(strict=True);grids=grids.resolve(strict=True);out=out.resolve()
    if out.exists() or out.is_relative_to(stage) or out.is_relative_to(grids):
        raise FileExistsError('refuse to overwrite staged or aligned data')
    meta=json.loads((stage/'status.json').read_text(encoding='utf-8'))
    if meta.get('schema')!=SCHEMA:raise ValueError('invalid stage')
    proof=grids/'_phonoweave_mfa_run_receipt.json'
    if not proof.is_file():
        raise ValueError('missing successful model-execution receipt; will not present unverified TextGrids as MFA phone alignment')
    receipt=json.loads(proof.read_text(encoding='utf-8'))
    if (receipt.get('schema')!='phonoweave.mfa_runtime_receipt.v0.1'
        or receipt.get('model_executed') is not True
        or receipt.get('model_identifier')!='mandarin_mfa'
        or receipt.get('frozen_stage_index_sha256')!=digest_file(stage/'staged_source_index.csv')):
        raise ValueError('MFA receipt has incompatible or stale source identity')
    index=list(csv.DictReader((stage/'staged_source_index.csv').open(encoding='utf-8',newline='')))
    allrows=[];missing=[];quality=[]
    for r in index:
        wav=stage/'corpus'/r['bank']/r['source_wav']
        lab=wav.with_suffix('.lab')
        if digest_file(wav)!=r['wav_sha256'] or digest_file(lab)!=r['lab_sha256']:
            raise ValueError('source or transcript changed after freeze')
        grid=grids/r['bank']/(Path(r['source_wav']).stem+'.TextGrid')
        if not grid.is_file():
            missing.append(dict(bank=r['bank'],wav=r['source_wav']))
            continue
        data=_parse_mfa_long_textgrid(grid)
        dur=float(r['source_duration_ms'])/1000
        for d in data:
            if d['end_s']>dur+.1:raise ValueError('TextGrid exceeds original WAV duration')
            allrows.append(dict(bank=r['bank'],source_wav=r['source_wav'],**d,
                                source_wav_sha256=r['wav_sha256'],
                                origin='MFA_acoustic_alignment_not_gold',
                                transcript_status=r['transcript_status'],
                                phone_boundaries_human_verified=False))
        phones=[d for d in data if d['tier'].lower() in ('phone','phones') and d['label'].strip()]
        words=[d for d in data if d['tier'].lower() in ('word','words') and d['label'].strip()]
        reference_syllables=lab.read_text(encoding='utf-8').split()
        issues=[]
        if not phones:issues.append('no_nonempty_phone_tier')
        if not words:issues.append('no_nonempty_word_tier')
        if len(words)!=len(reference_syllables):issues.append('word_count_differs_from_source_transcript')
        for tier in set(d['tier'] for d in data):
            intervals=[d for d in data if d['tier']==tier]
            if any(next_d['start_s']+1e-5<d['end_s'] for d,next_d in zip(intervals,intervals[1:])):
                issues.append('overlapping_intervals_in_'+tier)
        quality.append(dict(bank=r['bank'],source_wav=r['source_wav'],
                            transcript_syllables=len(reference_syllables),aligned_word_intervals=len(words),
                            aligned_phone_intervals=len(phones),
                            transcript_verified_before_alignment=r['transcript_status']=='source_score_or_author_verified',
                            issues='|'.join(issues),
                            result='needs_review' if issues or r['transcript_status']!='source_score_or_author_verified'
                                   else 'candidate_alignment_pending_gold_accuracy_test'))
    out.mkdir(parents=True,exist_ok=False)
    _csv(out/'aligned_intervals.csv',allrows,['bank','source_wav','tier','start_s','end_s','label','source_wav_sha256','origin','transcript_status','phone_boundaries_human_verified'])
    _csv(out/'alignment_quality_gate.csv',quality,['bank','source_wav','transcript_syllables','aligned_word_intervals','aligned_phone_intervals','transcript_verified_before_alignment','issues','result'])
    summary=dict(schema='phonoweave.imported_acoustic_intervals.v0.1',staged_wavs=len(index),
                 wavs_with_textgrids=len(index)-len(missing),missing_textgrids=len(missing),
                 all_tier_intervals=len(allrows),phone_intervals=sum(r['tier'].lower() in ('phones','phone') and bool(r['label'].strip()) for r in allrows),
                 gold_boundaries=0,alignment_accuracy_measured=False,
                 candidate_alignments_without_quality_issues=sum(not q['issues'] for q in quality),
                 rejected_or_review_required_count=sum(q['result']=='needs_review' for q in quality),
                 note='Model-generated candidate phone boundaries; independent annotated reference necessary to claim error tolerance.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description='Whole-utterance Mandarin MFA bridge, no false gold alignment')
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--preflight',type=Path,required=True);prep.add_argument('--templates',type=Path,required=True)
    prep.add_argument('--manifest',type=Path,required=True);prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--allow-unverified-pilot',action='store_true');prep.add_argument('--max-wavs',type=int)
    sub.add_parser('doctor')
    run=sub.add_parser('align');run.add_argument('--stage',type=Path,required=True);run.add_argument('--output',type=Path,required=True);run.add_argument('--mode',choices=('hf','legacy'),default='hf')
    im=sub.add_parser('import');im.add_argument('--stage',type=Path,required=True);im.add_argument('--textgrids',type=Path,required=True);im.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='prepare':x=build_corpus(args.preflight,args.templates,args.manifest,args.output,allow_unverified_pilot=args.allow_unverified_pilot,max_wavs=args.max_wavs)
    elif args.command=='doctor':x=doctor()
    elif args.command=='align':x=run_mfa(args.stage,args.output,mode=args.mode)
    else:x=import_textgrids(args.stage,args.textgrids,args.output)
    print(json.dumps(x,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
