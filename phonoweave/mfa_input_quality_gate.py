#!/usr/bin/env python3
"""Read-only, model-independent alignment diagnostics for PhonoWeave v0.3.

No TextGrid from this test is ground truth. A provisional pass is *not* a
measured phone-boundary accuracy claim, and local human editing remains needed.
"""
import argparse,csv,hashlib,json,math,re,wave,zipfile,os
from pathlib import Path

ROOT=Path(os.environ.get('PHONOWEAVE_MFA_PACK_DIR', Path(__file__).resolve().parent.parent)).resolve()
TONE_CHARS='˥˩˧˨˦'
PHONE_EXPECTED={
 'a':['a'],
 'ma':['m','a'],
 'rong':['{ɻ,ʐ}','u','ŋ'],
 'li':['l','i'],
 'shi':['ʂ','ʐ̩'],
 'sha':['ʂ','a'],
 'zi':['ts','z̩'],
}
# Mandarin MFA's acoustic/phones.txt lists all symbols used here. ʐ̩ and z̩
# are syllabic nuclei; never force these labels into guessed /i/ tokens.

def sha(data):return hashlib.sha256(data).hexdigest()

def check_sources(manifest):
    failures=[]
    for arm in ('corpus_hanzi','corpus_toned_pinyin'):
        for it in manifest['files']:
            wav=ROOT/arm/(it['id']+'.wav')
            if not wav.exists() or sha(wav.read_bytes())!=it['audio_sha256']:
                failures.append(str(wav))
    if failures:
        raise ValueError('WAV hash mismatch/missing in materialized pack: '+', '.join(failures))

def read_textgrid(path):
    # Parse MFA's long Praat TextGrid output; no external Python deps.
    s=path.read_text(encoding='utf-8-sig')
    sections=re.split(r'(?m)^\s*item \[\d+\]:\s*$',s)
    tiers={}
    for section in sections[1:]:
        name=re.search(r'(?m)^\s*name\s*=\s*"([^"]+)"',section)
        if not name:continue
        intervals=[]
        for m in re.finditer(r'(?ms)^\s*intervals \[\d+\]:\s*\n\s*xmin\s*=\s*([\d.+Ee-]+)\s*\n\s*xmax\s*=\s*([\d.+Ee-]+)\s*\n\s*text\s*=\s*"([^"]*)"',section):
            a,b=float(m.group(1)),float(m.group(2))
            if a>b:raise ValueError('reversed TextGrid interval')
            intervals.append((a,b,m.group(3)))
        tiers[name.group(1).lower()]=intervals
    if 'words' not in tiers or 'phones' not in tiers:raise ValueError('missing words or phones tier')
    return tiers

def normalize_phone(p):
    return p.translate(str.maketrans('','',TONE_CHARS))

def phone_contract(expected,observed):
    if len(expected)!=len(observed):return False
    for exp,obs in zip(expected,observed):
        base=normalize_phone(obs)
        choices=exp[1:-1].split(',') if exp.startswith('{') and exp.endswith('}') else [exp]
        if base not in choices:return False
    return True

def measure_energy_cover(wav_path,intervals):
    # Pure Python; mono PCM16 is required in this private pilot.
    from array import array
    import sys
    with wave.open(str(wav_path),'rb') as f:
        if f.getnchannels()!=1 or f.getsampwidth()!=2:raise ValueError('expected mono PCM16')
        sr=f.getframerate();v=array('h');v.frombytes(f.readframes(f.getnframes()))
        if sys.byteorder!='little':v.byteswap()
    denom=0.;num=0.
    windows=[]
    for a,b,label in intervals:
        if label.strip() and label not in ('sil','spn'):
            windows.append((max(0,int(a*sr)),min(len(v),int(math.ceil(b*sr)))))
    for i,x in enumerate(v):
        e=x*x
        denom+=e
        if any(a<=i<b for a,b in windows):num+=e
    return None if not denom else num/denom

def audit_one(src,labels,grid):
    tiers=read_textgrid(grid)
    words=[(a,b,t) for a,b,t in tiers['words'] if t.strip() and t not in ('sil','spn')]
    phones=[(a,b,t) for a,b,t in tiers['phones'] if t.strip() and t not in ('sil','spn')]
    duration_expected=src['duration_s']
    extent_ok=all(0<=a<=b<=duration_expected+.025 for a,b,_ in words+phones)
    target_syllables=src['syllables']
    mapped=[]
    if len(words)==len(target_syllables):
        for a,b,_ in words:
            # Center-based word association avoids duplicate phones at shared boundary
            mapped.append([t for x,y,t in phones if a-0.001<=(x+y)/2<=b+0.001])
    else:
        mapped=[]
    token_match=(len(mapped)==len(target_syllables) and
       all(phone_contract(PHONE_EXPECTED[syll],p) for syll,p in zip(target_syllables,mapped)))
    coverage=measure_energy_cover(ROOT/'corpus_hanzi'/(src['id']+'.wav'),words)
    cover_ok=coverage is not None and coverage>=.65
    verdict=('candidate_requires_human_boundary_review' if extent_ok and token_match and cover_ok
             else 'reject_as_phone_training_data')
    return {'id':src['id'],'provenance':src['kind'],'arm':labels,'words':len(words),
      'expected_syllables':len(target_syllables),'mfa_phone_sequences':' / '.join(' '.join(x) for x in mapped) if mapped else ' / '.join(t for _,_,t in phones),
      'expected_phone_sequences':' / '.join(' '.join(PHONE_EXPECTED[x]) for x in target_syllables),
      'phone_inventory_contract_pass':token_match,
      'word_energy_coverage':round(coverage,6) if coverage is not None else None,
      'coverage_diagnostic_pass':cover_ok,'interval_extents_pass':extent_ok,
      'verdict':verdict,'note':'Passing is not gold phone boundary accuracy; tone proxies are not author-attested lexical tones'}

def run(arms):
    manifest=json.loads((ROOT/'manifest.json').read_text())
    check_sources(manifest)
    report=[]
    for arm,folder in arms:
        for src in manifest['files']:
            fp=folder/(src['id']+'.TextGrid')
            if not fp.exists():
                report.append({'id':src['id'],'provenance':src['kind'],'arm':arm,'verdict':'missing_TextGrid','note':'No result; do not count as aligned'})
                continue
            try: report.append(audit_one(src,arm,fp))
            except Exception as e: report.append({'id':src['id'],'provenance':src['kind'],'arm':arm,'verdict':'invalid_TextGrid','note':str(e)})
    output=ROOT/'results_audit'
    output.mkdir(exist_ok=True)
    fields=['id','provenance','arm','words','expected_syllables','mfa_phone_sequences','expected_phone_sequences',
     'phone_inventory_contract_pass','word_energy_coverage','coverage_diagnostic_pass','interval_extents_pass','verdict','note']
    with (output/'alignment_qc.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(report)
    summary={'schema':'phonoweave.mfa_input_audit.v0.3','records':len(report),
      'passed_provisional':sum(r['verdict']=='candidate_requires_human_boundary_review' for r in report),
      'rejected':sum(r['verdict']=='reject_as_phone_training_data' for r in report),
      'missing':sum(r['verdict']=='missing_TextGrid' for r in report),
      'invalid':sum(r['verdict']=='invalid_TextGrid' for r in report),
      'gold_phone_boundaries':0,'experimental_efficacy':False,'statuses':report}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('QA:',len(report),'records;',summary['passed_provisional'],'provisional,',summary['rejected'],'rejected,',summary['missing'],'missing. Gold = 0.')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--previous-only',action='store_true')
    a=p.parse_args()
    if a.previous_only:
        run([('previous_bad_input_control',ROOT/'reference_first_run')])
    else:
        run([('hanzi_proxy',ROOT/'textgrids_hanzi'),('toned_pinyin_proxy',ROOT/'textgrids_toned_pinyin')])
