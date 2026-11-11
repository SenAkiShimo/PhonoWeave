from __future__ import annotations
import copy
import io
import json
import sys
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

from phonoweave.orthography_native_units import (
    acoustic_vector, _parse_oto, _read_wav, extract, evaluate,
    unsupervised_cluster_audit, main, SCHEMA,
)


def _wav(pcm: np.ndarray, sr: int = 44100) -> bytes:
    f = io.BytesIO()
    with wave.open(f, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.asarray(np.clip(pcm,-1,1)*32767,dtype='<i2').tobytes())
    return f.getvalue()


def _fixture(tmp_path):
    banks=[]
    labels=['q','q0','q2']
    for style,pitch in [('s','G3'),('s','D4'),('s','D5')]:
        name=tmp_path/f'{style}_{pitch}.zip';root=f'{style}_{pitch}/';prefix=pitch
        lines=[]
        with zipfile.ZipFile(name,'w',compression=zipfile.ZIP_DEFLATED) as z:
            sr=44100
            for ci,ctx in enumerate(['p','t']):
                x=np.zeros(sr*5)
                for ix,label in enumerate(labels):
                    # All 3 synthetic candidates in one source WAV.
                    offset=750+ix*1100
                    anchor=offset+110
                    start=round((anchor-200)/1000*sr)
                    end=round((anchor+140)/1000*sr)
                    t=np.arange(end-start)/sr
                    freq=[210,350,580][ix]
                    x[start:end]=.2*(np.sin(2*np.pi*freq*t)+.25*np.sin(2*np.pi*2*freq*t))
                    lines.append(f'{ctx}.wav={prefix}{label} {ctx},{offset},120,400,140,80')
                z.writestr(root+f'{ctx}.wav',_wav(x))
            z.writestr(root+'oto.ini','\n'.join(lines).encode())
        banks.append({'style':style,'pitch':pitch,'archive':str(name),'root':root,'prefix':prefix})
    return {'schema':SCHEMA,'language_id':'imaginary-tongue','user_base_symbol':'q',
            'user_realizations':labels,'contexts':['p','t'],
            'source_wavs_by_context':{'p':'p.wav','t':'t.wav'},
            'oto_relative_ms':[-125,-55], 'banks':banks}


def test_real_pcm_audio_extraction_and_user_identifiers(tmp_path):
    cfg=_fixture(tmp_path)
    rows,abstain=extract(cfg)
    assert not abstain
    assert len(rows)==3*3*2
    assert {x['symbol'] for x in rows}=={'q','q0','q2'}
    assert all(x['original_alias'].startswith(x['pitch']) for x in rows)
    assert all(x['source_pcm_sha256'] and x['source_zip_sha256'] for x in rows)
    assert all(x['candidate_status']=='unverified_oto_relative_acoustic_window' for x in rows)
    e=evaluate(rows,cfg)
    assert e['recommended_new_user_aliases']==[]
    assert e['unit_decision']=='insufficient_independent_sources_for_auto_alias_change'
    assert e['cross_context_top1']['total']==18
    assert e['unsupervised_cluster_audit']['status']=='exploratory_acoustic_clusters_only'


def test_untouched_source_and_no_output_overwrite(tmp_path):
    cfg=_fixture(tmp_path)
    manifest=tmp_path/'config.json';manifest.write_text(json.dumps(cfg))
    out=tmp_path/'results'
    assert main(['--manifest',str(manifest),'--output',str(out)])==0
    assert (out/'evidence.json').is_file()
    assert (out/'source_samples.json').is_file()
    assert (out/'sample_index.csv').is_file()
    with pytest.raises(ValueError,match='overwrite'):
        main(['--manifest',str(manifest),'--output',str(out)])


def test_invalid_manifest_and_duplicate_pitches_fail_closed(tmp_path):
    cfg=_fixture(tmp_path)
    bad=copy.deepcopy(cfg);bad['banks'][1]=bad['banks'][0]
    with pytest.raises(ValueError,match='duplicate subbank'):
        extract(bad)
    bad=copy.deepcopy(cfg);bad['user_realizations']=['q','q']
    with pytest.raises(ValueError,match='unique'):
        extract(bad)
    bad=copy.deepcopy(cfg);bad['source_wavs_by_context']['p']='../evil.wav'
    with pytest.raises(ValueError,match='unsafe'):
        extract(bad)
    bad=copy.deepcopy(cfg);bad['oto_relative_ms']=[-500,200]
    with pytest.raises(ValueError,match='window'):
        extract(bad)


def test_silence_abstains_and_oto_duplicates_are_not_silent_success(tmp_path):
    cfg=_fixture(tmp_path)
    a=Path(cfg['banks'][0]['archive'])
    with zipfile.ZipFile(a,'r') as z:
        files={n:z.read(n) for n in z.namelist()}
    p=cfg['banks'][0]['root']
    files[p+'p.wav']=_wav(np.zeros(44100*5))
    files[p+'oto.ini']+=b'\np.wav=G3q p,750,120,400,140,80'
    with zipfile.ZipFile(a,'w') as z:
        for n,b in files.items():z.writestr(n,b)
    rows,abstain=extract(cfg)
    assert len(rows)<18
    assert any(a['reason']=='missing_or_ambiguous_oto' for a in abstain)
    assert any(a['reason']=='insufficient_voicing' for a in abstain)


def test_opaque_script_labels_no_ipa_requirement(tmp_path):
    cfg=_fixture(tmp_path)
    cfg['user_realizations']=['我','X_0','a²']
    # Identifiers do not have IPA syntax requirements; they do need source rows.
    assert all(isinstance(x,str) for x in cfg['user_realizations'])
    cfg['user_realizations']=['q','q0','q2']
    rows,abstain=extract(cfg)
    assert len(rows)==18
    assert not abstain


def test_acoustic_vector_silent_abstains():
    assert acoustic_vector(np.zeros(5000),44100) is None
    assert acoustic_vector(np.ones(100),44100) is None


def test_oto_splitting_preserves_author_alias():
    d=_parse_oto(b'p.wav=D4q2 p,750,120,400,140,80','D4')
    assert d[('p.wav','q2 p')][0][1]=='D4q2 p'
    assert d[('p.wav','q2 p')][0][2][0]==750


def test_unsupervised_cluster_uses_bank_id_without_mutating_labels():
    cfg={'user_realizations':['a','a1','a2']}
    rng=np.random.default_rng(8)
    records=[]
    for bi in range(3):
        for context in ('p','t'):
            for idx,label in enumerate(cfg['user_realizations']):
                vec=np.array([float(idx==j) for j in range(3)]) + float(bi) + rng.normal(0,.01,3)
                records.append({'bank':f'b{bi}','symbol':label,'features':vec.tolist()})
    assert unsupervised_cluster_audit(records,cfg['user_realizations'])['posthoc_count_correct']==18


def test_eval_rejects_duplicate_same_context():
    r={'bank':'s_G3','style':'s','pitch':'G3','symbol':'x','context':'p','source_wav':'x.wav','oto_line':1,'features':[.1]*12}
    with pytest.raises(ValueError,match='duplicate'):
        evaluate([r,r],{'user_realizations':['x','y'],'contexts':['p','t'],'language_id':'test','user_base_symbol':'x'})


def test_single_user_label_can_request_unlabeled_subtype_search(tmp_path):
    cfg=_fixture(tmp_path)
    cfg['user_realizations']=['q']
    records,abstentions=extract(cfg)
    assert len(records)==6 and not abstentions
    result=evaluate(records,cfg)
    assert result['single_label_exploratory_mode']
    assert result['cross_context_top1']['total']==0
    assert result['recommended_new_user_aliases']==[]


def test_unicode_author_aliases_are_opaque_and_preserved(tmp_path):
    cfg=_fixture(tmp_path)
    rename={'q':'字','q0':'i0','q2':'a²'}
    for b in cfg['banks']:
        zpath=Path(b['archive'])
        with zipfile.ZipFile(zpath) as z:
            content={n:z.read(n) for n in z.namelist()}
        root=b['root']
        old=content[root+'oto.ini'].decode('utf-8')
        for old_symbol,new_symbol in rename.items():
            old=old.replace(f"{b['prefix']}{old_symbol} ",f"{b['prefix']}{new_symbol} ")
        content[root+'oto.ini']=old.encode('utf-8')
        with zipfile.ZipFile(zpath,'w') as z:
            for name,data in content.items():z.writestr(name,data)
    cfg['user_realizations']=list(rename.values())
    records,abstain=extract(cfg)
    assert len(records)==18 and not abstain
    assert {r['symbol'] for r in records}=={'字','i0','a²'}
    assert all(r['original_alias'].startswith(r['pitch']) for r in records)


def test_explicit_observations_separate_user_identity_from_oto_spelling(tmp_path):
    cfg=_fixture(tmp_path)
    symbols=['自己','0_i','核²']
    src=['q','q0','q2']
    cfg['user_realizations']=symbols
    cfg['observations']=[{'user_symbol':symbols[i],'context_id':ctx,
                          'source_wav':f'{ctx}.wav','oto_alias_body':f'{src[i]} {ctx}'}
                         for ctx in ['p','t'] for i in range(3)]
    cfg.pop('source_wavs_by_context')
    rows,abstain=extract(cfg)
    assert not abstain and len(rows)==18
    assert {r['symbol'] for r in rows}==set(symbols)
    assert all(r['original_alias'].startswith(r['pitch']+'q') for r in rows)
    assert not any(r['original_alias'].startswith(r['pitch']+'自己') for r in rows)


def test_repeated_user_realization_in_distinct_source_is_not_collapsed(tmp_path):
    cfg=_fixture(tmp_path)
    cfg['banks']=cfg['banks'][:1]
    bank=cfg['banks'][0]
    path=Path(bank['archive'])
    with zipfile.ZipFile(path) as z:
        content={n:z.read(n) for n in z.namelist()}
    root=bank['root']
    content[root+'p_repeat.wav']=content[root+'p.wav']
    content[root+'oto.ini']+=f"\np_repeat.wav={bank['prefix']}q p,750,120,400,140,80".encode()
    with zipfile.ZipFile(path,'w') as z:
        for n,data in content.items():z.writestr(n,data)
    cfg['observations']=[{'user_symbol':label,'context_id':ctx,
                          'source_wav':f'{ctx}.wav','oto_alias_body':f'{label} {ctx}'}
                         for ctx in ('p','t') for label in cfg['user_realizations']]
    cfg['observations'].append({'user_symbol':'q','context_id':'p',
                                'source_wav':'p_repeat.wav','oto_alias_body':'q p'})
    rows,abstain=extract(cfg)
    assert not abstain and len(rows)==7
    assert len({(row['bank'],row['source_wav'],row['oto_line']) for row in rows})==7
    assert evaluate(rows,cfg)['candidate_windows']==7
