import json
import io
import zipfile
import wave
from pathlib import Path

import numpy as np
import pytest
from phonoweave import user_native_batch as b


def wav_bytes():
    rate=44100
    t=np.arange(int(rate*2.4))/rate
    # Source audio with source-time aligned tone regions.
    audio=0.1*np.sin(2*np.pi*(290+80*(np.sin(2*np.pi*.8*t)>0))*t)
    out=io.BytesIO()
    with wave.open(out,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate)
        w.writeframes((audio*30000).astype('<i2').tobytes())
    return out.getvalue()


def fixture(tmp_path):
    banks=[]
    for style,pitch in [('mang','D4'),('lu','C4')]:
        root=f'{style}{pitch}/'
        archive=tmp_path/f'{style}{pitch}.zip'
        entries=[]
        for a,(left,right) in enumerate((('i0','m'),('ir','m'),('i0','sh'),('ir','sh'))):
            # Anchor occurs at 450ms, 900ms, etc.
            offset=(a+1)*450 - 20
            entries.append(f'voice.wav={pitch}{left} {right},{offset},100,10,25,15')
        entries.append(f'voice.wav={pitch}- a,20,10,10,5,5')
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr(root+'oto.ini','\n'.join(entries)+'\n')
            z.writestr(root+'voice.wav',wav_bytes())
        banks.append(dict(style=style,pitch=pitch,archive=str(archive),root=root,prefix=pitch))
    return {'schema':b.SCHEMA,'language_id':'user-local-script','banks':banks,
            'oto_relative_ms':[-125,-55]}


def test_generic_label_batch_and_no_ipa(tmp_path):
    cfg=fixture(tmp_path)
    r,blocked,banks=b.extract_batch(cfg)
    assert len(r)==8 and not blocked and len(banks)==2
    assert {x['symbol'] for x in r}=={'i0','ir'}
    assert all(x['source_alias'].startswith(x['pitch']) for x in r)
    result=b.evaluate_batch(r,blocked,banks,cfg)
    assert result['counts']['accepted_windows']==8
    assert result['unit_decision']=='NO_AUTO_SPLIT_OR_MERGE'
    assert result['full_phone_boundaries_confirmed']==0
    assert not result['proposed_new_aliases']
    assert result['label_aggregated_clustering_audit']['label_information_used_for_aggregation']
    assert result['evaluation']['source_wav_template']['tested']==0  # same source file template


def test_opaque_unicode_user_symbol(tmp_path):
    cfg=fixture(tmp_path)
    cfg['include_left_symbols']=['i0']
    records,blocked,_=b.extract_batch(cfg)
    assert len(records)==4 and {r['symbol'] for r in records}=={'i0'}
    assert all(r['context'] in ('m','sh') for r in records)


def test_physical_provenance_and_no_overwrite(tmp_path):
    cfg=fixture(tmp_path)
    out=tmp_path/'out'
    first=b.run(cfg,out)
    assert first['counts']['accepted_windows']==8
    assert len((out/'source_index.csv').read_text().splitlines())==9
    assert 'review_queue.csv' in {p.name for p in out.iterdir()}
    with pytest.raises(FileExistsError):b.run(cfg,out)


def test_duplicate_bank_rejected(tmp_path):
    cfg=fixture(tmp_path);cfg['banks'].append(cfg['banks'][0])
    with pytest.raises(ValueError,match='duplicate bank'):b.extract_batch(cfg)


def test_invalid_window_rejected(tmp_path):
    cfg=fixture(tmp_path);cfg['oto_relative_ms']=[200,-55]
    with pytest.raises(ValueError,match='unsafe candidate window'):b.extract_batch(cfg)


def test_missing_source_rejects_only_affected_rows(tmp_path):
    cfg=fixture(tmp_path)
    # add one source-unknown row, should abstain, preserving valid candidates
    src=Path(cfg['banks'][0]['archive'])
    with zipfile.ZipFile(src) as z:
        content={n:z.read(n) for n in z.namelist()}
    root=cfg['banks'][0]['root']
    content[root+'oto.ini']+=b'missing.wav=D4i0 q,1000,100,10,30,10\n'
    with zipfile.ZipFile(src,'w') as z:
        for k,v in content.items():z.writestr(k,v)
    r,blocked,_=b.extract_batch(cfg)
    assert len(r)==8 and len(blocked)==1
    assert blocked[0]['reason']=='missing_source_wav'


def test_two_token_alias_not_phoneme_identity(tmp_path):
    cfg=fixture(tmp_path)
    r,_,_=b.extract_batch(cfg)
    assert all(r['symbol'] in ('ir','i0') for r in r)
    assert all('oto_line' in r and 'source_pcm_sha256' in r for r in r)
