"""Regression tests for user-native temporal features and strict heldout evaluation."""
import csv
import hashlib
import io
import json
from pathlib import Path
import wave
import zipfile

import numpy as np
import pytest

from phonoweave import temporal_context_eval as t


def _fake_source(tmp_path):
    sr=44100
    ts=np.arange(0,2.2,1/sr)
    # Non-stationary tone is not a real phoneme reference.
    audio=(np.sin(2*np.pi*280*ts)*.09).astype(np.float32)
    pcm=np.asarray(audio*32767,dtype='<i2').tobytes()
    buff=io.BytesIO()
    with wave.open(buff,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes(pcm)
    zpath=tmp_path/'private_fake.zip'
    with zipfile.ZipFile(zpath,'w') as z:
        z.writestr('fake/x.wav',buff.getvalue())
        z.writestr('fake/oto.ini','\n'*4+'x.wav=D4声1 m,800,20,20,200,200\n')
    record={'bank':'fake_D4','style':'fake','pitch':'D4','symbol':'声1',
        'context':'m','source_wav':'x.wav','oto_line':'5','source_alias':'D4声1 m',
        'source_zip':'private_fake.zip','source_zip_sha256':t._checksum(zpath),
        'source_pcm_sha256':hashlib.sha256(pcm).hexdigest(),'start_ms':875.0,
        'features_original':([1.]+[0.]*11)}
    manifest={'banks':[{'style':'fake','pitch':'D4','archive':str(zpath),
        'root':'fake/','prefix':'D4'}]}
    return [record],manifest,zpath


def test_single_zip_pass_is_provenance_bound_and_read_only(tmp_path):
    rows,manifest,archive=_fake_source(tmp_path)
    old=t._checksum(archive)
    views,en,v,ok=t.extract_temporal(rows,manifest)
    assert ok.shape==(1,5) and ok.sum()==5
    assert all(a.shape==(1,12) for a in views.values())
    assert t._checksum(archive)==old
    assert all(float(x)>-60 for x in en[0])
    rows[0]['source_pcm_sha256']='0'*64
    with pytest.raises(ValueError,match='source PCM SHA-256 changed'):
        t.extract_temporal(rows,manifest)


def test_original_batch_contract_requires_exact_row_order(tmp_path):
    data=tmp_path/'batch';data.mkdir()
    row={'bank':'b','source_wav':'x.wav','oto_line':'1','symbol':'声1'}
    with (data/'source_index.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    (data/'feature_vectors.json').write_text(json.dumps([{'bank':'b','source_wav':'x.wav',
        'oto_line':2,'features':[0]*12}]),encoding='utf-8')
    with pytest.raises(ValueError,match='provenance mismatch'):t.load_v2(data)
    payload=[{'bank':'b','source_wav':'x.wav','oto_line':1,'features':[0]*12}]
    (data/'feature_vectors.json').write_text(json.dumps(payload),encoding='utf-8')
    assert t.load_v2(data)[0]['features_original']==[0]*12


def test_no_heldout_information_in_bank_center():
    X=np.array([[10.,2.],[12.,4.],[100.,500.],[101.,700.]])
    bank=np.array(['a','a','a','a']);train=np.array([True,True,False,False])
    z=t.center_using_training_only(X,bank,train)
    X2=X.copy();X2[~train]+=10000
    zz=t.center_using_training_only(X2,bank,train)
    np.testing.assert_allclose(z[train],zz[train])
    np.testing.assert_allclose(z[train],np.array([[-1.,-1.],[1.,1.]]))


def test_wholly_unseen_bank_uses_training_global_mean():
    X=np.array([[2.,1.],[4.,1.],[15.,3.]])
    bank=np.array(['a','a','new']);train=np.array([True,True,False])
    z=t.center_using_training_only(X,bank,train)
    np.testing.assert_allclose(z[-1],np.array([12.,2.]))


def test_source_template_disjoint_heldout_and_no_ipa_labels():
    rows=[{'bank':'b','source_wav':f'u{i}.wav','symbol':'声音0'} for i in range(25)]
    g=t.frozen_test_templates(rows)
    assert g==t.frozen_test_templates(rows)
    assert len(g)==8 and not (g&(set(x['source_wav'] for x in rows)-g))
    assert t.SCHEMA.endswith('v3')


def test_selected_mode_uses_only_dev_labels():
    # Partition: 17 dev templates and 8 frozen templates, two arbitrary labels.
    groups=[f'ut{i:02}.wav' for i in range(25)]
    rows=[]
    for group in groups:
        for label in ['ꙗ','i2']:
            rows.append({'source_wav':group,'bank':'b','symbol':label,
                         'style':'style','context':'following'})
    rng=np.random.default_rng(16)
    X={name:rng.normal(size=(len(rows),12)) for name in t.MODES}
    hold=t.frozen_test_templates(rows)
    a=t.dev_selection(X,rows,hold)
    changed=[dict(row) for row in rows]
    for row in changed:
        if row['source_wav'] in hold:row['symbol']='ꙗ' if row['symbol']=='i2' else 'i2'
    b=t.dev_selection(X,changed,hold)
    assert a['selected_mode']==b['selected_mode']
    assert a['dev_scores']==b['dev_scores']
    assert len(a['heldout']['original_70ms']['per_group'])==8


def test_temporary_zero_energy_fallback_never_claims_boundaries():
    rows=[{'features_original':[1.]+[0.]*11}]
    views={k:np.zeros((1,12)) for k in t.SLICES_MS}
    modes,chosen=t.construct_modes(rows,views,np.full((1,5),-120.),
                                    np.full((1,5),1e9),np.zeros((1,5),bool))
    assert modes['stable_region'][0,0]==1.
    assert modes['original_70ms'].shape==(1,12)
    assert len(t.MODES)==7


def test_source_group_exclusion_rejects_leakage():
    X=np.array([[1.],[2.],[3.],[4.]])
    y=np.array(['i0','ir','i0','ir']); groups=np.array(['one','one','two','two'])
    banks=np.array(['bank']*4)
    with pytest.raises(ValueError,match='leakage'):
        t.score_templates(X,y,groups,banks,['one'],['one'])
    assert t.score_templates(X,y,groups,banks,['one'],['two'])['tested']==2


def test_tampered_user_symbol_fails_original_oto_source(tmp_path):
    rows,manifest,_=_fake_source(tmp_path)
    rows[0]['symbol']='new_guess_1'
    with pytest.raises(ValueError,match='source OTO alias/label'):
        t.extract_temporal(rows,manifest)
