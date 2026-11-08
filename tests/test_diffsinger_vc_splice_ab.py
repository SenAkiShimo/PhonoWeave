"""Synthetic ZIP regression checks; test outputs are not scientific outcomes."""
from __future__ import annotations

import io
import json
from pathlib import Path
import wave
import zipfile

import numpy as np
import pytest

from phonoweave.diffsinger_vc_splice_ab import (
    LEFTS, POLICIES, RIGHTS, SCHEMA, render_microsplice_ab,
)


def _wav(seed, rate=44100):
    t=np.arange(round(rate*3.0))/rate
    rng=np.random.default_rng(seed)
    x=.10*np.sin((220+15*seed)*2*np.pi*t)+.01*rng.standard_normal(len(t))
    pcm=(np.clip(x,-.8,.8)*32767).astype('<i2').tobytes()
    b=io.BytesIO()
    with wave.open(b,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes(pcm)
    return b.getvalue()


def _bank(tmp_path, style="mang", pitch="D4"):
    path=tmp_path/f'{style}_{pitch}.zip'
    root=f'{style}_{pitch}/'
    records=[]
    for i,left in enumerate(LEFTS):
        records.append(f'vcsection2.wav={pitch}{left} m,{900+350*i},100,900,150,100')
    for i,left in enumerate(LEFTS):
        wav='ingi0ir.wav' if left=='ir' else 'vcsection3.wav'
        records.append(f'{wav}={pitch}{left} sh,{900+350*i},100,900,150,100')
    for right in RIGHTS:
        records.append(f'a.wav={pitch}{right}a,1000,100,900,150,100')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr(root+'oto.ini','\n'.join(records)+'\n')
        for i,name in enumerate(('vcsection2.wav','vcsection3.wav','ingi0ir.wav','a.wav')):
            z.writestr(root+name,_wav(i+1))
    return {"style":style,"pitch":pitch,"archive":str(path),"root":root,"prefix":pitch}


def _manifest(tmp_path):
    return {"schema":"phonoweave.diffsinger_vc_splice_input.v1",
        "source_provenance":"author_confirmed_diffsinger_direct",
        "banks":[_bank(tmp_path)]}


def test_fixed_K2_per_bank_and_audio_is_real(tmp_path):
    data=_manifest(tmp_path)
    output=tmp_path/'rendered'
    report=render_microsplice_ab(data, output)
    assert report['schema']==SCHEMA
    assert len(report['results'])==len(POLICIES)*len(LEFTS)*len(RIGHTS)
    assert report['study_gates']['actual_pcm_wav_rendered'] is True
    assert report['study_gates']['independent_diffsinger_teacher_reference'] is False
    assert report['study_gates']['evidence_of_optimal_split_share'] is False
    assert all(r['source_donor']['phonetic_boundary_verified'] is False for r in report['results'])
    rows=[r for r in report['results'] if r['policy'].startswith('k2_')]
    assert {r['total_vc_budget_units_per_bank'] for r in rows}=={4}
    assert {r['total_vc_budget_ms_per_bank'] for r in rows}=={480}
    preserved=[r for r in report['results'] if r['policy'].startswith('k3_')]
    assert {r['total_vc_budget_ms_per_bank'] for r in preserved}=={720}
    for r in rows:
        path=output/r['rendered_wav']
        assert path.exists()
        with wave.open(str(path)) as w:
            assert w.getnchannels()==1 and w.getframerate()==44100
            assert w.getnframes()>=12000
            assert max(abs(a) for a in np.frombuffer(w.readframes(w.getnframes()),dtype='<i2'))>100
    assert len(set(r['wav_sha256'] for r in rows))==6  # 3 physical donor choices per right


def test_no_overwrite_and_source_unmodified(tmp_path):
    data=_manifest(tmp_path)
    src=Path(data['banks'][0]['archive'])
    orig=src.read_bytes()
    output=tmp_path/'rendered'
    render_microsplice_ab(data, output)
    with pytest.raises(FileExistsError):render_microsplice_ab(data,output)
    assert src.read_bytes()==orig


def test_provenance_required(tmp_path):
    data=_manifest(tmp_path)
    data['source_provenance']='unknown'
    with pytest.raises(ValueError,match='provenance'):render_microsplice_ab(data,tmp_path/'bad')


def test_reject_missing_target_alias(tmp_path):
    data=_manifest(tmp_path)
    archive=Path(data['banks'][0]['archive'])
    with zipfile.ZipFile(archive) as z:
        items={n:z.read(n) for n in z.namelist()}
    oto=data['banks'][0]['root']+'oto.ini'
    items[oto]=items[oto].replace(b'D4i0 sh,',b'D4xx sh,')
    with zipfile.ZipFile(archive,'w') as z:
        for name,content in items.items():z.writestr(name,content)
    with pytest.raises(ValueError,match='missing/duplicate OTO'):render_microsplice_ab(data,tmp_path/'bad')


def test_no_duplicate_training_banks(tmp_path):
    data=_manifest(tmp_path)
    data['banks'].append(dict(data['banks'][0]))
    with pytest.raises(ValueError,match='duplicate style/pitch'):render_microsplice_ab(data,tmp_path/'bad')


def test_K2_policy_no_third_donor():
    for key,mapping in POLICIES.items():
        expected=2 if key.startswith('k2_') else 3
        assert set(mapping)==set(LEFTS)
        assert set(mapping.values())<=set(LEFTS)
        assert len(set(mapping.values()))==expected
