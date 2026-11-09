"""Synthetic regression tests: NOT empirical DiffSinger reconstruction results."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from phonoweave.diffsinger_teacher_benchmark import (
    _canonical_hash, _sha_file, prepare_intake, audit_and_compare,
    REPORT_SCHEMA, SCHEMA, EXPECTED_POLICIES, HIGH_BUDGET,
)


def _write_wav(path, *, freq=220, phase=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    sr=44100
    t=np.arange(sr,dtype=float)/sr
    samples=(.1*np.sin(2*np.pi*freq*t+phase)+.035*np.sin(2*np.pi*2*freq*t+phase))
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr)
        f.writeframes((samples*32767).astype('<i2').tobytes())


def _fixture(tmp_path):
    root=tmp_path/'renders'
    root.mkdir()
    frozen={'schema':'test_frozen_score','score_id':'fixture_test','phrases':[
      {'id':'phrase_a'},{'id':'phrase_b'}]}
    score_sha=_canonical_hash(frozen)
    banks=('mang_D4','mang_G3','mang_G4','mang_D5','lu_C4','lu_F4','lu_B4')
    arm_order=(*EXPECTED_POLICIES,HIGH_BUDGET)
    renders=[];ledger=[];midis=[]
    for k,bank in enumerate(banks):
        for phrase in ('phrase_a','phrase_b'):
            midi=f'score_midi/{bank}_{phrase}.mid'
            (root/midi).parent.mkdir(parents=True,exist_ok=True)
            (root/midi).write_bytes(f'{bank}:{phrase}'.encode())
            midis.append(midi)
            for j,policy in enumerate(arm_order):
                rel=f'{bank}/{policy}/{phrase}.wav'
                p=root/rel
                _write_wav(p,freq=200+5*k+7*j,phase=(1 if phrase=='phrase_a' else 2)*.2)
                renders.append({'bank':bank,'score_phrase':phrase,'policy':policy,
                   'score_sha256':score_sha,'rendered_wav':rel,'wav_sha256':_sha_file(p),
                   'test_target_was_independently_generated':False,
                   'teacher_quality_scored':False})
        for policy in arm_order:
            ledger.append({'bank':bank,'policy':policy,'total_deduplicated_source_ms':
                           2380 if policy==HIGH_BUDGET else 2140})
    manifest={'schema':'phonoweave.diffsinger_note_phrase_ab.v1',
      'experiment_gates':{'independent_DiffSinger_teacher_reference_present':False},
      'frozen_score':frozen,'score_sha256':score_sha,
      'renders':renders,'budget_ledger':ledger,'score_midi_files':midis}
    (root/'render_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    intake_file=tmp_path/'intake.json'
    intake=prepare_intake(root,intake_file)
    return root,intake_file,intake


def _provide_one(tmp_path, root, receipt_path, *, use_existing=False):
    receipt=json.loads(receipt_path.read_text())
    row=receipt['teacher_reference_slots'][0]
    if use_existing:
        render=json.loads((root/'render_manifest.json').read_text())['renders'][0]
        source=(root/render['rendered_wav']).read_bytes()
        dest=tmp_path/row['relative_wav'];dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(source)
    else:
        dest=tmp_path/row['relative_wav'];_write_wav(dest,freq=333,phase=.5)
    row.update(wav_sha256=_sha_file(dest),
      model_or_checkpoint_identifier='synthetic_test_model_not_real',
      new_generation_from_frozen_score=True,source_wav_reused_or_spliced=False,
      style_verified_by_operator=True,note_and_lyric_sequence_verified=True,
      onset_alignment_verified=True,settings_or_project_reference='synthetic-test')
    receipt_path.write_text(json.dumps(receipt),encoding='utf-8')
    return dest


def test_prepare_requires_all_valid_56_preview_files_and_midi(tmp_path):
    root, receipt, intake=_fixture(tmp_path)
    assert intake['schema']==SCHEMA
    assert len(intake['teacher_reference_slots'])==14
    assert all(x['wav_sha256'] is None for x in intake['teacher_reference_slots'])
    with pytest.raises(FileExistsError):prepare_intake(root,receipt)
    example=root/'mang_D4/k2_keep_i_ir/phrase_a.wav'
    example.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='integrity mismatch'):
        prepare_intake(root,tmp_path/'second.json')


def test_no_teacher_audio_means_zero_quality_scores(tmp_path):
    root,receipt,_=_fixture(tmp_path)
    report=audit_and_compare(root,receipt)
    assert report['schema']==REPORT_SCHEMA
    assert report['status']=='blocked_no_independent_reference_comparisons'
    assert report['reference_verified_slots']==0
    assert report['complete_teacher_set'] is False
    assert report['diagnostics']==[]
    assert not report['split_share_efficacy_established']
    assert len(report['blocked'])==14


def test_declared_new_synthetic_reference_produces_only_equal_budget_diagnostics(tmp_path):
    root,receipt,_=_fixture(tmp_path)
    _provide_one(tmp_path,root,receipt)
    report=audit_and_compare(root,receipt)
    assert report['reference_verified_slots']==1
    assert report['complete_teacher_set'] is False
    assert len(report['diagnostics'])==3
    assert {r['policy'] for r in report['diagnostics']}==set(EXPECTED_POLICIES)
    assert all(r['source_budget_ms']==2140 for r in report['diagnostics'])
    assert all(r['spectral_log10_shape_rmse']>=0 for r in report['diagnostics'])
    assert all(r['active_teacher_frames']>=40 for r in report['diagnostics'])
    assert not report['K3_in_equal_budget_comparisons']
    assert not report['split_share_efficacy_established']


def test_teacher_cannot_be_existing_k2_preview(tmp_path):
    root,receipt,_=_fixture(tmp_path)
    _provide_one(tmp_path,root,receipt,use_existing=True)
    report=audit_and_compare(root,receipt)
    assert report['reference_verified_slots']==0
    assert 'teacher_is_existing_concatenative_preview' in [r['reason'] for r in report['blocked']]


def test_no_empty_or_fake_attestation(tmp_path):
    root,receipt,_=_fixture(tmp_path)
    _provide_one(tmp_path,root,receipt)
    d=json.loads(receipt.read_text());d['teacher_reference_slots'][0]['model_or_checkpoint_identifier']=''
    receipt.write_text(json.dumps(d))
    report=audit_and_compare(root,receipt)
    assert report['reference_verified_slots']==0
    assert report['blocked'][0]['reason']=='fresh_render_identity_or_alignment_unattested'


def test_nonmatching_score_or_render_hash_rejected(tmp_path):
    root,receipt,_=_fixture(tmp_path)
    d=json.loads(receipt.read_text());d['score_sha256']='0'*64
    receipt.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='not bound'):
        audit_and_compare(root,receipt)


def test_path_and_slot_substitution_rejected(tmp_path):
    root,receipt,_=_fixture(tmp_path)
    d=json.loads(receipt.read_text());d['teacher_reference_slots'][0]['relative_wav']='../../etc/shadow'
    receipt.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='altered'):
        audit_and_compare(root,receipt)
    d['teacher_reference_slots'][0]['relative_wav']=f"teacher/{d['teacher_reference_slots'][0]['bank']}/{d['teacher_reference_slots'][0]['phrase']}.wav"
    d['teacher_reference_slots'][1]=deepcopy(d['teacher_reference_slots'][0]);receipt.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='duplicate'):
        audit_and_compare(root,receipt)


def test_teacher_file_hash_mismatch_blocks_comparison(tmp_path):
    root,receipt,_=_fixture(tmp_path)
    teacher=_provide_one(tmp_path,root,receipt)
    with teacher.open('ab') as f:f.write(b'corruption')
    report=audit_and_compare(root,receipt)
    assert report['reference_verified_slots']==0
    assert report['blocked'][0]['reason']=='teacher_hash_missing_or_mismatch'


def test_hash_scan_does_not_auto_attest_source(tmp_path):
    from phonoweave.diffsinger_teacher_benchmark import scan_new_teacher_wavs
    root,receipt,_=_fixture(tmp_path)
    d=json.loads(receipt.read_text())
    target=tmp_path/d['teacher_reference_slots'][0]['relative_wav']
    _write_wav(target, freq=350)
    scanned_file=tmp_path/'scanned.json'
    result=scan_new_teacher_wavs(receipt,tmp_path,scanned_file)
    assert result['scanned_wav_slots']==1
    assert result['teacher_reference_slots'][0]['wav_sha256']==_sha_file(target)
    assert result['teacher_reference_slots'][0]['new_generation_from_frozen_score'] is False
    assert result['teacher_reference_slots'][0]['model_or_checkpoint_identifier'] is None
    check=audit_and_compare(root,scanned_file)
    assert check['reference_verified_slots']==0
    assert check['blocked'][0]['reason']=='fresh_render_identity_or_alignment_unattested'

def _write_release_tail(path, *, extra_ms=110, decays=True):
    """1 s reference plus source-independent final release."""
    sr=44100
    extra=round(extra_ms*sr/1000)
    t=np.arange(sr+extra,dtype=float)/sr
    samples=.08*np.sin(2*np.pi*333*t+.5)
    if decays:
        samples[sr:]*=np.linspace(1,0,extra)**2
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr)
        f.writeframes((samples*32767).astype('<i2').tobytes())


def test_trailing_release_does_not_force_avoidable_re_render(tmp_path):
    root,receipt,_=_fixture(tmp_path)
    teacher=_provide_one(tmp_path,root,receipt)
    _write_release_tail(teacher,extra_ms=110,decays=True)
    data=json.loads(receipt.read_text())
    data['teacher_reference_slots'][0]['wav_sha256']=_sha_file(teacher)
    receipt.write_text(json.dumps(data))
    result=audit_and_compare(root,receipt)
    assert result['reference_verified_slots']==1
    assert len(result['diagnostics'])==3
    assert all(x['source_budget_ms']==2140 for x in result['diagnostics'])
    assert not result['split_share_efficacy_established']


def test_sustained_extra_note_or_long_overhang_not_accepted(tmp_path):
    root,receipt,_=_fixture(tmp_path)
    teacher=_provide_one(tmp_path,root,receipt)
    data=json.loads(receipt.read_text())
    for extra_ms,decays in ((110,False),(160,True)):
        _write_release_tail(teacher,extra_ms=extra_ms,decays=decays)
        data['teacher_reference_slots'][0]['wav_sha256']=_sha_file(teacher)
        receipt.write_text(json.dumps(data))
        report=audit_and_compare(root,receipt)
        assert report['reference_verified_slots']==0
        assert report['blocked'][0]['reason']=='teacher_score_duration_mismatch_align_externally'
