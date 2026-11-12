"""Synthetic waveforms prove evaluation plumbing, never scientific efficacy."""
from __future__ import annotations
import csv
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from phonoweave.vc36_capture import write_plan,ingest_session,apply_boundary_csv
from phonoweave.vc36_heldout_probe_v09 import evaluate,export_evaluation,SCHEMA


def _wav(path,freq,rate=48000):
    path.parent.mkdir(parents=True,exist_ok=True)
    t=np.arange(2*rate,dtype=np.float64)/rate
    # Distinct source PCM and predictable per-condition frequencies.
    samples=(0.12*np.sin(2*np.pi*freq*t)*32767).astype('<i2')
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate)
        f.writeframes(samples.tobytes())


def _prepared(tmp_path):
    root=tmp_path/'study'
    manifest=root/'manifest.json'
    plan=write_plan(manifest,speaker_id='test_singer')
    for sid,day in [('S01','2026-11-10'),('S02','2026-11-11')]:
        folder=tmp_path/('source_'+sid)
        for row in plan['recordings']:
            if row['session_id']!=sid:continue
            left_idx=('i','ir','i0').index(row['left_label'])
            right_idx=('m','sh').index(row['right_label'])
            freq=320+left_idx*170+right_idx*65+(row['repeat']-2)*2
            if sid=='S02':freq+=0.6
            _wav(folder/(row['take_id']+'.wav'),freq)
        ingest_session(manifest,folder,session_id=sid,capture_date=day,
                       device_id='mic',room_id='room',gain_setting='fixed',
                       distance_cm=18,setup_reset=True)
    data=json.loads(manifest.read_text())
    ann=tmp_path/'boundaries.csv'
    cols=['take_id','source_wav_sha256','transition_onset_ms',
          'transition_offset_ms','boundary_reviewer','qc_manual_reviewed']
    with ann.open('w',newline='',encoding='utf-8') as file:
        w=csv.DictWriter(file,fieldnames=cols);w.writeheader()
        for row in data['recordings']:
            w.writerow({'take_id':row['take_id'],'source_wav_sha256':row['wav_sha256'],
                        'transition_onset_ms':850,'transition_offset_ms':1150,
                        'boundary_reviewer':'human','qc_manual_reviewed':'true'})
    apply_boundary_csv(manifest,ann)
    return root,manifest


def test_holdout_vs_pool_uses_the_same_s01_recordings(tmp_path):
    root,manifest=_prepared(tmp_path)
    r=evaluate(manifest)
    assert r['schema']==SCHEMA
    assert r['train_session']=='S01' and r['heldout_session']=='S02'
    assert r['training_unique_wavs']==18 and r['heldout_unique_wavs']==18
    assert len(r['per_take'])==18 and len(r['per_condition'])==6
    assert all(row['take_id'].startswith('S02_') for row in r['per_take'])
    assert r['same_training_material_both_estimators'] is True
    assert r['mean_pooled_minus_separate_mse']>0  # Synthetic class tones only.
    assert r['research_decision']['automatic_split_or_merge_approved'] is False
    assert r['research_decision']['real_synthesis_gain_established'] is False
    target=tmp_path/'results'
    export_evaluation(r,target,manifest_path=manifest)
    saved=json.loads((target/'heldout_acoustic_proxy.json').read_text())
    assert saved['source_manifest_sha256']==r['source_manifest_sha256']
    assert len(list(csv.DictReader((target/'heldout_per_take.csv').open())))==18
    with pytest.raises(FileExistsError):
        export_evaluation(r,target,manifest_path=manifest)
    with pytest.raises(ValueError,match='outside'):
        export_evaluation(r,root/'results',manifest_path=manifest)


def test_missing_manual_boundaries_refuses_analysis(tmp_path):
    root=tmp_path/'study';manifest=root/'manifest.json'
    write_plan(manifest,speaker_id='test_singer')
    with pytest.raises(ValueError,match='not analysis_ready'):
        evaluate(manifest)


def test_modified_source_refuses_analysis(tmp_path):
    root,manifest=_prepared(tmp_path)
    data=json.loads(manifest.read_text())
    source=root/data['recordings'][0]['relative_wav']
    source.write_bytes(b'forged')
    with pytest.raises(ValueError,match='not analysis_ready'):
        evaluate(manifest)


def test_short_review_interval_refuses_even_if_capture_audit_ready(tmp_path):
    root,manifest=_prepared(tmp_path)
    data=json.loads(manifest.read_text())
    data['recordings'][0]['transition_offset_ms']=870  # only 20ms
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='below 40ms'):
        evaluate(manifest)
