"""Synthetic regression checks; NOT evidence of real phoneme alignment."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import pytest
from phonoweave import silence_gated_syllable_corpus as seg


def test_quiet_valley_finds_cut_not_oto_phone_boundary():
    env=np.zeros(600)
    env[45:130]=.1
    env[265:345]=.1
    env[460:545]=.1
    cuts,proof,status=seg._choose_cuts(env,[750.,2900.,4850.],6000.)
    assert status=='quiet_valleys_confirmed'
    assert len(cuts)==4 and len(proof)==2
    assert all(x['quiet_duration_ms']>=80 for x in proof)
    assert cuts[1] != 750 and cuts[2] != 2900


def test_refuse_split_when_there_is_no_80ms_silence():
    env=np.full(500,.08)
    cuts,proof,status=seg._choose_cuts(env,[1000.,3000.],5000.)
    assert not cuts and not proof and status=='no_80ms_quiet_valley'


def test_duplicate_same_surface_is_not_an_extra_syllable():
    a=[dict(oto_offset_ms=str(v),oto_line=str(i),syllable_surface='rong') for i,v in enumerate((22000,22023),start=1)]
    keep,tokens,drop,status=seg._collapse_near_duplicate_anchors(a,['rong','rong'])
    assert status=='ok' and tokens==['rong'] and len(drop)==1
    assert drop[0]['delta_ms']==23


def test_different_surface_collision_fails_closed():
    a=[dict(oto_offset_ms=str(v),oto_line=str(i),syllable_surface=t) for i,(v,t) in enumerate(((22000,'rong'),(22023,'dong')),start=1)]
    keep,tokens,drop,status=seg._collapse_near_duplicate_anchors(a,['rong','dong'])
    assert status=='different_surface_anchor_collision' and not keep


def test_pcm_crop_is_sample_exact_and_contains_fricative_lead_in():
    rate=16000
    pcm=np.zeros(int(rate*2),dtype='<i2')
    pcm[round(.6*rate):round(1.35*rate)]=4200
    loc,status=seg._trim_silence(pcm,rate,0,2000)
    assert status=='acoustic_activity_with_conservative_padding'
    start,end=loc
    assert start<round(.6*rate) and end>round(1.35*rate)
    assert np.array_equal(pcm[start:end],pcm[start:end].copy())


def test_short_model_interval_lift_requires_actual_import_data(tmp_path):
    stage=tmp_path/'stage';stage.mkdir()
    (stage/'status.json').write_text(json.dumps({'schema':'phonoweave.whole_utterance_mfa_bridge.v0.1'}))
    imported=tmp_path/'imported';imported.mkdir()
    with pytest.raises(FileNotFoundError):
        seg.lift_to_original_time(stage,imported,tmp_path/'output.csv')


def test_correct_source_coordinates_preserved_in_lift(tmp_path):
    stage=tmp_path/'stage';stage.mkdir()
    (stage/'status.json').write_text(json.dumps({'schema':'phonoweave.whole_utterance_mfa_bridge.v0.1'}))
    clip={'bank':'mang_D4','relative_wav':'mang_D4/a__009.wav',
          'source_wav':'a.wav','source_zip_member':'LynneQingD4/a.wav',
          'source_zip_sha256':'abc','source_start_ms':'12000','source_end_ms':'14100',
          'duration_ms':'2100','syllable_index':'8','syllable':'ma',
          'wav_sha256':'def','transcript_status':'UNVERIFIED_PILOT_ONLY'}
    with (stage/'clips.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(clip));w.writeheader();w.writerow(clip)
    imported=tmp_path/'imported';imported.mkdir()
    entry={'bank':'mang_D4','source_wav':'a__009.wav','tier':'phones',
           'label':'m','start_s':'0.45','end_s':'0.57','source_wav_sha256':'def'}
    with (imported/'aligned_intervals.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(entry));w.writeheader();w.writerow(entry)
    target=tmp_path/'lifted.csv'
    result=seg.lift_to_original_time(stage,imported,target)
    assert result=={'intervals_lifted':1,'phone_gold_count':0}
    row=list(csv.DictReader(target.open()))[0]
    assert row['source_wav']=='a.wav' and row['original_start_s']=='12.45'
    assert row['original_end_s']=='12.57'
    with pytest.raises(FileExistsError):seg.lift_to_original_time(stage,imported,target)


def test_corrupt_lift_source_hash_rejected(tmp_path):
    stage=tmp_path/'stage';stage.mkdir()
    (stage/'status.json').write_text(json.dumps({'schema':'phonoweave.whole_utterance_mfa_bridge.v0.1'}))
    clip={'bank':'mang_D4','relative_wav':'mang_D4/a__001.wav',
          'source_wav':'a.wav','source_zip_member':'LynneQingD4/a.wav',
          'source_zip_sha256':'source','source_start_ms':'0',
          'duration_ms':'2000','syllable_index':'0','syllable':'a',
          'wav_sha256':'real','transcript_status':'UNVERIFIED_PILOT_ONLY'}
    with (stage/'clips.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(clip));w.writeheader();w.writerow(clip)
    imported=tmp_path/'imported';imported.mkdir()
    entry={'bank':'mang_D4','source_wav':'a__001.wav','tier':'phones',
           'label':'a','start_s':'0','end_s':'1.5','source_wav_sha256':'notreal'}
    with (imported/'aligned_intervals.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(entry));w.writeheader();w.writerow(entry)
    with pytest.raises(ValueError,match='altered'):seg.lift_to_original_time(stage,imported,tmp_path/'out.csv')
