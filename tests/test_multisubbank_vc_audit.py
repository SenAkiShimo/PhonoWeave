"""Synthetic software checks; do not imply any human-speech results."""
from __future__ import annotations
import copy
from pathlib import Path
import io
import json
import wave
import zipfile
import numpy as np
import pytest

from phonoweave.multisubbank_vc_audit import (
    SCHEMA, TARGETS, D4_REVIEWED_INTERVALS,
    parse_oto, build_audit, write_reports, sample_metrics,
)


def _synth_wav(seed: int, *, identical=False):
    sr=44100
    t=np.arange(3*sr)/sr
    data=0.09*np.sin(2*np.pi*(220+seed*5)*t)
    for st in [1.07, 1.72]:
        a,b=round((st+.01)*sr),round((st+.16)*sr)
        noise=np.random.default_rng(seed+int(st*100)).normal(0,.06,b-a)
        data[a:b]=noise
    binary=(np.clip(data,-.9,.9)*32767).astype('<i2').tobytes()
    buf=io.BytesIO()
    with wave.open(buf,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr)
        w.writeframes(binary)
    return buf.getvalue()


def _bank(tmp_path, pitch, *, seed=0, bad=None):
    prefix=pitch;root=f'Bank{pitch}/';arc=tmp_path/f'{pitch}.zip'
    # All six targets; same alignment for both pitches, distinct waveforms.
    targets=[('i','m','vcsection2.wav',1000),('ir','m','vcsection2.wav',1600),
             ('i0','m','vcsection2.wav',2050),('i','sh','vcsection3.wav',1000),
             ('ir','sh','ingi0ir.wav',1000),('i0','sh','vcsection3.wav',1600)]
    lines=[]
    for left,right,wav,offset in targets:
        if bad=='missing' and (left,right)==('i0','sh'):continue
        lines.append(f'{wav}={prefix}{left} {right},{offset},100,300,90,40')
    if bad=='bad_oto':lines[0]=lines[0].replace(',100,300,90,40',',nan,300,90,40')
    with zipfile.ZipFile(arc,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr(root+'oto.ini','\n'.join(lines)+'\n')
        for wav in ('vcsection2.wav','vcsection3.wav','ingi0ir.wav'):
            z.writestr(root+wav,_synth_wav(seed+(0 if bad=='cloned_pcm' else len(wav))))
    return dict(style='mang',pitch=pitch,archive=str(arc),root=root,prefix=pitch,reference=pitch=='D4')


def _config(tmp_path, *, clone=False):
    return dict(schema='phonoweave.multisubbank_input.v1',banks=[
        _bank(tmp_path,'D4',seed=0),
        _bank(tmp_path,'G3',seed=0 if clone else 4),
    ])


def test_frozen_pilot_groups_and_no_split_claim(tmp_path):
    data=build_audit(_config(tmp_path))
    assert data['schema']==SCHEMA
    assert data['summary']['subbanks']==2
    assert data['summary']['target_oto_intervals']==12
    assert data['summary']['all_subbanks_cover_all_six_targets']
    assert data['approved_split_or_share_decisions']==0
    assert data['independent_recording_status']=='unknown'
    assert len(data['pairwise'])==12
    assert data['summary']['source_wav_duration_spread_ms_max']==0
    assert data['subbanks'][1]['fraction_exact_oto_rows_vs_D4']==1
    assert len([x for x in data['targets'] if x['status']=='human_reviewed_D4'])==6
    assert len([x for x in data['targets'] if x['status']=='transferred_window_unverified'])==6


def test_no_overwrite_and_source_archives_untouched(tmp_path):
    config=_config(tmp_path)
    before={p:Path(p).read_bytes() for p in [b['archive'] for b in config['banks']]}
    result=build_audit(config)
    dest=tmp_path/'report'
    write_reports(result,dest)
    assert (dest/'summary.json').exists()
    with pytest.raises(FileExistsError):write_reports(result,dest)
    for p,value in before.items():assert Path(p).read_bytes()==value


def test_missing_source_fails_closed_as_unmeasured(tmp_path):
    cfg=_config(tmp_path)
    cfg['banks'][1]=_bank(tmp_path,'G3',seed=4,bad='missing')
    report=build_audit(cfg)
    assert not report['summary']['all_subbanks_cover_all_six_targets']
    assert any(x['status']=='missing_or_duplicate_source_oto' for x in report['targets'])
    assert len(report['pairwise'])<12


def test_nonfinite_oto_and_missing_reference_rejected(tmp_path):
    cfg=_config(tmp_path)
    cfg['banks'][1]=_bank(tmp_path,'G3',seed=4,bad='bad_oto')
    with pytest.raises(ValueError,match='nonfinite'):build_audit(cfg)
    cfg=_config(tmp_path)
    cfg['banks'][0]['reference']=False
    with pytest.raises(ValueError,match='one D4'):build_audit(cfg)


def test_duplicate_pcm_does_not_establish_independent_takes(tmp_path):
    cfg=_config(tmp_path,clone=True)
    result=build_audit(cfg)
    assert result['summary']['target_pcm_hash_cross_bank_duplicates']
    assert result['independent_recording_status']=='unknown'


def test_invalid_input_and_unmappable_prefix(tmp_path):
    cfg=_config(tmp_path)
    cfg['banks'][1]['root']='../../evil/'
    with pytest.raises(ValueError,match='invalid ZIP'):build_audit(cfg)
    cfg=_config(tmp_path)
    cfg['banks'][1]['prefix']='x'
    result=build_audit(cfg)
    assert not result['summary']['all_subbanks_cover_all_six_targets']


def test_features_no_fabricated_signal():
    x=np.zeros(44100)
    assert sample_metrics(x,44100,300) is None
    x=np.sin(2*np.pi*200*np.arange(44100)/44100)*0.1
    m=sample_metrics(x,44100,300)
    assert m is not None and m['hf_fraction']<.001
    assert len(m['shape'])==12


def test_nonindependent_same_length_not_mislabeled():
    assert len(TARGETS)==6
    assert set(D4_REVIEWED_INTERVALS)=={(a,b) for a,b,_ in TARGETS}

def test_author_confirmed_diffsinger_provenance_changes_population(tmp_path):
    cfg=_config(tmp_path)
    cfg['source_provenance']={
        'generation_method':'diffsinger_direct_render',
        'attestation':'author_confirmed',
        'model_checkpoint_identity':'unknown',
        'shared_input_score_or_timing':'unknown',
    }
    report=build_audit(cfg)
    assert report['evaluation_population']=='diffsinger_generated_voicebank_audio'
    assert report['independent_recording_status']=='not_human_recording_repeats_synthetic_output'
    assert report['source_provenance']['model_checkpoint_identity']=='unknown'
    assert report['approved_split_or_share_decisions']==0

def test_unconfirmed_generation_provenance_fails_closed(tmp_path):
    cfg=_config(tmp_path)
    cfg['source_provenance']={'generation_method':'diffsinger_direct_render',
                              'attestation':'not_provided'}
    with pytest.raises(ValueError,match='author confirmation'):
        build_audit(cfg)
