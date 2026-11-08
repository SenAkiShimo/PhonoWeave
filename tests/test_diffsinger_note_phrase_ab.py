"""Software tests on synthetic fixture WAVs; no real-singing efficacy evidence."""
from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import wave
import zipfile

import numpy as np
import pytest

from phonoweave.diffsinger_note_phrase_ab import (
    OUTPUT_SCHEMA, SCORE_SCHEMA, _frozen_validate, _pitch_root, frozen_score,
    render_note_phrase_ab, rubberband_process,
)


def _wav(rate=44100,seed=0,length_seconds=3.0):
    t=np.arange(round(length_seconds*rate))/rate
    rng=np.random.default_rng(seed)
    audio=.08*np.sin(2*np.pi*(220+seed*15)*t)+.003*rng.standard_normal(len(t))
    output=io.BytesIO()
    with wave.open(output,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate)
        w.writeframes((audio*32767).astype('<i2').tobytes())
    return output.getvalue()


def _bank(tmp,style='mang',pitch='D4'):
    root=f'{style}_{pitch}/'
    archive=tmp/f'{style}_{pitch}.zip'
    aliases=[]
    for i,l in enumerate(('i','ir','i0')):
        aliases.append(f'vcsection2.wav={pitch}{l} m,{900+350*i},110,120,140,90')
    for i,l in enumerate(('i','ir','i0')):
        wav='ingi0ir.wav' if l=='ir' else 'vcsection3.wav'
        aliases.append(f'{wav}={pitch}{l} sh,{900+350*i},110,120,140,90')
    aliases.extend([f'eriin.wav={pitch}li,900,110,120,140,90',
                    f'ingi0ir.wav={pitch}shi,1200,110,120,140,90',
                    f'ingi0ir.wav={pitch}zi,1900,110,120,140,90',
                    f'a.wav={pitch}ma,1000,110,120,140,90',
                    f'a.wav={pitch}sha,1700,110,120,140,90'])
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr(root+'oto.ini','\n'.join(aliases)+'\n')
        for i,name in enumerate(('a.wav','eriin.wav','ingi0ir.wav','vcsection2.wav','vcsection3.wav')):
            z.writestr(root+name,_wav(seed=i+1))
    return {'style':style,'pitch':pitch,'root':root,'prefix':pitch,'archive':str(archive)}


def _manifest(tmp):
    return {'schema':'phonoweave.diffsinger_vc_splice_input.v1',
            'source_provenance':'author_confirmed_diffsinger_direct',
            'voicebank_id':'LynneQing','banks':[_bank(tmp)]}


def _transform(x, rate, *, target_frames, semitones):
    # Software test shim only; *not* a pitch-preserving singing renderer.
    src=np.linspace(0,1,len(x))
    return np.interp(np.linspace(0,1,target_frames),src,x)


def test_frozen_score_six_targets_and_pitch_root():
    score=frozen_score();_frozen_validate(score)
    assert score['schema']==SCORE_SCHEMA
    assert {(x['left'],x['right']) for p in score['phrases'] for x in p['pairs']}=={
        (x,y) for x in ('i','ir','i0') for y in ('m','sh')}
    assert _pitch_root('D4')==62
    assert _pitch_root('C4')==60
    with pytest.raises(ValueError):_pitch_root('not-a-pitch')


def test_synthetic_render_four_policies_two_phrases_and_equal_budget(tmp_path):
    config=_manifest(tmp_path)
    score=frozen_score()
    out=tmp_path/'out'
    report=render_note_phrase_ab(config,score,out,transform=_transform)
    assert report['schema']==OUTPUT_SCHEMA
    assert report['engine']=='injected_test_transform_not_external_engine'
    assert len(report['renders'])==8
    assert {x['VC_source_ms'] for x in report['budget_ledger'] if x['policy'].startswith('k2_')}=={480}
    assert {x['total_deduplicated_source_ms'] for x in report['budget_ledger'] if x['policy'].startswith('k2_')}=={2140}
    assert {x['VC_source_ms'] for x in report['budget_ledger'] if x['policy'].startswith('k3_')}=={720}
    assert not report['experiment_gates']['independent_DiffSinger_teacher_reference_present']
    assert not report['experiment_gates']['split_share_quality_gain_established']
    for row in report['renders']:
        with wave.open(str(out/row['rendered_wav'])) as wav:
            assert wav.getframerate()==44100
            assert wav.getnchannels()==1
            assert abs(wav.getnframes()*1000/44100-row['actual_milliseconds'])<.02
        assert row['target_note_count']==6
        assert not row['teacher_quality_scored']
    with pytest.raises(FileExistsError):render_note_phrase_ab(config,score,out,transform=_transform)


def test_generated_phrases_not_external_teacher_and_source_unchanged(tmp_path):
    config=_manifest(tmp_path)
    src=Path(config['banks'][0]['archive']);original=src.read_bytes()
    out=tmp_path/'out'
    render_note_phrase_ab(config,frozen_score(),out,transform=_transform)
    assert src.read_bytes()==original
    receipt=json.loads((out/'teacher_reference_REQUIRED.json').read_text())
    assert receipt['reference_wavs']==[]
    assert receipt['status']=='required_unprovided_not_scored'


def test_bad_score_and_hidden_extra_condition_rejected(tmp_path):
    config=_manifest(tmp_path)
    bad=deepcopy(frozen_score())
    bad['phrases'][1]['pairs'][0]['left']='i'
    with pytest.raises(ValueError,match='repeated'):render_note_phrase_ab(config,bad,tmp_path/'bad',transform=_transform)
    bad=deepcopy(frozen_score());bad['phrases'][0]['pairs'][0]['semitones']=[0,999]
    with pytest.raises(ValueError,match='pitch/duration'):render_note_phrase_ab(config,bad,tmp_path/'bad',transform=_transform)


def test_no_cross_style_donor_and_missing_oto_alias_rejected(tmp_path):
    config=_manifest(tmp_path)
    config['banks'].append(_bank(tmp_path,style='lu',pitch='C4'))
    report=render_note_phrase_ab(config,frozen_score(),tmp_path/'out',transform=_transform)
    assert len(report['renders'])==16
    assert {r['style'] for r in report['renders']}=={'mang','lu'}
    assert all(r['source_CV_aliases']['i'].startswith(r['pitch']) for r in report['renders'])


def test_ffmpeg_rubberband_actual_audio_if_available():
    if not shutil.which('ffmpeg'):pytest.skip('ffmpeg not installed in GitHub runner')
    rate=44100
    x=.1*np.sin(2*np.pi*293.665*np.arange(int(rate*.45))/rate)
    try:
        y=rubberband_process(x,rate,target_frames=round(rate*.6),semitones=3)
    except RuntimeError as error:
        if 'rubberband' in str(error).lower():pytest.skip('FFmpeg lacks rubberband filter')
        raise
    assert len(y)==round(rate*.6)
    assert np.max(abs(y))>.01
    assert np.max(abs(y))<1
    assert not np.allclose(y[:len(x)],x)


def test_extra_budget_reference_is_not_matched():
    assert frozen_score()['target_reference'].startswith('absent_external')
