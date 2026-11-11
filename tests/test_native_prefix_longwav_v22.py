"""Read-only native prefix / long-WAV regression tests (no phonetic claims)."""
import json
import wave
from pathlib import Path
import pytest
from phonoweave.native_recording_plan_v20 import (
    TARGET_SCHEMA, read_bank_alias_prefixes, build_native_recording_workflow,
)

def make_wav(path: Path, amplitude: int = 100):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(amplitude.to_bytes(2,'little',signed=True)*1000)

def manifest(path, *, prefix=None, line='a shw', wav='a shw.wav'):
    data = {'schema': TARGET_SCHEMA,
            'targets': [{'line': line, 'wav': wav, 'subbanks': ['D4']}]}
    if prefix is not None:
        data['bank_alias_prefixes'] = {'D4': prefix}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    return path

def test_author_declared_bank_prefix_locates_alias_in_long_recording(tmp_path):
    root=tmp_path/'bank'; sub=root/'D4'
    make_wav(sub/'a.wav')
    (sub/'oto.ini').write_text('a.wav=D4a shw,0,0,0,0,0\n',encoding='utf-8')
    data=build_native_recording_workflow(root,expected_reclist=manifest(tmp_path/'target.json',prefix='D4'))
    assert data['recording_plan']['recording_lines']==[]
    row=data['coverage']['template_states'][0]['banks'][0]
    assert row['status']=='audio_present_registered'
    assert row['match']=='declared_prefix_oto_alias'
    assert data['voicebank']['declared_alias_prefix_banks']==['D4']

def test_no_prefix_is_never_guessed(tmp_path):
    root=tmp_path/'bank'; sub=root/'D4'
    make_wav(sub/'a.wav')
    (sub/'oto.ini').write_text('a.wav=D4a shw,0,0,0,0,0\n',encoding='utf-8')
    data=build_native_recording_workflow(root,expected_reclist=manifest(tmp_path/'target.json'))
    assert data['coverage']['template_states'][0]['banks'][0]['status']=='unverified_template_mapping'
    assert data['recording_plan']['recording_lines']==[]
    assert data['recording_plan']['oto_review_without_rerecording']

def test_wrong_prefix_does_not_match(tmp_path):
    root=tmp_path/'bank'; sub=root/'D4'
    make_wav(sub/'a.wav')
    (sub/'oto.ini').write_text('a.wav=D4a shw,0,0,0,0,0\n',encoding='utf-8')
    data=build_native_recording_workflow(root,expected_reclist=manifest(tmp_path/'target.json',prefix='G3'))
    assert data['coverage']['template_states'][0]['banks'][0]['status']=='unverified_template_mapping'
    assert data['recording_plan']['recording_lines']==[]

def test_same_named_alias_in_two_long_wavs_is_review_not_rerecord(tmp_path):
    root=tmp_path/'bank'; sub=root/'D4'
    make_wav(sub/'a.wav');make_wav(sub/'b.wav',amplitude=200)
    (sub/'oto.ini').write_text(
        'a.wav=D4a shw,0,0,0,0,0\n'
        'b.wav=D4a shw,0,0,0,0,0\n',encoding='utf-8')
    data=build_native_recording_workflow(root,expected_reclist=manifest(tmp_path/'target.json',prefix='D4'))
    assert data['coverage']['template_states'][0]['banks'][0]['status']=='ambiguous_oto_alias'
    assert data['recording_plan']['recording_lines']==[]
    assert data['recording_plan']['oto_review_without_rerecording']

def test_one_long_wav_provides_two_aliases_without_duplicate_recording(tmp_path):
    root=tmp_path/'bank'; sub=root/'D4'
    make_wav(sub/'a.wav')
    (sub/'oto.ini').write_text('a.wav=D4a shw,0,0,0,0,0\n'
                             'a.wav=D4a hw,100,0,0,0,0\n', encoding='utf-8')
    data={'schema':TARGET_SCHEMA,'bank_alias_prefixes':{'D4':'D4'},
          'targets':[{'line':'a shw','subbanks':['D4']},
                     {'line':'a hw','subbanks':['D4']}]}
    m=tmp_path/'target.json';m.write_text(json.dumps(data),encoding='utf-8')
    report=build_native_recording_workflow(root,expected_reclist=m)
    assert report['recording_plan']['selected_count']==0
    assert report['coverage']['expected_template_count']==2
    assert all(x['banks'][0]['match']=='declared_prefix_oto_alias' for x in report['coverage']['template_states'])

@pytest.mark.parametrize('prefix',['../','D4\n','D4\\','D4\x00',''])
def test_unsafe_prefix_rejected(tmp_path,prefix):
    with pytest.raises(ValueError,match='alias prefix'):
        read_bank_alias_prefixes(manifest(tmp_path/'target.json',prefix=prefix))

def test_text_template_keeps_original_exact_match_rules(tmp_path):
    m=tmp_path/'reclist.txt';m.write_text('a shw\n',encoding='utf-8')
    assert read_bank_alias_prefixes(m)=={}
