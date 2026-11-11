"""Synthetic invariants and frozen-archive preflight checks. No phone accuracy claims."""
import json
import wave
import zipfile
import io
from pathlib import Path
import pytest

from phonoweave.native_pilot_v8 import (
    prepare, verify_sequences, extract_audio, _safe_key, _pcm_sha,
    _sha_bytes, SCHEMA, SEQUENCE_SCHEMA,
)

ARCHIVE_DIR = Path('/mnt/data/pw_real_v8')


def archive_args():
    return [ARCHIVE_DIR / n for n in [
        'PhonoWeave_User_Native_Batch_Discovery_v2.zip',
        'PhonoWeave_User_Native_Temporal_Context_v3.zip',
        'PhonoWeave_WholeAudio_Unanchored_Events_v4.zip',
        'PhonoWeave_UserNative_EventLink_v5.zip',
    ]]


def test_real_artifacts_v4_v5_identity_and_preflight(tmp_path):
    out = tmp_path / 'pilot'
    report = prepare(*archive_args(), out)
    assert report['verified_v4_v5_sources'] == 175
    assert report['source_hash_mismatches'] == 0
    assert report['pilot_files'] == 4 and report['development_smoke_files'] == 2
    assert report['pilot_duration_seconds'] == pytest.approx(147.446)
    assert report['can_run_v6'] is False
    coh = json.loads((out/'cohort_frozen.json').read_text())
    assert len(coh['sources']) == 6
    assert coh['corpus']['v5_candidate_regions'] == 19610
    assert coh['corpus']['v5_hint_regions'] == 12197
    assert coh['schema'] == SCHEMA
    assert 'NOT untouched confirmatory' in ' '.join(coh['important_limitations'])
    raw = (out/'cohort_frozen.json').read_bytes()
    assert report['cohort_sha256'] == _sha_bytes(raw)
    assert '/mnt/data/LynneQing' not in json.dumps(coh)


def test_template_explicitly_fails_script_verification(tmp_path):
    out = tmp_path / 'pilot'
    prepare(*archive_args(), out)
    seq = json.loads((out/'sequences_TEMPLATE.json').read_text())
    assert seq['schema'] == SEQUENCE_SCHEMA
    assert len(seq['entries']) == 4
    assert all(not item['units'] and item['provenance'].startswith('PENDING')
               for item in seq['entries'])
    with pytest.raises(ValueError, match='independent recording-script provenance'):
        verify_sequences(out/'cohort_frozen.json', out/'sequences_TEMPLATE.json')


def test_script_requirements_and_provenance_honesty(tmp_path):
    out = tmp_path/'pilot'
    prepare(*archive_args(), out)
    seq = json.loads((out/'sequences_TEMPLATE.json').read_text())
    for item in seq['entries']:
        item['provenance'] = 'independent_recording_script'
        item['units'] = ['示例', 'i0']
    p=tmp_path/'sample.json';p.write_text(json.dumps(seq, ensure_ascii=False))
    r=verify_sequences(out/'cohort_frozen.json',p)
    assert r['verified_sources']==4 and r['units']==8
    assert 'human assertion' in r['warning']
    seq['entries'][0]['source_pcm_sha256']='0'*64
    p.write_text(json.dumps(seq))
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_sequences(out/'cohort_frozen.json',p)


def test_reject_extra_or_duplicate_script_entries(tmp_path):
    out=tmp_path/'pilot';prepare(*archive_args(),out)
    seq=json.loads((out/'sequences_TEMPLATE.json').read_text())
    seq['entries'].append(dict(seq['entries'][0]))
    p=tmp_path/'bad.json';p.write_text(json.dumps(seq))
    with pytest.raises(ValueError, match='duplicate'):
        verify_sequences(out/'cohort_frozen.json',p)


def test_no_overwrite_for_frozen_cohort(tmp_path):
    out=tmp_path/'pilot';prepare(*archive_args(),out)
    with pytest.raises(FileExistsError):prepare(*archive_args(),out)


def test_safe_path_names():
    with pytest.raises(ValueError):_safe_key('mang_D4', '../foo.wav')
    with pytest.raises(ValueError):_safe_key('evil/path', 'a.wav')


def synthetic_wav():
    b=io.BytesIO()
    with wave.open(b,'wb') as w:
        w.setframerate(44100);w.setnchannels(1);w.setsampwidth(2)
        w.writeframes(b'\x00\x02'*200)
    return b.getvalue()


def write_frozen(path, data):
    raw=(json.dumps(data)+'\n').encode('utf8')
    path.write_bytes(raw)
    (path.parent/'cohort_sha256.txt').write_text(_sha_bytes(raw)+'  cohort_frozen.json\n')


def test_extract_audio_verifies_zip_and_pcm(tmp_path):
    raw=synthetic_wav();zipfile_path=tmp_path/'archive.zip'
    with zipfile.ZipFile(zipfile_path,'w') as z:
        z.writestr('inside/solo.wav',raw)
    archive_sha=_sha_bytes(zipfile_path.read_bytes())
    cohort={'schema':SCHEMA,'sources':[{'bank':'mang_D4','source_wav':'solo.wav',
            'source_archive_sha256':archive_sha,'source_pcm_sha256':_pcm_sha(raw),
            'cohort_role':'retrospective_pilot'}]}
    c=tmp_path/'cohort.json';write_frozen(c,cohort)
    m=tmp_path/'manifest.json';m.write_text(json.dumps({'banks':{'mang_D4':{'archive':str(zipfile_path),'root':'inside/'}}}))
    report=extract_audio(c,m,tmp_path/'audio')
    assert report['audio_sources_verified']==1
    assert (tmp_path/'audio/mang_D4/solo.wav').read_bytes()==raw
    cohort['sources'][0]['source_pcm_sha256']='0'*64;write_frozen(c,cohort)
    with pytest.raises(ValueError,match='PCM SHA mismatch'):
        extract_audio(c,m,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_audio_archive_mismatch_refuses_partial_export(tmp_path):
    raw=synthetic_wav();za=tmp_path/'in.zip'
    with zipfile.ZipFile(za,'w') as z:z.writestr('inside/a.wav',raw)
    co={'schema':SCHEMA,'sources':[{'bank':'lu_C4','source_wav':'a.wav',
        'source_pcm_sha256':_pcm_sha(raw),'source_archive_sha256':'f'*64}]}
    c=tmp_path/'c.json';write_frozen(c,co)
    m=tmp_path/'m.json';m.write_text(json.dumps({'banks':{'lu_C4':{'archive':str(za),'root':'inside/'}}}))
    with pytest.raises(ValueError,match='ZIP SHA mismatch'):
        extract_audio(c,m,tmp_path/'audio')
    assert not (tmp_path/'audio').exists()


def test_cohort_tamper_detection(tmp_path):
    out=tmp_path/'pilot';prepare(*archive_args(),out)
    p=out/'cohort_frozen.json'
    p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='hash differs'):
        verify_sequences(p,out/'sequences_TEMPLATE.json')