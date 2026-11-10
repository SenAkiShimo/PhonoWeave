"""Synthetic integration tests only: NOT claims about Mandarin alignment quality."""
from __future__ import annotations
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import wave
import zipfile

import numpy as np
import pytest
from phonoweave.whole_utterance_mfa_bridge import (
    SCHEMA, _parse_mfa_long_textgrid, build_corpus, doctor,
    digest_file, import_textgrids, prepare_run_command, run_mfa,
)


def bank(root:Path, bid='mang_D4', syllables=('a','ba')):
    arc=root/'test.zip'; sub='Bank/'
    pcm=(1000*np.sin(np.arange(44100)*2*np.pi*200/44100)).astype('<i2').tobytes()
    b=io.BytesIO()
    with wave.open(b,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(44100);w.writeframes(pcm)
    with zipfile.ZipFile(arc,'w') as z:
        z.writestr(sub+'a.wav',b.getvalue())
        z.writestr(sub+'vcsection.wav',b.getvalue())
        z.writestr(sub+'oto.ini','a.wav=D4- a,0,100,200,70,40\na.wav=D4a,30,100,200,70,40\na.wav=D4ba,500,100,200,70,40\n')
    pre=root/'pre';pre.mkdir()
    columns=['bank','style','pitch','source_archive','archive_sha256','oto_sha256','wav_count']
    with (pre/'source_banks.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerow(dict(bank=bid,style='mang',pitch='D4',source_archive=arc.name,archive_sha256=digest_file(arc),oto_sha256=sha256(zipfile.ZipFile(arc).read(sub+'oto.ini')).hexdigest(),wav_count=2))
    columns=['bank','source_wav','source_wav_zip_path','duration_ms','sample_rate','ota_oto_sha256']
    with (pre/'source_wavs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for name in ['a.wav','vcsection.wav']:
            w.writerow(dict(bank=bid,source_wav=name,source_wav_zip_path=sub+name,duration_ms=1000,sample_rate=44100,ota_oto_sha256=sha256(zipfile.ZipFile(arc).read(sub+'oto.ini')).hexdigest()))
    trans=dict(schema='phonoweave.whole_utterance_verified_transcript.v0.1',templates=[
        dict(source_wav='a.wav',banks=[bid],verification='not_verified',draft_sequence=list(syllables)),
        dict(source_wav='vcsection.wav',banks=[bid],verification='not_verified',draft_sequence=[]),
    ])
    templates=root/'templates.json';templates.write_text(json.dumps(trans))
    manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(banks=[dict(style='mang',pitch='D4',archive=str(arc),root=sub)])))
    return pre,templates,manifest


def test_no_unauthorized_draft_transcription_or_fabricated_phone_boundaries(tmp_path):
    pre,templates,manifest=bank(tmp_path)
    result=build_corpus(pre,templates,manifest,tmp_path/'stage')
    assert result['staged_wavs']==0
    assert result['excluded_wavs']==2
    assert result['phone_boundaries_created']==0
    with pytest.raises(FileExistsError):build_corpus(pre,templates,manifest,tmp_path/'stage')


def test_explicit_diagnostic_pilot_stages_original_pcm_and_does_not_elevate_truth(tmp_path):
    pre,t,m=bank(tmp_path)
    result=build_corpus(pre,t,m,tmp_path/'stage',allow_unverified_pilot=True)
    assert result['staged_wavs']==1 and result['excluded_wavs']==1
    assert result['staged_syllable_tokens']==2
    assert (tmp_path/'stage/corpus/mang_D4/a.lab').read_text().strip()=='a ba'
    assert (tmp_path/'stage/corpus/mang_D4/a.wav').read_bytes()==zipfile.ZipFile(tmp_path/'test.zip').read('Bank/a.wav')
    assert 'UNVERIFIED_PILOT_ONLY' in (tmp_path/'stage/staged_source_index.csv').read_text()
    assert result['acoustic_model_ran'] is False


def test_verified_transcript_needs_evidence_and_trusted_archive(tmp_path):
    pre,t,m=bank(tmp_path)
    data=json.loads(t.read_text())
    data['templates'][0]['verification']='verified_against_source_score'
    t.write_text(json.dumps(data))
    report=build_corpus(pre,t,m,tmp_path/'n')
    assert report['staged_wavs']==0
    data['templates'][0]['verification_evidence']='frozen original synthesis score id test'
    t.write_text(json.dumps(data))
    report=build_corpus(pre,t,m,tmp_path/'verified')
    assert report['staged_wavs']==1
    assert 'source_score_or_author_verified' in (tmp_path/'verified/staged_source_index.csv').read_text()
    with zipfile.ZipFile(tmp_path/'test.zip','a') as z:z.writestr('tamper','hi')
    with pytest.raises(ValueError,match='hash mismatch'):build_corpus(pre,t,m,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_mfa_commands_are_real_not_fake_and_explicitly_report_missing(tmp_path,monkeypatch):
    assert prepare_run_command(Path('/a'),Path('/b'))[1]=='align_hf'
    assert prepare_run_command(Path('/a'),Path('/b'),mode='legacy')[1]=='align'
    with pytest.raises(ValueError):prepare_run_command(Path('/a'),Path('/b'),mode='made_up')
    pre,t,m=bank(tmp_path)
    build_corpus(pre,t,m,tmp_path/'stage',allow_unverified_pilot=True)
    monkeypatch.setattr('phonoweave.whole_utterance_mfa_bridge.shutil.which',lambda name:None)
    assert doctor()['mfa_installed'] is False
    with pytest.raises(RuntimeError,match='MFA missing'):run_mfa(tmp_path/'stage',tmp_path/'align_out')


def _grid() -> str:
    return '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1
        intervals: size = 1
        intervals [1]:
            xmin = 0.05
            xmax = 0.95
            text = "a"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1
        intervals: size = 2
        intervals [1]:
            xmin = 0.05
            xmax = 0.4
            text = "p"
        intervals [2]:
            xmin = 0.4
            xmax = 0.95
            text = "a"
'''


def test_import_requires_source_bound_successful_model_receipt(tmp_path):
    pre,t,m=bank(tmp_path)
    build_corpus(pre,t,m,tmp_path/'stage',allow_unverified_pilot=True)
    aligned=tmp_path/'grids/mang_D4';aligned.mkdir(parents=True)
    (aligned/'a.TextGrid').write_text(_grid())
    assert len(_parse_mfa_long_textgrid(aligned/'a.TextGrid'))==3
    with pytest.raises(ValueError,match='missing successful model-execution receipt'):
        import_textgrids(tmp_path/'stage',tmp_path/'grids',tmp_path/'out')
    receipt=dict(schema='phonoweave.mfa_runtime_receipt.v0.1',model_executed=True,model_identifier='mandarin_mfa',frozen_stage_index_sha256=digest_file(tmp_path/'stage/staged_source_index.csv'))
    (tmp_path/'grids/_phonoweave_mfa_run_receipt.json').write_text(json.dumps(receipt))
    result=import_textgrids(tmp_path/'stage',tmp_path/'grids',tmp_path/'out')
    assert result['phone_intervals']==2 and result['gold_boundaries']==0
    assert result['alignment_accuracy_measured'] is False
    assert (tmp_path/'out/aligned_intervals.csv').exists()
    with pytest.raises(FileExistsError):import_textgrids(tmp_path/'stage',tmp_path/'grids',tmp_path/'out')


def test_stale_receipt_and_bad_phone_times_rejected(tmp_path):
    pre,t,m=bank(tmp_path);build_corpus(pre,t,m,tmp_path/'stage',allow_unverified_pilot=True)
    g=tmp_path/'grids';(g/'mang_D4').mkdir(parents=True)
    (g/'mang_D4/a.TextGrid').write_text(_grid().replace('xmax = 0.95\n            text = "a"','xmax = 9.5\n            text = "a"'))
    receipt=dict(schema='phonoweave.mfa_runtime_receipt.v0.1',model_executed=True,model_identifier='mandarin_mfa',frozen_stage_index_sha256='0'*64)
    (g/'_phonoweave_mfa_run_receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='stale'):import_textgrids(tmp_path/'stage',g,tmp_path/'out')
    receipt['frozen_stage_index_sha256']=digest_file(tmp_path/'stage/staged_source_index.csv')
    (g/'_phonoweave_mfa_run_receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='exceeds original'):import_textgrids(tmp_path/'stage',g,tmp_path/'out')
