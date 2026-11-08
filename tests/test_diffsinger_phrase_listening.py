"""Blind package provenance/anti-leakage tests: synthetic PCM only."""
from pathlib import Path
from copy import deepcopy
import csv
import json
import pytest

from phonoweave.diffsinger_phrase_listening import SCHEMA,package_blind_listening
from phonoweave.diffsinger_note_phrase_ab import frozen_score,render_note_phrase_ab
from test_diffsinger_note_phrase_ab import _manifest,_transform,_bank


def test_blind_full_match_only_K2_and_immutable_wavs(tmp_path):
    config=_manifest(tmp_path)
    config['banks'].append(_bank(tmp_path,'lu','C4'))
    frozen=tmp_path/'rendered'
    render_note_phrase_ab(config,frozen_score(),frozen,transform=_transform)
    result=package_blind_listening(frozen,tmp_path/'blind')
    assert result['schema']==SCHEMA
    assert len(result['trials'])==12
    assert not result['scientific_effect_inferred']
    answer=list(csv.DictReader((tmp_path/'blind/ResearchOnly/answer_key_PRIVATE.csv').open()))
    assert len(answer)==12
    assert all(r['A_policy'].startswith('k2_') and r['B_policy'].startswith('k2_') for r in answer)
    for trial in result['trials']:
        assert (tmp_path/'blind'/trial['A_wav']).is_file()
        assert (tmp_path/'blind'/trial['B_wav']).is_file()
    assert not any('k3' in s for s in (tmp_path/'blind/index.html').read_text().splitlines())
    with pytest.raises(FileExistsError):package_blind_listening(frozen,tmp_path/'blind')


def test_blind_rejects_tampered_audio_and_missing_style(tmp_path):
    config=_manifest(tmp_path)
    config['banks'].append(_bank(tmp_path,'lu','C4'))
    root=tmp_path/'render'
    report=render_note_phrase_ab(config,frozen_score(),root,transform=_transform)
    target=root/report['renders'][0]['rendered_wav']
    target.write_bytes(target.read_bytes()+b'BAD')
    with pytest.raises(ValueError,match='hash mismatch'):
        package_blind_listening(root,tmp_path/'out')
    with pytest.raises(ValueError,match='same|exact equal-budget'):
        package_blind_listening(root,tmp_path/'out',focus_banks=('unknown_bank',))
