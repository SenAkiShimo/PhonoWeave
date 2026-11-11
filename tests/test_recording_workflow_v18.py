"""Regression tests for the one-command, non-destructive recording workflow.

Synthetic Mandarin OTO/WAV fixture is a wiring test, not evidence of verified
phoneme segmentation, recording savings or acoustically independent takes.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import phonoweave.recording_workflow_v18 as mod


def report(root: Path) -> dict:
    return {
        'schema': mod.SCHEMA,
        'voicebank': {'root': str(root), 'oto_entries': 2,
                      'oto_entries_with_existing_wav': 2},
        'coverage': {'recognized_syllable_observations': 2},
        'evidence': {'gaps': [], 'diagnostic_recording_gaps': 1,
                     'needs_other_validation': 0},
        'recording_plan': {'selected_count': 1, 'max_lines': 8,
                           'already_scheduled_count': 0,
                           'recording_lines': ['a_fa_a'],
                           'selected': [{'line': 'a_fa_a', 'base_unit': 'f',
                                         'context_family': 'other', 'gap_type': 'coverage_limited',
                                         'existing_distinct_pcm_sources': 1}]},
        'warnings': [],
    }


def stand_ins(monkeypatch, root: Path, *, missing=0, recognized=3, unsupported=0,
              gaps=1, suggestions=1):
    calls = {'analysis': 0, 'source': 0}
    inspect = NS(oto_files=1, entries=3, valid_entries=3-missing,
                 missing_wavs=missing, parse_warnings=0, subbanks=(1,))
    coverage = NS(observations=recognized, onset_observations=recognized,
                  zero_onset_observations=0,
                  items=tuple([NS(status='analyzed', base_unit='f', observations=2)] +
                              [NS(status='unsupported', base_unit='z', observations=1)] * unsupported))
    analysis = NS(decisions=(NS(base_unit='f', class_name='fricative', decision='unresolved',
                                confidence='low', acoustic_evidence='partial_coverage_limited',
                                synthesis_evidence='not_tested'),))
    completion = NS(gaps=tuple(NS(base_unit='f', class_name='fricative', gap_type='coverage_limited',
                                  priority='high', recommended_action='supplemental_recording',
                                  role_scope='internal', context_families=('other',), rationale=('test',))
                                for _ in range(gaps)),
                    supplemental_recording=tuple(range(gaps)), perceptual_validation=())
    monkeypatch.setattr(mod, 'inspect_voicebank', lambda path: inspect)
    monkeypatch.setattr(mod, 'analyze_coverage', lambda path: coverage)
    def analyze(path):
        calls['analysis'] += 1
        return analysis
    monkeypatch.setattr(mod, 'analyze_voicebank_inventory', analyze)
    monkeypatch.setattr(mod, 'build_evidence_completion_plan', lambda a: completion)
    monkeypatch.setattr(mod, 'build_supplement_plan', lambda a: NS(requests=()))
    def inspect_sources(path):
        calls['source'] += 1
        return {'fingerprinted_source_wav_paths': 2,
                'unreadable_or_missing_wav_count': 0}
    monkeypatch.setattr(mod, 'inspect_recording_sources', inspect_sources)
    monkeypatch.setattr(mod, '_read_existing_reclist', lambda path: ['a_fu_a'] if path else [])
    def plan(a,b,c,*,max_lines,existing_lines):
        return {'recording_lines': ['a_fa_a'][:min(max_lines,suggestions)],
                'selected_count': min(max_lines,suggestions),
                'max_lines': max_lines,
                'selected': [report(root)['recording_plan']['selected'][0]][:min(max_lines,suggestions)],
                'already_scheduled_count': len(existing_lines),
                'unfilled_by_original_selector': [],
                'auto_split_merge': False}
    monkeypatch.setattr(mod, 'source_aware_plan', plan)
    return calls


def test_single_pipeline_and_explicit_source_aware(monkeypatch, tmp_path):
    calls = stand_ins(monkeypatch, tmp_path)
    result = mod.build_recording_workflow(tmp_path, max_lines=5,
                                          existing_reclist=tmp_path/'scheduled.txt')
    assert calls == {'analysis': 1, 'source': 1}
    assert result['schema'] == mod.SCHEMA
    assert result['recording_plan']['already_scheduled_count'] == 1
    assert result['recording_plan']['recording_lines'] == ['a_fa_a']
    assert result['restrictions']['voicebank_unchanged']
    assert result['restrictions']['no_new_phoneme_or_alias_generated']


def test_zero_budget_no_fake_completion(monkeypatch, tmp_path):
    stand_ins(monkeypatch, tmp_path)
    data = mod.build_recording_workflow(tmp_path, max_lines=0)
    assert data['recording_plan']['recording_lines'] == []
    assert any('does not prove complete' in x for x in data['warnings'])


def test_invalid_root_refused_before_scan(tmp_path):
    with pytest.raises(ValueError, match='existing directory'):
        mod.build_recording_workflow(tmp_path/'absent')


def test_empty_oto_failure(monkeypatch, tmp_path):
    stand_ins(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, 'inspect_voicebank', lambda path: NS(oto_files=0, entries=0, valid_entries=0))
    with pytest.raises(ValueError, match='oto.ini'):
        mod.build_recording_workflow(tmp_path)


def test_missing_all_wavs_fails(monkeypatch, tmp_path):
    stand_ins(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, 'inspect_voicebank', lambda path: NS(oto_files=1, entries=3, valid_entries=0))
    with pytest.raises(ValueError, match='missing'):
        mod.build_recording_workflow(tmp_path)


def test_no_mandarin_labels_is_not_complete(monkeypatch, tmp_path):
    stand_ins(monkeypatch, tmp_path, recognized=0)
    with pytest.raises(ValueError, match='recognized Mandarin'):
        mod.build_recording_workflow(tmp_path)


def test_warnings_are_visible(monkeypatch, tmp_path):
    stand_ins(monkeypatch, tmp_path, missing=1, unsupported=1)
    data = mod.build_recording_workflow(tmp_path)
    assert len(data['warnings']) == 2
    assert data['coverage']['onset_type_counts']['unsupported'] == 1


@pytest.mark.parametrize('invalid', [-1, 10001, True, 1.5, '8'])
def test_budget_validation(monkeypatch, tmp_path, invalid):
    stand_ins(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match='max_lines'):
        mod.build_recording_workflow(tmp_path, max_lines=invalid)


def test_preview_does_not_write(monkeypatch, tmp_path, capsys):
    stand_ins(monkeypatch, tmp_path)
    before = set(tmp_path.iterdir())
    assert mod.main([str(tmp_path),'--max-lines','1']) == 0
    assert set(tmp_path.iterdir()) == before
    assert 'a_fa_a' in capsys.readouterr().out


def test_explicit_export_creates_two_files(tmp_path):
    src = tmp_path/'voicebank'; src.mkdir()
    dest = tmp_path/'new_results'
    result = report(src)
    assert mod.export_recording_workflow(result,dest,voicebank=src) == dest
    assert (dest/'supplement_reclist.txt').read_text() == 'a_fa_a\n'
    assert json.loads((dest/'recording_plan.json').read_text()) == result
    with pytest.raises(FileExistsError):
        mod.export_recording_workflow(result,dest,voicebank=src)


def test_export_cannot_modify_source_tree(tmp_path):
    src = tmp_path/'voicebank'; src.mkdir()
    with pytest.raises(ValueError, match='outside'):
        mod.export_recording_workflow(report(src),src/'subdir',voicebank=src)
    assert not (src/'subdir').exists()


def test_export_parent_must_exist(tmp_path):
    src = tmp_path/'voicebank'; src.mkdir()
    with pytest.raises(ValueError,match='parent directory'):
        mod.export_recording_workflow(report(src),tmp_path/'missing'/'new',voicebank=src)


def test_preview_json(monkeypatch, tmp_path, capsys):
    stand_ins(monkeypatch,tmp_path)
    assert mod.main([str(tmp_path),'--json']) == 0
    assert json.loads(capsys.readouterr().out)['schema'] == mod.SCHEMA


def test_cli_rejects_existing_export_before_analysis(monkeypatch, tmp_path):
    stand_ins(monkeypatch,tmp_path)
    with pytest.raises(SystemExit):
        mod.main([str(tmp_path),'--export',str(tmp_path)])


def test_end_to_end_small_real_oto_pcm_fixture(monkeypatch, tmp_path):
    """No artificial alias/new phone created; only legal diagnostics are proposed."""
    from phonoweave.inventory import InventoryDecision, VoicebankInventoryAnalysis
    src = tmp_path/'source'; src.mkdir()
    for name in ('fa','fu'):
        with wave.open(str(src/(name+'.wav')), 'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(44100)
            w.writeframes(b'\x00\x00' * 600)
    (src/'oto.ini').write_text('fa.wav=fa,0,0,0,0,0\nfu.wav=fu,0,0,0,0,0\n',encoding='utf-8')
    fake_analysis=VoicebankInventoryAnalysis(
        voicebank=src,
        decisions=[InventoryDecision(base_unit='f',class_name='fricative',
            acoustic_evidence='partial_coverage_limited',synthesis_evidence='not_tested',
            decision='unresolved',confidence='low',
            notes=('context_families=rounded,other','role_scope=internal'))])
    monkeypatch.setattr(mod,'analyze_voicebank_inventory',lambda path:fake_analysis)
    data=mod.build_recording_workflow(src,max_lines=2)
    assert data['voicebank']['oto_entries']==2
    assert data['coverage']['recognized_syllable_observations']==2
    assert data['recording_plan']['selected_count']<=2
    assert data['recording_plan']['recording_lines']
    assert not any((src/x).exists() for x in ('new_reclist.txt','recording_plan.json'))