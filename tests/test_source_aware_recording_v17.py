"""Standalone local component tests; real GitHub CI tests the full repo integration."""
from __future__ import annotations
import json
import wave
from dataclasses import dataclass
from pathlib import Path
import struct
import pytest
from phonoweave.diagnostic_selector import DiagnosticItem
from phonoweave.evidence_gap import EvidenceCompletionPlan, EvidenceGap
from phonoweave.supplement_plan import SupplementRequest, SupplementTarget
import phonoweave.source_aware_recording_v17 as mod


def wav(path: Path, payload: bytes, *, rate=44100):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(payload)


def sample_bytes(n=250, bias=0):
    return struct.pack('<' + 'h'*n, *(i+bias for i in range(n)))


def test_pcm_identity_catches_byte_identical_audio(tmp_path):
    a,b=tmp_path/'a.wav',tmp_path/'b.wav'
    wav(a,sample_bytes()); wav(b,sample_bytes())
    assert mod.pcm_identity(a)==mod.pcm_identity(b)


def test_pcm_identity_distinguishes_true_audio_change(tmp_path):
    a,b=tmp_path/'a.wav',tmp_path/'b.wav'
    wav(a,sample_bytes()); wav(b,sample_bytes(bias=1))
    assert mod.pcm_identity(a)!=mod.pcm_identity(b)


def test_pcm_identity_accounts_for_sample_rate(tmp_path):
    a,b=tmp_path/'a.wav',tmp_path/'b.wav'
    wav(a,sample_bytes()); wav(b,sample_bytes(),rate=48000)
    assert mod.pcm_identity(a)!=mod.pcm_identity(b)


def test_pcm_identity_rejects_broken_wav(tmp_path):
    p=tmp_path/'a.wav'; p.write_bytes(b'nonsense')
    with pytest.raises((wave.Error,EOFError,ValueError)):
        mod.pcm_identity(p)


@dataclass
class Entry:
    wav_path: Path
    oto_path: Path

@dataclass
class Observation:
    base_unit: str
    final: str
    entry: Entry


def test_oto_aliases_and_copied_pcm_are_not_extra_recordings(tmp_path,monkeypatch):
    a,b,c,d=[tmp_path/x for x in ('a.wav','b.wav','c.wav','d.wav')]
    wav(a,sample_bytes());wav(b,sample_bytes());wav(c,sample_bytes(bias=1));d.write_bytes(b'bad')
    oto=tmp_path/'oto.ini'
    observations=[Observation('f','a',Entry(a,oto)), Observation('f','a',Entry(a,oto)),
                  Observation('f','a',Entry(b,oto)), Observation('f','a',Entry(c,oto)),
                  Observation('f','a',Entry(d,oto))]
    monkeypatch.setattr(mod,'load_voicebank',lambda root:([],['warning']))
    monkeypatch.setattr(mod,'collect_observations',lambda *args: observations)
    result=mod.inspect_recording_sources(tmp_path)
    key=('f','a')
    assert result['oto_rows'][key]==5
    assert result['distinct_pcm'][key]==2
    assert result['distinct_wav_paths'][key]==3
    assert result['unreadable_or_missing_wav_count']==1
    assert result['oto_parse_warning_count']==1


def gap(base,contexts,priority='high',action='supplemental_recording'):
    return EvidenceGap(base_unit=base,class_name='fricative',gap_type='coverage_limited',
                       priority=priority,recommended_action=action,role_scope='internal',
                       context_families=contexts,rationale=())


def plan(count=2):
    from types import SimpleNamespace
    return SimpleNamespace(requests=(SupplementRequest(base_unit='f',class_name='fricative',gap_type='coverage_limited',priority='high',role_scope='internal',targets=(SupplementTarget(context_family='other',diagnostic_items=count),),pitch_policy='repeat_each_item_across_existing_oto_sets',automatic_round_limit=1,stop_rule='stop'),))


def fake_candidates(request,counts,family):
    # Existing v16 sort would favor 'fa' by OTO count (2 < 6 < 50).
    return [DiagnosticItem('f','a','fa',family,'internal',counts.get(('f','a'),0),1),
            DiagnosticItem('f','ei','fei',family,'internal',counts.get(('f','ei'),0),1),
            DiagnosticItem('f','u','fu',family,'internal',counts.get(('f','u'),0),1)]


def ev():
    return {'oto_rows': {('f','a'):2,('f','ei'):6,('f','u'):50},
            'distinct_pcm':{('f','a'):2,('f','ei'):3,('f','u'):1},
            'distinct_wav_paths':{('f','a'):2,('f','ei'):3,('f','u'):1},
            'oto_directories_with_audio':{('f','a'):2,('f','ei'):2,('f','u'):1},
            'unreadable_or_missing_wav_count':0,'oto_parse_warning_count':0,
            'fingerprinted_source_wav_paths':6}


def test_prefers_fewer_distinct_pcm_even_if_oto_count_large(monkeypatch):
    monkeypatch.setattr(mod,'_candidates',fake_candidates)
    chosen,_=mod.select_source_aware_items(plan(count=2),ev())
    assert [x.syllable for x in chosen.items]==['fu','fa']


def test_scheduled_line_consumes_slot_not_audio_coverage(monkeypatch):
    monkeypatch.setattr(mod,'_candidates',fake_candidates)
    chosen,scheduled=mod.select_source_aware_items(plan(count=2),ev(),['a_fu_a'])
    assert [x.syllable for x in chosen.items]==['fa']
    assert scheduled[0]['status']=='existing_script_line_not_verified_audio'


def test_two_preplanned_lines_do_not_create_unnecessary_new_tasks(monkeypatch):
    monkeypatch.setattr(mod,'_candidates',fake_candidates)
    chosen,scheduled=mod.select_source_aware_items(plan(count=2),ev(),['a_fu_a','a_fa_a'])
    assert not chosen.items and len(scheduled)==2


def test_planned_replicate_uses_next_carrier(monkeypatch):
    monkeypatch.setattr(mod,'_candidates',lambda req,counts,family:[fake_candidates(req,counts,family)[0]])
    chosen,scheduled=mod.select_source_aware_items(plan(count=2),ev(),['a_fa_a'])
    assert len(scheduled)==1
    assert [(x.syllable,x.replicate) for x in chosen.items]==[('fa',2)]


def test_report_keeps_source_counts_and_budget(monkeypatch):
    monkeypatch.setattr(mod,'_candidates',fake_candidates)
    p=EvidenceCompletionPlan(gaps=(gap('f',('other',)),))
    r=mod.source_aware_plan(p,plan(count=2),ev(),max_lines=1,existing_lines=['a_fu_a'])
    assert r['selected_count']==1
    assert r['recording_lines']==['a_fa_a']
    assert r['already_scheduled_count']==1
    assert r['selected'][0]['existing_distinct_pcm_sources']==2
    assert r['no_new_synthesis_units_authorized'] is True
    assert r['diagnostic_slots_requested']==2


def test_no_coverage_gap_means_zero_new_recording(monkeypatch):
    monkeypatch.setattr(mod,'_candidates',fake_candidates)
    r=mod.source_aware_plan(EvidenceCompletionPlan(gaps=()),plan(count=1),ev())
    assert r['selected_count']==0


def test_perceptual_only_never_recording(monkeypatch):
    monkeypatch.setattr(mod,'_candidates',fake_candidates)
    p=EvidenceCompletionPlan(gaps=(gap('f',('other',),action='perceptual_validation'),))
    r=mod.source_aware_plan(p,plan(count=1),ev())
    assert not r['recording_lines']


def test_no_legal_candidates_unfilled_not_fabricated(monkeypatch):
    monkeypatch.setattr(mod,'_candidates',lambda *args:[])
    chosen,pre=mod.select_source_aware_items(plan(count=2),ev())
    assert not chosen.items and not pre
    assert chosen.unfilled and 'available=0' in chosen.unfilled[0]


def test_missing_audio_not_counted_as_existing_evidence(monkeypatch,tmp_path):
    monkeypatch.setattr(mod,'load_voicebank',lambda root:([],[]))
    missing=tmp_path/'absent.wav'
    monkeypatch.setattr(mod,'collect_observations',lambda *args:[Observation('f','a',Entry(missing,tmp_path/'oto.ini'))])
    r=mod.inspect_recording_sources(tmp_path)
    assert r['oto_rows'][('f','a')]==1
    assert r['distinct_pcm'].get(('f','a'),0)==0
    assert r['unreadable_or_missing_wav_count']==1


def test_cli_output_is_preview_only_and_refuses_overwrite(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(mod,'_candidates',fake_candidates)
    monkeypatch.setattr(mod,'plan_voicebank',lambda *args,**kwargs:mod.source_aware_plan(EvidenceCompletionPlan(gaps=(gap('f',('other',)),)),plan(),ev(),max_lines=kwargs['max_lines']))
    assert mod.main([str(tmp_path),'--max-lines','1'])==0
    assert not list(tmp_path.iterdir())
    out=tmp_path/'new.txt';report=tmp_path/'report.json'
    assert mod.main([str(tmp_path),'--max-lines','1','--write-reclist',str(out),'--write-report',str(report)])==0
    assert out.read_text()=='a_fu_a\n'
    assert json.loads(report.read_text())['schema']==mod.SCHEMA
    with pytest.raises(SystemExit): mod.main([str(tmp_path),'--write-reclist',str(out)])


@pytest.mark.parametrize('budget',[-1,10001])
def test_invalid_budget_fails_closed(monkeypatch,budget,tmp_path):
    with pytest.raises(SystemExit):mod.main([str(tmp_path),'--max-lines',str(budget)])