"""Recording-budget scheduling checks, not independent phone-boundary validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from phonoweave.diagnostic_selector import DiagnosticItem, DiagnosticSelection
from phonoweave.evidence_gap import EvidenceCompletionPlan, EvidenceGap
from phonoweave.recording_budget_v16 import budget_recording_selection, main


def gap(base: str, contexts: tuple[str, ...], *, priority='high', action='supplemental_recording'):
    return EvidenceGap(base_unit=base, class_name='fricative', gap_type='coverage_limited',
                       priority=priority, recommended_action=action, role_scope='internal',
                       context_families=contexts, rationale=())


def item(base: str, syllable: str, context: str, *, replicate=1, observed=0, role='internal'):
    return DiagnosticItem(base_unit=base, final=syllable[len(base):], syllable=syllable,
                          context_family=context, role_scope=role,
                          existing_observations=observed, replicate=replicate)


def case():
    plan=EvidenceCompletionPlan(gaps=(gap('f',('rounded','other')),gap('r',('front',),priority='medium')))
    sel=DiagnosticSelection(items=(
        item('f','fu','rounded',observed=0),
        item('f','fa','other',observed=2),
        item('f','fo','rounded',observed=3),
        item('r','ri','front',observed=0),
        item('r','ri','front',replicate=2,observed=0),
    ),unfilled=('x:other:requested=1:available=0',))
    return plan,sel


def test_bounded_cover_each_family_before_replicates():
    p,s=case()
    r=budget_recording_selection(p,s,max_lines=3)
    assert r['recording_lines']==['a_fa_a','a_fu_a','a_ri_a']
    assert r['contexts_in_selected_lines']==3
    assert r['selected_count']==3
    assert all(row['reason']=='global_line_budget_reached' for row in r['deferred_or_skipped'])


def test_same_family_second_replicate_comes_after_firsts():
    p,s=case()
    r=budget_recording_selection(p,s,max_lines=10)
    assert r['recording_lines']==['a_fa_a','a_fu_a','a_ri_a','a_fo_a','e_ri_e']
    assert len(set(r['recording_lines']))==5
    assert r['unfilled_by_original_selector']==['x:other:requested=1:available=0']


def test_zero_budget_explicitly_defers_all():
    p,s=case()
    r=budget_recording_selection(p,s,max_lines=0)
    assert r['recording_lines']==[]
    assert len(r['deferred_or_skipped'])==5
    assert r['contexts_in_selected_lines']==0


def test_existing_reclist_lines_are_not_rescheduled():
    p,s=case()
    r=budget_recording_selection(p,s,max_lines=2,existing_lines=[' # unrelated','a_fa_a','a_fu_a'])
    assert r['recording_lines']==['a_ri_a','a_fo_a']
    assert [o['reason'] for o in r['deferred_or_skipped']].count('line_already_in_supplied_reclist')==2


def test_duplicate_generated_line_never_claims_two_recordings():
    p=EvidenceCompletionPlan(gaps=(gap('f',('other',)),))
    s=DiagnosticSelection(items=(item('f','fa','other'),item('f','fa','other')),unfilled=())
    r=budget_recording_selection(p,s,max_lines=4)
    assert r['recording_lines']==['a_fa_a']
    assert r['deferred_or_skipped'][0]['reason']=='duplicate_line_in_suggestions'


def test_perceptual_validation_never_creates_recording_task():
    p=EvidenceCompletionPlan(gaps=(gap('f',('other',),action='perceptual_validation'),))
    s=DiagnosticSelection(items=(item('f','fa','other'),),unfilled=())
    r=budget_recording_selection(p,s)
    assert r['recording_lines']==[]
    assert r['auto_split_merge'] is False


def test_high_priority_before_medium_for_first_pass():
    p,s=case()
    r=budget_recording_selection(p,s,max_lines=2)
    assert set(r['recording_lines'])=={'a_fa_a','a_fu_a'}
    assert 'a_ri_a' not in r['recording_lines']


def test_no_gap_no_reclist_even_if_selection_contaminated():
    r=budget_recording_selection(EvidenceCompletionPlan(gaps=()),DiagnosticSelection(items=(item('f','fa','other'),),unfilled=()))
    assert r['selected_count']==0
    assert r['existing_voicebank_unchanged']


@pytest.mark.parametrize('invalid',[True,-1,2.5,'3',10001])
def test_invalid_budgets_fail_closed(invalid):
    p,s=case()
    with pytest.raises(ValueError,match='max_lines'):
        budget_recording_selection(p,s,max_lines=invalid)


def test_wrong_context_rejected():
    p=EvidenceCompletionPlan(gaps=(gap('f',('rounded',)),))
    s=DiagnosticSelection(items=(item('f','fa','other'),),unfilled=())
    with pytest.raises(ValueError,match='outside'):
        budget_recording_selection(p,s)


def test_invalid_observation_count_rejected():
    p,s=case()
    s=DiagnosticSelection(items=(item('f','fa','other',observed=-1),),unfilled=())
    with pytest.raises(ValueError,match='counts'):
        budget_recording_selection(p,s)


def test_report_metadata_disclaims_actual_takes():
    p,s=case();r=budget_recording_selection(p,s,max_lines=1)
    assert r['language_scope']=='existing_mandarin_syllable_selector_only'
    assert r['line_count_is_not_number_of_total_pitch_takes']
    assert r['aliases_changed'] is False
    assert r['selected'][0]['why'].startswith('diagnose_coverage_limited')


def test_cli_preview_no_output_and_writer_no_overwrite(monkeypatch,tmp_path,capsys):
    p,s=case()
    import phonoweave.recording_budget_v16 as mod
    monkeypatch.setattr(mod,'plan_voicebank',lambda *args,**kwargs:budget_recording_selection(p,s,max_lines=kwargs['max_lines']))
    out=tmp_path/'reclist.txt';report=tmp_path/'reason.json'
    assert main([str(tmp_path),'--max-lines','1'])==0
    assert not out.exists()
    assert main([str(tmp_path),'--max-lines','2','--write-reclist',str(out),'--write-report',str(report)])==0
    assert out.read_text()=='a_fa_a\na_fu_a\n'
    assert json.loads(report.read_text())['selected_count']==2
    with pytest.raises(SystemExit):
        main([str(tmp_path),'--write-reclist',str(out)])
    assert out.read_text()=='a_fa_a\na_fu_a\n'


def test_cli_json_no_ipa_or_auto_alias(monkeypatch,tmp_path,capsys):
    p,s=case()
    import phonoweave.recording_budget_v16 as mod
    monkeypatch.setattr(mod,'plan_voicebank',lambda *args,**kwargs:budget_recording_selection(p,s,max_lines=kwargs['max_lines']))
    assert main([str(tmp_path),'--max-lines','1','--json'])==0
    out=json.loads(capsys.readouterr().out)
    assert out['selected_count']==1
    assert out['aliases_changed'] is False
    assert out['mode']=='diagnostic_recording_suggestions_not_new_synthesis_units'