import copy
import json
import pytest

from phonoweave.recording_scheme_decisions import (
    make_decisions_template, compile_decisions, save_review_template,
    write_reviewed_scheme,
)

def plan():
    return {
        "schema":"phonoweave.recording_scheme_action_plan.v0.1",
        "speaker_id":"s", "consolidated_tasks":2, "source_review_rows":3,
        "tasks":[
            {"task_id":"RS-001","review_action":"review_cross_pitch_coverage",
             "pitch_groups":["A#3"],"unit_id":"c_plain","base_unit":"c",
             "source_examples":[]},
            {"task_id":"RS-002","review_action":"review_internal_candidate_mapping",
             "pitch_groups":["A#3","D#4"],"unit_id":"p_internal_i",
             "base_unit":"p",
             "source_examples":[
                 {"pitch_group":"A#3","source_oto":"A#3/oto.ini","line_number":10},
                 {"pitch_group":"D#4","source_oto":"D#4/oto.ini","line_number":11},
             ]},
        ],
        "optional_diagnostics":[{"recording_line":"a_pi_a","syllable":"pi",
                                 "base_unit":"p"}],
        "optional_diagnostic_lines":1,
    }

def test_default_review_cannot_approve_anything():
    p=plan()
    template=make_decisions_template(p)
    result=compile_decisions(p,template)
    assert result["pending_pitch_cells"]==3
    assert result["approved_diagnostic_lines"]==0
    assert result["production_alias_export"]=="disabled"

def test_explicit_human_review_compiles_one_recording(tmp_path):
    p=plan()
    template=make_decisions_template(p)
    template["reviewer_id"]="reviewer-1"
    row=template["task_reviews"][1]
    row.update(resolution="candidate_accepted",note="Verified against WAV",
               evidence_reference="A#3/oto.ini:10")
    template["diagnostic_reviews"][0].update(
        decision="approve",related_task_id="RS-002",
        note="Additional diagnostic independent take approved")
    result=compile_decisions(p,template)
    assert result["reviewed_pitch_cells"]==1
    assert result["approved_diagnostic_lines"]==1
    out=tmp_path/"reviewed"
    write_reviewed_scheme(result,out)
    assert (out/"approved_diagnostic_reclist.txt").read_text()=="a_pi_a\n"
    with pytest.raises(FileExistsError):
        write_reviewed_scheme(result,out)

def test_forged_or_stale_review_fails():
    p=plan()
    template=make_decisions_template(p)
    altered=copy.deepcopy(p)
    altered["tasks"][0]["unit_id"]="changed"
    with pytest.raises(ValueError,match="match"):
        compile_decisions(altered,template)
    template["task_reviews"][0]["pitch_group"]="G#4"
    with pytest.raises(ValueError,match="unknown or duplicate"):
        compile_decisions(p,template)

def test_resolution_evidence_and_reviewer_required():
    p=plan()
    template=make_decisions_template(p)
    template["task_reviews"][0]["resolution"]="missing_verified"
    with pytest.raises(ValueError,match="note and evidence"):
        compile_decisions(p,template)
    template["task_reviews"][0].update(
        note="Inspected source",evidence_reference="D#4/oto.ini:21")
    with pytest.raises(ValueError,match="reviewer_id"):
        compile_decisions(p,template)

def test_unresolved_task_cannot_approve_diagnostic():
    p=plan()
    template=make_decisions_template(p)
    template["reviewer_id"]="reviewer-1"
    template["diagnostic_reviews"][0].update(
        decision="approve",related_task_id="RS-002",note="propose take")
    with pytest.raises(ValueError,match="reviewed task evidence"):
        compile_decisions(p,template)

def test_template_non_overwrite(tmp_path):
    path=tmp_path/"reviews.json"
    save_review_template(plan(),path)
    with pytest.raises(FileExistsError):
        save_review_template(plan(),path)
    assert json.loads(path.read_text())["task_reviews"][0]["resolution"]=="pending"
