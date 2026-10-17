import pytest
from phonoweave.recording_scheme_action_plan import build_action_plan, action_plan_markdown


def _cell(uid, pitch, action):
    return {"unit_id":uid,"pitch_group":pitch,
            "recommended_review_action":action}


def _queue(uid,pitch,action,syllables="",priority=2):
    return {"unit_id":uid,"pitch_group":pitch,"base_unit":uid.split("_")[0],
            "review_action":action,"suggested_syllables":syllables,
            "priority":priority}


def test_three_pitch_repetition_collapses_without_losing_pitch_list():
    action="review_role_scoped_mapping"
    pitches=["A#3","D#4","G#4"]
    plan=build_action_plan("s",[_cell("b_internal_u",p,action) for p in pitches],
                           [_queue("b_internal_u",p,action) for p in pitches],
                           [],[])
    assert plan["consolidated_tasks"]==1
    assert plan["source_review_rows"]==3
    assert plan["tasks"][0]["pitch_groups"]==pitches
    assert plan["recordings_automatically_authorized"]==0
    assert "b_internal_u" in action_plan_markdown(plan)


def test_internal_evidence_keeps_example_provenance_and_optional_lines():
    action="review_internal_candidate_mapping"
    link={"unit_id":"d_internal_i","pitch_group":"A#3",
          "source_oto":"A#3/oto.ini","line_number":42,
          "source_wav":"A#3/_a_diA#3.wav","original_alias":"a dA#3",
          "syllable":"di","review_status":"candidate_link_not_synthesis_validated"}
    plan=build_action_plan("s",[_cell("d_internal_i","A#3",action)],
                           [_queue("d_internal_i","A#3",action,"di")],[link],
                           [{"syllable":"di","base_unit":"d",
                             "context_family":"i_series","role_scope":"internal",
                             "existing_observations":1,
                             "recording_line":"a_di_a"}])
    assert plan["tasks"][0]["source_examples"][0]["line_number"]==42
    assert plan["optional_diagnostic_lines"]==1
    assert "a_di_a" in action_plan_markdown(plan)


def test_mismatched_review_action_is_rejected():
    with pytest.raises(ValueError,match="agree"):
        build_action_plan("s",[_cell("x","A#3","review_context_mapping")],
                          [_queue("x","A#3","review_role_scoped_mapping")],[],[])
