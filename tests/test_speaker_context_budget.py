import pytest
from phonoweave.speaker_context_budget import compile_speaker_context_budget

def fixture():
    pitches=["A#3","D#4"]
    names=["a_ba_pa","a_ba_ma","o_bo_po","u_bu_pu"]
    rows=[{"sequence":name,"status":"all_pitch_groups_observed",
           "pitch_groups_observed":pitches,"pitch_groups_not_observed":[],
           "source_wavs":[{"pitch_group":p,"source_wav":f"{p}/_{name}{p}.wav",
                           "oto_files":[f"{p}/oto.ini"]} for p in pitches]}
          for name in names]
    blueprint={"schema":"phonoweave.reference_reclist_blueprint.v0.1",
               "pitch_groups":pitches,"sequences":rows,"interpretable_sequences":4,
               "all_pitch_reference_sequences":4,"reference_reclist":names}
    units=[{"unit_id":"b_internal_other","base_unit":"b",
            "contexts":"internal:other","decision":"split_recommended",
            "confidence":"moderate","synthesis_evidence":"supported_under_proxy"},
           {"unit_id":"b_internal_u","base_unit":"b",
            "contexts":"internal:u_series","decision":"unresolved",
            "confidence":"low","synthesis_evidence":"not_tested"}]
    link={"unit_id":"b_internal_other","base_unit":"b","context":"internal:other",
          "pitch_group":"A#3","source_oto":"A#3/oto.ini",
          "source_wav":"A#3/_a_ba_paA#3.wav","line_number":"8",
          "boundary_order":"1","syllable":"ba","previous_syllable":"a",
          "next_syllable":"ba","evidence":"ordered_reclist_filename_and_oto_boundary",
          "review_status":"candidate_link_not_synthesis_validated"}
    return blueprint,units,[link]

def test_budget_and_approval_guardrails():
    b,u,l=fixture()
    result=compile_speaker_context_budget(b,u,l,3,repeat_target=2)
    assert result["selection"]["selected_line_count"]<=3
    assert not result["production_ready"]
    assert result["recordings_authorized"]==0
    assert "b_internal_u" in result["unresolved_or_not_proxy_supported_units_not_optimized"]
    assert result["protected_observable_unit_pitch_cells"]==1
    assert result["accepted_internal_candidate_links"]==1

def test_corrupt_internal_link_rejected():
    b,u,l=fixture()
    l[0]["syllable"]="pa"
    with pytest.raises(ValueError,match="boundary"):
        compile_speaker_context_budget(b,u,l,2)

def test_deterministic_and_budget_zero():
    b,u,l=fixture()
    a=compile_speaker_context_budget(b,u,l,0)
    assert a["selection"]["selected_lines"]==[]
    assert a==compile_speaker_context_budget(b,u,l,0)
