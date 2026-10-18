import pytest
from phonoweave.recording_scheme_budget import compile_recording_budget, write_recording_budget

def blueprint():
    pitches=["A#3","D#4","G#4"]
    names=["a_ba_pa","a_ba_ma","o_bo_po"]
    rows=[]
    for name in names:
        rows.append({
            "sequence":name,"status":"all_pitch_groups_observed",
            "pitch_groups_observed":pitches,
            "pitch_groups_not_observed":[],
            "source_wavs":[{"pitch_group":pitch,
                            "source_wav":f"{pitch}/_{name}{pitch}.wav",
                            "oto_files":[f"{pitch}/oto.ini"]}
                           for pitch in pitches],
        })
    return {"schema":"phonoweave.reference_reclist_blueprint.v0.1",
            "pitch_groups":pitches,"sequences":rows,
            "interpretable_sequences":3,"all_pitch_reference_sequences":3,
            "reference_reclist":names,"excluded_wav_groups":{}}

def test_budget_zero_and_three_coverages():
    source=blueprint()
    empty=compile_recording_budget(source,0)
    assert empty["selected_reclist"]==[]
    full=compile_recording_budget(source,3)
    assert full["feature_universe_sizes"]["syllables"]==7
    assert full["feature_universe_sizes"]["exact_adjacent_pairs"]==4
    for mode in ("syllables","syllables_and_starts","exact_adjacent_pairs"):
        assert full["strategies"][mode]["coverage"]["syllables"]["fraction"]==1
        assert full["strategies"][mode]["coverage"]["exact_adjacent_pairs"]["fraction"]==1

def test_budget_capped_and_deterministic():
    source=blueprint()
    a=compile_recording_budget(source,1)
    b=compile_recording_budget(source,1)
    assert a==b
    assert len(a["selected_reclist"])==1
    assert a["new_recordings_authorized"]==0
    assert not a["selection_is_production_ready"]

def test_corrupt_provenance_refused():
    source=blueprint()
    source["sequences"][0]["source_wavs"].pop()
    with pytest.raises(ValueError,match="counts inconsistent|reference reclist"):
        compile_recording_budget(source,2)

def test_write_preserves_existing_output(tmp_path):
    plan=compile_recording_budget(blueprint(),2)
    output=tmp_path/"budget"
    write_recording_budget(plan,output)
    assert (output/"pilot_reclist.txt").read_text()
    with pytest.raises(FileExistsError):
        write_recording_budget(plan,output)

def test_bad_budget_refused():
    with pytest.raises(ValueError,match="budget"):
        compile_recording_budget(blueprint(),-1)
