import pytest
from phonoweave.reference_reclist_blueprint import compile_reference_reclist, write_reference_reclist

def entry(pitch, sequence, number=1, role="initial_cv"):
    wav=f"{pitch}/_{sequence}{pitch}.wav"
    alias=f"- ba{pitch}"
    return {"schema":"phonoweave.lossless_alias_record.v0.1",
            "source":{"source_oto":f"{pitch}/oto.ini","line_number":number,
                      "original_alias":alias,"source_wav":wav},
            "role":{"source_oto":f"{pitch}/oto.ini","line_number":number,
                    "original_alias":alias,"pitch_suffix":pitch,
                    "pitch_suffix_status":"matched","role":role}}

def test_three_pitch_reference_list_and_source_count(tmp_path):
    records=[entry(p,"a_ba_pa") for p in ("A#3","D#4","G#4")]
    report=compile_reference_reclist(records)
    assert report["unique_source_wavs"]==3
    assert report["all_pitch_reference_sequences"]==1
    assert report["reference_reclist"]==["a_ba_pa"]
    out=tmp_path/"export"
    write_reference_reclist(report,out)
    assert (out/"reference_reclist.txt").read_text()=="a_ba_pa\n"
    with pytest.raises(FileExistsError):
        write_reference_reclist(report,out)

def test_partial_pitch_is_not_presented_as_complete():
    data=[entry("A#3","a_ba"),entry("D#4","a_ba"),
          entry("G#4","a_pa")]
    result=compile_reference_reclist(data)
    assert result["all_pitch_reference_sequences"]==0
    assert result["reference_reclist"]==[]
    assert all(x["status"]=="some_pitch_groups_not_observed"
               for x in result["sequences"])

def test_duplicate_sequence_pitch_is_ambiguous():
    a=entry("A#3","a_ba")
    b=entry("A#3","a_ba",number=2)
    b["source"]["source_wav"]="A#3/sub/_a_baA#3.wav"
    result=compile_reference_reclist([a,b,entry("D#4","a_ba")])
    assert result["reference_reclist"]==[]
    assert result["sequences"][0]["status"]=="ambiguous_duplicate_source"

def test_provenance_mismatch_refused():
    a=entry("A#3","a_ba")
    a["role"]["line_number"]=100
    with pytest.raises(ValueError,match="provenance"):
        compile_reference_reclist([a])
