from dataclasses import dataclass
import pytest
from phonoweave.recording_scheme_coverage import build_recording_coverage


@dataclass(frozen=True)
class Unit:
    id: str
    base_unit: str = "sh"
    contexts: tuple = ("plain",)
    decision: str = "split_recommended"


def rec(alias, role, unit="sh_plain", pitch="A#3", line=1, wav="take.wav"):
    origin={"source_oto":f"{pitch}/oto.ini","line_number":line,
            "original_alias":alias,"source_wav":wav,
            "mapping_status":"candidate_review_only"}
    compiled={"source_oto":origin["source_oto"],"line_number":line,
              "original_alias":alias,"candidate_unit_id":unit,
              "compile_status":"roundtrip_review_only","role":role,
              "right_token":"sha","pitch_suffix":pitch}
    return origin,compiled


def test_cross_pitch_gap_without_claim_of_missing_recording():
    a,b=rec("shaA#3","standalone_cv")
    c,d=rec("- shaA#3","initial_cv",line=2)
    e,f=rec("- shaD#4","initial_cv",pitch="D#4",line=3)
    rows,actions,summary=build_recording_coverage(
        [Unit("sh_plain")],[a,c,e],[b,d,f])
    assert len(rows)==2
    gap=next(row for row in rows if row["pitch_group"]=="D#4")
    assert gap["coverage_status"]=="cross_pitch_alias_gap"
    assert gap["missing_cv_vs_other_pitches"]=="sha"
    assert gap["referenced_source_wavs"]==1
    assert summary["recording_requests_auto_authorized"]==0


def test_role_scoped_abstention_not_fake_absence():
    a,b=rec("shaA#3","standalone_cv",unit="")
    rows,actions,summary=build_recording_coverage(
        [Unit("p_internal_i",base_unit="p",contexts=("internal:i_series",))],
        [a],[b])
    assert rows[0]["coverage_status"]=="mapping_scope_unresolved"
    assert actions[0]["review_action"]=="review_role_scoped_mapping"
    assert summary["units_without_traceable_cv_links"]==1


def test_source_mismatch_and_duplicate_units_rejected():
    a,b=rec("shaA#3","standalone_cv")
    b["line_number"]=9
    with pytest.raises(ValueError,match="provenance"):
        build_recording_coverage([Unit("sh_plain")],[a],[b])
    with pytest.raises(ValueError,match="duplicate"):
        build_recording_coverage([Unit("x"),Unit("x")],[],[])


def test_internal_link_updates_queue_without_claiming_validated_mapping():
    a,b=rec("a bA#3","vc_transition",unit="",line=4)
    a["source_wav"]="A#3/_a_baA#3.wav"
    link={"source_oto":a["source_oto"],"line_number":a["line_number"],
          "source_wav":a["source_wav"],"original_alias":a["original_alias"],
          "pitch_group":"A#3","unit_id":"b_internal_other",
          "base_unit":"b","context":"internal:other","syllable":"ba",
          "review_status":"candidate_link_not_synthesis_validated"}
    rows,actions,summary=build_recording_coverage(
        [Unit("b_internal_other",base_unit="b",
              contexts=("internal:other",))],[a],[b],[link])
    assert rows[0]["coverage_status"]=="internal_candidate_evidence"
    assert rows[0]["internal_candidate_links"]==1
    assert actions[0]["review_action"]=="review_internal_candidate_mapping"
    assert summary["internal_candidate_unit_pitch_cells"]==1
    assert summary["recording_requests_auto_authorized"]==0
    corrupted=dict(link,original_alias="forged")
    with pytest.raises(ValueError,match="provenance"):
        build_recording_coverage(
            [Unit("b_internal_other",base_unit="b",
                  contexts=("internal:other",))],[a],[b],[corrupted])
