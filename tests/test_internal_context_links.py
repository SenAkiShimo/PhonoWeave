"""Conservative internal-context inference from ordered real-style CVVC."""
from dataclasses import dataclass
from phonoweave.internal_context_links import link_internal_contexts


@dataclass(frozen=True)
class Unit:
    id: str
    base_unit: str
    contexts: tuple[str, ...]


def event(i, role, alias, right, offset, *, left="", wav="A#3/_a_ba_paA#3.wav"):
    origin={"source_oto":"A#3/oto.ini","line_number":i,"original_alias":alias}
    src={**origin,"source_wav":wav}
    role_row={**origin,"role":role,"right_token":right,
              "left_token":left,"pitch_suffix":"A#3",
              "pitch_suffix_status":"matched"}
    oto={**origin,"offset_ms":offset}
    comp={**origin,"compile_status":"roundtrip_review_only"}
    return src,role_row,oto,comp


def test_aligned_reclist_creates_candidate_not_production_mapping():
    events=[
        event(1,"initial_vowel","- aA#3","a",100),
        event(2,"vc_transition","a bA#3","b",300,left="a"),
        event(3,"vc_transition","a pA#3","p",600,left="a"),
    ]
    u=[Unit("b_internal_other","b",("internal:other",)),
       Unit("p_internal_other","p",("internal:other",))]
    rows,info=link_internal_contexts(u,*[[x[i] for x in events] for i in range(4)])
    assert len(rows)==2
    assert {x["unit_id"] for x in rows}=={"b_internal_other","p_internal_other"}
    assert info["production_mapping"]=="disabled"
    assert info["linked_unit_pitch_cells"]==2


def test_bad_sequence_abstains():
    events=[
        event(1,"initial_vowel","- aA#3","a",100),
        event(2,"vc_transition","a bA#3","b",300,left="a",
              wav="A#3/_a_ba_paA#3.wav"),
    ]
    rows,info=link_internal_contexts(
        [Unit("b_internal_other","b",("internal:other",))],
        *[[x[i] for x in events] for i in range(4)])
    assert rows==[]
    assert info["abstained_groups"]["boundary_count_mismatch"]==1


def test_wrong_provenance_rejected():
    events=[event(1,"initial_vowel","- aA#3","a",100)]
    arrays=[[x[i] for x in events] for i in range(4)]
    arrays[2][0]["line_number"]=99
    import pytest
    with pytest.raises(ValueError,match="provenance"):
        link_internal_contexts([], *arrays)
