"""Prototype reclist compiler: contract and optimization tests."""
from __future__ import annotations
from phonoweave.adaptive_reclist_prototype_v50 import _minimum_cover, _ari, compile_scheme


def test_set_cover_observed_cv_not_arbitrary_combination():
    edges=[("m","a"),("m","i"),("sh","a")]
    needed=["C:m","C:sh","V:a","V:i"]
    covered={(c,v):{"C:"+c,"V:"+v} for c,v in edges}
    result=_minimum_cover(edges,needed,covered)
    assert result=={("m","i"),("sh","a")}


def test_context_partition_ari_is_label_permutation_invariant():
    assert abs(_ari([["a","i"],["u","e"]],[["u","e"],["a","i"]])-1.0)<1e-8


def test_synthetic_proxy_generates_real_recording_lines_without_vc_supervision():
    banks=["B1","B2","B3","B4"]
    entries=[]
    # one modest, perfectly stable C:m contrast across vowels
    for role,symbol,contexts in (("C","m",["a","i"]),
                                 ("C","sh",["a","i"]),
                                 ("V","a",["m","sh"]),
                                 ("V","i",["m","sh"])):
        fold=[{"heldout_bank":bank,"k_results":[
            {"k":1,"training_banks_only_context_groups":[contexts]},
            {"k":2,"training_banks_only_context_groups":[[contexts[0]],[contexts[1]]]},
        ]} for bank in banks]
        entries.append({"phoneme_role":role,"author_phoneme":symbol,
                        "common_contexts":contexts,"banks":banks,
                        "results":[
                            {"k":1,"bank_heldout_transductive_mse":1.0,
                             "per_bank_heldout_mse":{bank:1.0 for bank in banks}},
                            {"k":2,"k1_minus_k_mse":0.2,
                             "per_bank_heldout_mse":{bank:0.8 for bank in banks}},
                        ],"fold_details":fold})
    result=compile_scheme({"schema":"phonoweave.cv_oto_bank_stability.v4.2",
                           "phonemes_screened":4,"phonemes_with_proxy_scores":4,
                           "items":entries})
    assert result["author_C_coverage"]==2
    assert result["author_V_coverage"]==2
    assert result["total_recording_tokens"]==2
    assert {r["recording_token"] for r in result["recording_lines"]} <= {"ma","mi","sha","shi"}
    assert result["source_summary"]["VC_existing_unit_retain_policy"].startswith("KEEP_ALL")
    assert all(r["phase"]=="CORE" for r in result["recording_lines"])


def test_invalid_schema_fails_closed():
    import pytest
    with pytest.raises(ValueError,match="expected PhonoWeave"):
        compile_scheme({"schema":"wrong","items":[]})
