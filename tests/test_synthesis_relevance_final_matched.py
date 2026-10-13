"""Test strict matched-final counterfactual and unavailable-comparison audit."""
from pathlib import Path
import json
from phonoweave.synthesis_relevance_unit_merging_cost import Unit
from phonoweave.synthesis_relevance_final_matched import (
    matched_trials,co_recorded_pairs,analyze_bank,FINAL_PAIRS,SCHEMA
)

def u(wav,final,ctx,duration=100,group="D4",v=0.5):
    x=tuple([v]*48)
    return Unit(group,wav,ctx,final,duration,x,x,0.,0.)

def test_matched_pair_requires_distinct_source_wav_and_duration_balance():
    units=[
        u("target","a","plain",100),
        u("same","a","plain",101),
        u("cross","ua","rounded",102),
    ]
    trials,miss=matched_trials(units)
    t=next(r for r in trials if r["wav_id"]=="target")
    assert t["same_donor_wav_id"]=="same"
    assert t["cross_donor_wav_id"]=="cross"
    assert t["pair"]=="a/ua"
    assert t["delta_reconstruction"]==0

def test_matching_excludes_other_vowel_family_and_other_subbank():
    units=[u("a","a","plain"),u("b","an","plain"),
           u("c","ua","rounded"),u("d","a","plain",group="D5")]
    trials,miss=matched_trials(units)
    assert not trials
    assert miss["no_different_WAV_same_final_within_10ms"]>=1

def test_identical_source_is_not_a_valid_counterfactual_donor():
    units=[u("same","a","plain"),u("same","ua","rounded")]
    trials,_=matched_trials(units)
    assert not trials
    assert len(co_recorded_pairs(units))==1

def test_duration_balance_rejects_imbalanced_target_not_valid_reverse_target():
    units=[u("target","a","plain",100),u("same","a","plain",90),
           u("other","ua","rounded",100)]
    trials,miss=matched_trials(units)
    # target@100: same-final donor is 10 ms away while cross donor is 0 ms
    # away. Its mismatch gap exceeds the prespecified 5-ms arm balance.
    assert not any(t["wav_id"]=="target" for t in trials)
    assert miss["arm_duration_balance_exceeds_5ms"]>=1
    # same@90 is a separate eligible target: both candidates are 10 ms
    # away. A rejected target must not invalidate another valid trial.
    reverse=next(t for t in trials if t["wav_id"]=="same")
    assert reverse["same_duration_mismatch_ms"]==10
    assert reverse["cross_duration_mismatch_ms"]==10

def test_sparse_empty_bank_is_unevaluable(tmp_path):
    (tmp_path/"oto.ini").write_text("",encoding="utf-8")
    result=analyze_bank(tmp_path)
    assert result["strict_overall"]["reconstruction"]["status"]=="insufficient"
    assert result["strict_target_events"]==0

def test_frozen_pairs():
    assert FINAL_PAIRS[:2]==(("a","ua"),("ai","uai"))
    assert SCHEMA.endswith("v0.9")
