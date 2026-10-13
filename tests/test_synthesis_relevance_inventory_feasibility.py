"""Inventory counting, mutually-exclusive barriers and lower-bound tests."""
from phonoweave.synthesis_relevance_unit_merging_cost import Unit
from phonoweave.synthesis_relevance_inventory_feasibility import feasibility,SCHEMA

def u(wav,final,ctx,dt=100,group="G4"):
    x=(0.5,)*48
    return Unit(group,wav,ctx,final,dt,x,x,0.,0.)

def status(r,name):
    return r["mutually_exclusive_target_statuses"].get(name,0)

def test_empty_inventory():
    r=feasibility([])
    assert r["strict_trial_replay"]==0
    assert r["physical_WAVs"]==0
    assert r["subbanks"]==[]

def test_single_wav_with_both_finals_counts_as_one_replication():
    r=feasibility([u("w","a","plain"),u("w","ua","rounded")])
    pair=r["subbanks"][0]["pairs"][0]
    assert pair["plain_WAVs"]==1 and pair["rounded_WAVs"]==1
    assert pair["co_recorded_WAVs"]==1
    assert pair["minimum_additional_final_recordings_for_two_per_side"]==2
    assert pair["lower_bound_additional_WAVs_if_both_finals_can_share_each_new_WAV"]==1
    assert status(r,"missing_independent_same_final")==2
    assert r["strict_trial_replay"]==0

def test_repeated_oto_events_are_not_independent_wavs():
    r=feasibility([u("p","a","plain"),u("p","a","plain"),
                   u("r","ua","rounded")])
    p=r["subbanks"][0]["finals"][0]
    assert p["physical_WAVs"]==1 and p["oto_events"]==2
    assert p["aliased_or_repeated_events_in_same_WAV"]==1

def test_structural_candidates_can_fail_duration_caliper():
    r=feasibility([u("p1","a","plain",100),u("p2","a","plain",140),
                   u("r1","ua","rounded",100),u("r2","ua","rounded",140)])
    pair=r["subbanks"][0]["pairs"][0]
    assert pair["bidirectional_two_WAV_per_final_feasible"]
    assert status(r,"same_final_duration_caliper_failure")==4
    assert r["strict_trial_replay"]==0

def test_complete_candidate_matches_frozen_strict_protocol():
    r=feasibility([u("p1","a","plain",100),u("p2","a","plain",101),
                   u("r1","ua","rounded",102),u("r2","ua","rounded",103)])
    assert r["strict_trial_replay"]==4
    assert status(r,"eligible")==4
    assert r["aggregate_counts"]["final_recording_deficit_two_each"]==0
    assert SCHEMA.endswith("v1.0")
