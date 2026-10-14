"""Budget parity, WAV group isolation, train-only decision and abstention."""
from phonoweave.synthesis_relevance_unit_merging_cost import Unit
from phonoweave.synthesis_relevance_budget_benchmark import (
    allocate,evaluate_bank,PAIR_NAMES,SCHEMA
)

def sample(wav,final,context,core,subbank="D4"):
    x=tuple(float(core) for _ in range(48))
    return Unit(subbank,wav,context,final,110.,x,x,0.,0.)

def inventory(n=7):
    final_pairs=(("a","ua"),("ai","uai"),("an","uan"),("ang","uang"))
    units=[]
    for j,(a,b) in enumerate(final_pairs):
        for i in range(n):
            units.append(sample(f"{j}-plain-{i}",a,"plain",j+0.01*i))
            units.append(sample(f"{j}-round-{i}",b,"rounded",j+1+0.01*i))
    return units

def test_empty_inventory_abstains():
    result=evaluate_bank([])
    assert result["status"]=="insufficient"
    assert result["budget_curves"]==[]

def test_holdout_wav_disjoint_for_every_final():
    parts=allocate(inventory())
    assert len(parts)==4
    assert all(p["status"]=="evaluable" for p in parts)
    for p in parts:
        train={u.wav_id for rows in p["train"].values() for u in rows}
        test={u.wav_id for rows in p["test"].values() for u in rows}
        assert train and test and train.isdisjoint(test)

def test_budget_parity_and_equal_eval_population():
    r=evaluate_bank(inventory())
    assert r["status"]=="evaluable"
    assert r["complete_four_pair_subbanks"]==1
    assert len(r["budget_curves"])==15
    for budget in range(5):
        selected=[row for row in r["budget_curves"] if row["budget"]==budget]
        assert len(selected)==3
        assert len({row["test_WAVs"] for row in selected})==1
        assert all(len(row["selected_split_pairs"])==budget for row in selected)
    b0=[row["mean_heldout_log_spectral_distance"] for row in r["budget_curves"] if row["budget"]==0]
    b4=[row["mean_heldout_log_spectral_distance"] for row in r["budget_curves"] if row["budget"]==4]
    assert len(set(b0))==len(set(b4))==1

def test_less_than_two_train_wavs_abstains():
    # Two physical source WAVs per final are insufficient for the
    # prespecified 2-train / 1-test protocol, even before scoring.
    r=evaluate_bank(inventory(2))
    assert r["status"]=="insufficient"
    assert r["budget_curves"]==[]
    assert SCHEMA.endswith("v1.1")
