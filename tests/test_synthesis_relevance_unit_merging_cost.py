"""Counterfactual donor matching and WAV-level inference tests."""
import numpy as np
from pathlib import Path
from phonoweave.synthesis_relevance_unit_merging_cost import (
    Unit,_signature,_distance,_penalty,trial_rows,paired_stats,
    extract,run_manifest,SCHEMA
)

def unit(wav,context,value,subbank="D4",duration=140):
    core=tuple([value]*48)
    late=tuple([0.]*48)
    return Unit(subbank,wav,context,"u" if context=="rounded" else "i",
                duration,core,late,0.,0.)

def test_signature_finite_and_distance_symmetry():
    x=np.random.default_rng(42).normal(0.,0.1,44100)
    a,db=_signature(x,44100,100.)
    b,_=_signature(x,44100,120.)
    assert len(a)==48 and np.isfinite(db)
    assert _distance(a,b)==_distance(b,a)
    assert _distance(a,a)==0

def test_same_context_reconstruction_outperforms_cross_on_known_signal():
    units=[
        unit(f"p{i}","plain",0.2+0.005*i) for i in range(10)
    ]+[
        unit(f"r{i}","rounded",1.5+0.005*i) for i in range(10)
    ]
    trials=trial_rows(units)
    assert len(trials)==20
    assert all(t["donors_per_arm"]==3 for t in trials)
    assert all(t["delta_reconstruction"]>0 for t in trials)
    s=paired_stats(trials,"delta_reconstruction",iterations=199)
    assert s["status"]=="evaluated" and s["two_sided_WAV_sign_flip_p"]<=.05

def test_no_self_donor_no_cross_subbank_and_not_enough_controls():
    units=[unit("a","plain",0),unit("b","rounded",1)]
    assert trial_rows(units)==[]
    many=[unit(f"p{i}","plain",0,subbank="A3") for i in range(6)]
    many += [unit(f"r{i}","rounded",1,subbank="D5") for i in range(6)]
    assert trial_rows(many)==[]

def test_physical_wav_not_pseudoreplicated():
    trials=[{"wav_id":"A","delta_reconstruction":.2} for _ in range(50)]
    trials += [{"wav_id":f"x{i}","delta_reconstruction":.1} for i in range(5)]
    s=paired_stats(trials,"delta_reconstruction",iterations=99)
    assert s["independent_wavs"]==6

def test_empty_bank_has_explicit_insufficient_status(tmp_path):
    (tmp_path/"oto.ini").write_text("",encoding="utf-8")
    units,meta=extract(tmp_path)
    assert not units and meta["valid_units"]==0
    assert paired_stats([],"delta_seam")["status"]=="insufficient"
    assert SCHEMA.endswith("v0.8")
