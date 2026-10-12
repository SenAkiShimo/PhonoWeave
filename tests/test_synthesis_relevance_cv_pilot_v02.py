"""Regression tests: correct OTO duration, stratified targets, donor WAV independence."""
from pathlib import Path
import numpy as np
import pytest
from phonoweave.audio import AudioSegment
from phonoweave.synthesis_relevance_cv_pilot_v02 import (
    CVEvent,_donors,_balanced_targets,evaluate_voicebank,
)

def ev(identifier, wav, subbank, context, duration=100.0, alignment=30000.0):
    x=np.zeros(1103)
    seg=AudioSegment(x,44100,0,25)
    return CVEvent(identifier,wav,subbank,identifier,"sh",context,44100,duration,alignment,seg,seg)

def test_donors_match_relative_oto_not_absolute_wav_location():
    target=ev("target","target.wav","D4","plain",duration=100,alignment=60000)
    a=ev("near","a.wav","D4","rounded",duration=101,alignment=30000)
    b=ev("far","b.wav","D4","rounded",duration=140,alignment=60000)
    assert _donors(target,[target,b,a],"rounded")==[a,b]

def test_one_donor_per_physical_wav_and_other_subbank_excluded():
    target=ev("target","target.wav","D4","plain")
    repeated=ev("dup","dup.wav","D4","rounded",duration=101)
    repeated2=ev("dup2","dup.wav","D4","rounded",duration=102)
    other=ev("other","other.wav","D4","rounded",duration=103)
    bank=ev("bank","bank.wav","D5","rounded",duration=100)
    same=ev("same","target.wav","D4","rounded",duration=100)
    chosen=_donors(target,[repeated,repeated2,other,bank,same],"rounded")
    assert [x.source_wav for x in chosen]==["dup.wav","other.wav"]

def test_round_robin_target_subbanks():
    arr=[ev(f"A{i}",f"a{i}.wav","A","plain") for i in range(8)]
    arr += [ev(f"B{i}",f"b{i}.wav","B","plain") for i in range(5)]
    sampled=_balanced_targets(arr,"plain",6)
    assert len(sampled)==6
    assert sum(x.subbank=="A" for x in sampled)==3
    assert sum(x.subbank=="B" for x in sampled)==3

def test_empty_and_invalid_bank(tmp_path:Path):
    result=evaluate_voicebank(tmp_path,base="sh")
    assert result["schema"].endswith("v0.2")
    assert result["targets"]==[]
    assert result["context_summary"][0]["targets"]==0
    assert result["speaker_profile_modified"] is False
    with pytest.raises(ValueError,match="supports"):
        evaluate_voicebank(tmp_path,base="b")
