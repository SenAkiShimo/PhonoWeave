"""Smoke tests for boundary-contrast pilot and anti-leakage donor rules."""
from pathlib import Path
import numpy as np
import pytest

from phonoweave.audio import AudioSegment
from phonoweave.synthesis_relevance_cv_pilot import (
    WindowSample, _donors, evaluate_voicebank, WINDOW_MS,
)

def sample(identity, wav, context, *, subbank="A", rate=44100, pre=100.0):
    wave=np.sin(2*np.pi*0.003*np.arange(1103)).astype(float)
    seg=AudioSegment(wave,rate,0.,WINDOW_MS)
    return WindowSample(
        identity=identity,source_wav=wav,subbank=subbank,alias="sh"+context,
        base="sh",context=context,sample_rate=rate,preutterance_ms=pre,
        left=seg,right=seg
    )

def test_donor_matching_excludes_same_wav_and_other_subbanks():
    target=sample("target","one.wav","rounded")
    donor=sample("other","two.wav","plain",pre=103.)
    unwanted=[sample("same","one.wav","plain"),
              sample("bank","three.wav","plain",subbank="B"),
              sample("rate","four.wav","plain",rate=48000)]
    assert _donors(target,[target,donor,*unwanted],"plain")==[donor]

def test_donor_order_uses_oto_alignment_not_audio_penalty():
    target=sample("target","one.wav","rounded",pre=100.)
    closest=sample("closest","a.wav","plain",pre=101.)
    distant=sample("distant","b.wav","plain",pre=145.)
    assert _donors(target,[distant,closest],"plain")==[closest,distant]

def test_invalid_consonant_rejected(tmp_path:Path):
    with pytest.raises(ValueError,match="only supports"):
        evaluate_voicebank(tmp_path,base="b")

def test_empty_bank_retains_explicit_no_evidence(tmp_path:Path):
    result=evaluate_voicebank(tmp_path)
    assert result["schema"].startswith("phonoweave.synthesis_relevance")
    assert all(r["targets"]==0 for r in result["context_summary"])
    assert result["profile_or_inventory_modified"] is False
