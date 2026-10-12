"""Frozen 110 ms comparator, short-window limits, and anti-leakage checks."""
import numpy as np
import pytest
from pathlib import Path
from phonoweave.synthesis_relevance_short_interval import (
    short_features, extract, MIN_MS, SCHEMA, evaluate_bank
)
from phonoweave.synthesis_relevance_acoustic_separability import _features as legacy

def test_short_interval_has_nine_finite_features():
    rng=np.random.default_rng(41)
    samples=rng.normal(0.,0.15,44100)
    for d in (40.,60.,90.,109.,130.):
        vals=short_features(samples,44100,100.,100.+d)
        assert len(vals)==9
        assert all(np.isfinite(vals))

def test_short_rejects_outside_and_very_short():
    x=np.ones(44100)
    with pytest.raises(ValueError,match="shorter_than_40ms"):
        short_features(x,44100,100.,139.)
    with pytest.raises(ValueError,match="window_outside_wav"):
        short_features(x,44100,990.,1040.)
    with pytest.raises(ValueError,match="nonfinite"):
        short_features(x,44100,float("nan"),150.)

def test_long_baseline_unchanged():
    noise=np.random.default_rng(1).normal(size=44100)*0.1
    assert len(legacy(noise,44100,100.,300.))==9
    with pytest.raises(ValueError,match="short OTO"):
        legacy(noise,44100,100.,180.)

def test_empty_voicebank_returns_insufficient(tmp_path:Path):
    tmp_path.joinpath("oto.ini").write_text("",encoding="utf-8")
    protocols,coverage=extract(tmp_path)
    assert coverage["valid_events_by_protocol"]["short_interior"]==0
    assert all(not p for p in protocols.values())
    result=evaluate_bank(tmp_path,permutations=20)
    assert result["protocols"]["short_interior"]["grouped_cv"]["status"]=="insufficient"

def test_schema_and_threshold_frozen():
    assert SCHEMA.endswith("v0.7")
    assert MIN_MS==40.0

def test_legacy_protocol_matches_independent_frozen_extractor(tmp_path:Path,monkeypatch):
    from phonoweave.synthesis_relevance_short_interval import extract
    from phonoweave.synthesis_relevance_acoustic_separability import Observation
    import phonoweave.synthesis_relevance_short_interval as module
    canonical=[Observation("wav1","D4","plain",tuple([0.1]*9))]
    monkeypatch.setattr(module,"frozen_legacy_extract",lambda root,base:(
        canonical,{"events":1,"physical_wavs":1}
    ))
    protocols,coverage=extract(tmp_path)
    assert protocols["legacy"]==canonical
    assert coverage["legacy_replay_matches_frozen_v05"] is True
    assert coverage["legacy_replay_events"]==1
