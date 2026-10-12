"""A/B trial staging safety and waveform identity tests."""
import json
import wave
from pathlib import Path
import numpy as np
import pytest
from phonoweave.synthesis_relevance_ab_staging import (
    _assemble,_slice,choose_trial,stage_trial,
)
def test_assemble_has_same_right_audio_and_expected_length():
    rate=1000
    left=np.full(120,0.2)
    right=np.full(600,0.4)
    joined,pre=_assemble(left,right,rate)
    assert len(joined)==712
    assert joined[-500:].tolist()==pytest.approx(right[-500:].tolist())
    assert pre==pytest.approx(116.)
def test_slice_refuses_out_of_bounds():
    with pytest.raises(Exception):
        _slice(np.ones(100),1000,10.,25.,50.)
def test_choose_trial_checks_requested_context_and_selection():
    rows=[{"event_id":"a","target_context":"plain","subbank":"D4",
           "same_donor_event_ids":["b"],"cross_donor_event_ids":["c"]}]
    assert choose_trial(rows)["event_id"]=="a"
    with pytest.raises(ValueError):
        choose_trial(rows,context="rounded")
def test_refuses_invalid_or_existing_destination(tmp_path):
    source=tmp_path/"singer"
    source.mkdir()
    pilot=tmp_path/"pilot.json"
    pilot.write_text(json.dumps({"schema":"invalid"}))
    with pytest.raises(ValueError):
        stage_trial(source,pilot,tmp_path/"new")
    assert not (tmp_path/"new").exists()
    with pytest.raises(ValueError):
        stage_trial(source,pilot,source/"child")
