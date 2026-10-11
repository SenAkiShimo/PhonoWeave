"""Result contract tests with controlled predictions and real WAV metadata."""
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from phonoweave.calibration_io import create_calibration_session, save_calibration_recording_to_session
from phonoweave.calibration_protocol import live_calibration_protocol
from phonoweave.calibration_results import export_calibration_results, serialize_candidate


def wav_bytes():
    audio=b"\x00\x00"*48000
    return (b"RIFF"+struct.pack("<I",36+len(audio))+b"WAVEfmt "+
            struct.pack("<IHHIIHH",16,1,1,16000,32000,2,16)+
            b"data"+struct.pack("<I",len(audio))+audio)


def test_contract_provides_separate_candidate_and_shadow():
    p={"base_unit":"x","context_family":"plain","syllable":"xi"}
    predicted=SimpleNamespace(status="paired_candidate",reason="frication_followed_by_vowel",
                              method="strict_v2.1",consonant_onset_ms=1000.,cv_boundary_ms=1200.)
    row=serialize_candidate(prompt_index=12,prompt=p,occurrence=1,nominal_cue_ms=1083.,
        prediction=predicted,shoulder={"status":"shadow_candidate","onset_ms":976.},
        selection={"onset_ms":976.,"applied":True,"reason":"supported_precursor"})
    assert row["detector"]["consonant_onset_ms"]==1000.
    assert row["experimental_refinement"]["selected_onset_ms"]==976.
    assert row["review_required"] is True


def test_unpaired_result_does_not_fake_selected_onset():
    p={"base_unit":"x","context_family":"plain","syllable":"xi"}
    result=serialize_candidate(prompt_index=12,prompt=p,occurrence=1,nominal_cue_ms=1083.,
        prediction=SimpleNamespace(status="unresolved",reason="no_frication",method="strict_v2.1",
                                  consonant_onset_ms=None,cv_boundary_ms=None),
        shoulder={"status":"unavailable","onset_ms":None},
        selection={"onset_ms":None,"applied":False,"reason":"not_paired"})
    assert result["experimental_refinement"]["selected_onset_ms"] is None
    assert result["detector"]["status"]=="unresolved"


def test_invalid_temporal_order_rejected():
    p={"base_unit":"x","context_family":"plain","syllable":"xi"}
    predicted=SimpleNamespace(status="paired_candidate",reason="x",method="strict_v2.1",
                              consonant_onset_ms=1300.,cv_boundary_ms=1200.)
    with pytest.raises(ValueError,match="ordered"):
        serialize_candidate(prompt_index=12,prompt=p,occurrence=1,nominal_cue_ms=1083.,
            prediction=predicted,shoulder={},
            selection={"onset_ms":1300.,"applied":False,"reason":"short_precursor"})


def test_partial_session_manifest_and_no_manual_labels(tmp_path,monkeypatch):
    session_id,session_dir=create_calibration_session(live_calibration_protocol(),tmp_path)
    protocol=live_calibration_protocol()
    save_calibration_recording_to_session(session_dir,12,protocol.prompts[12],wav_bytes(),16000)
    save_calibration_recording_to_session(session_dir,0,protocol.prompts[0],wav_bytes(),16000)

    import phonoweave.calibration_results as module
    monkeypatch.setattr(module,"acoustic_tracks",lambda x,sr:{"dummy":1})
    monkeypatch.setattr(module,"cv_tracks",lambda x,sr:{"dummy":2})
    monkeypatch.setattr(module,"detect_excess_fallback",
        lambda a,b,c:SimpleNamespace(status="paired_candidate",reason="synthetic",
            method="strict_v2.1",consonant_onset_ms=c,cv_boundary_ms=c+150.))
    monkeypatch.setattr(module,"trace_early_frication",
        lambda features,onset,cue:{"status":"shadow_candidate","onset_ms":onset-4.})
    monkeypatch.setattr(module,"select_early_onset",
        lambda **kwargs:{"onset_ms":kwargs["baseline_onset_ms"],"applied":False,"reason":"short_precursor"})

    payload=export_calibration_results(session_dir)
    assert payload["summary"]["recorded_prompts"]==2
    assert payload["summary"]["supported_recorded_prompts"]==1
    assert payload["summary"]["unsupported_recorded_prompts"]==1
    assert payload["summary"]["target_occurrences"]==2
    assert payload["model"]["default_voicebank_decisions_modified"] is False
    assert len(payload["predictions"])==2
    assert "manual_onset_ms" not in json.dumps(payload)
    assert (session_dir/"analysis"/"calibration_results_v1.json").exists()
