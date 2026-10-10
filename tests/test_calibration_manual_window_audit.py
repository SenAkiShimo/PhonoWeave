from __future__ import annotations

import numpy as np
from phonoweave.calibration_manual_window_audit import gate_audit, first_sustained, spectral_tone_audit


def _tracks():
    n=160
    f={"centers":np.arange(n,dtype=float)*4-133,
       "high_db":np.full(n,-80.),
       "rms_db":np.full(n,-60.),
       "flatness":np.full(n,.02),
       "zcr":np.full(n,.02),
       "periodicity":np.full(n,.95)}
    for key,value in (("high_db",-42.),("flatness",.4),("zcr",.25),("periodicity",.1)):
        f[key][60:95]=value
    return f


def test_first_sustained():
    assert first_sustained(np.array([0,1,1,1,0],dtype=bool))==1
    assert first_sustained(np.array([0,1,1,0],dtype=bool)) is None


def test_manual_gold_exposes_rms_rejection():
    f=_tracks()
    result=gate_audit(f,cue_ms=0.,gold_ms=120.)
    assert result["full_joint_frames"]==0
    assert result["no_rms_candidate_ms_diagnostic_only"]==107.
    assert result["gold"]["inside_automatic_search"]
    assert result["gold"]["no_rms_candidate_error_ms_diagnostic_only"]==-13.
    assert result["gold"]["local_gate_counts"]["rms_rise"]==0


def test_manual_outside_search_not_accepted_as_in_window():
    result=gate_audit(_tracks(),cue_ms=0.,gold_ms=1000.)
    assert not result["gold"]["inside_automatic_search"]
    assert result["gold"]["local_frame_count"]==0


def test_tone_audit_handles_zero_audio():
    result=spectral_tone_audit(np.zeros(16000),16000,250.)
    assert result["available"] and result["rms_dbfs"] < -100.
