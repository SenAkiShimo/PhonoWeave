from __future__ import annotations
import numpy as np
from phonoweave.calibration_noise_gate_audit import gate_attribution, longest_run

def test_contiguous_persistence():
    assert longest_run(np.array([False,True,True,False,True])) == 2
    assert longest_run(np.zeros(12,dtype=bool)) == 0

def _synthetic_track(high, fraction):
    t=np.arange(len(high),dtype=float)*4.0 + 500.0
    return {"centers_ms":t,"high_dbfs":np.asarray(high,dtype=float),
            "high_fraction_db":np.asarray(fraction,dtype=float)}

def test_fraction_ablation_identifies_blocking_gate():
    high=np.full(250,-90.0)
    fraction=np.full(250,-25.0)
    high[120:165]=-55.0
    r=gate_attribution(_synthetic_track(high,fraction),1000.0)
    assert r["strict_sustained_gate_pass"] is False
    assert "high_fraction" in r["single_gate_ablation_restores_persistence"]
    assert r["ablation_diagnostic_only"]["high_fraction"]["sustained_without_gate"]

def test_all_gates_pass_on_sustained_high_band():
    high=np.full(250,-90.0)
    fraction=np.full(250,-25.0)
    high[120:165]=-55.0
    fraction[120:165]=-4.0
    r=gate_attribution(_synthetic_track(high,fraction),1000.0)
    assert r["strict_sustained_gate_pass"]
    assert r["single_gate_ablation_restores_persistence"] == []

def test_manual_anchor_does_not_change_gate_outcome():
    high=np.full(250,-90.0)
    fraction=np.full(250,-25.0)
    high[120:165]=-55.0
    fraction[120:165]=-4.0
    tracks=_synthetic_track(high,fraction)
    a=gate_attribution(tracks,1000.0)
    b=gate_attribution(tracks,1000.0,manual_onset_ms=990.0)
    assert a["strict_sustained_gate_pass"]==b["strict_sustained_gate_pass"]
    assert b["manual_context_diagnostic_only"]["frames"]>0
