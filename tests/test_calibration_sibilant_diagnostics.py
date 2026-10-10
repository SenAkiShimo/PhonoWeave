from __future__ import annotations

import numpy as np
from phonoweave.calibration_sibilant_diagnostics import summarize_gates


def _features():
    n=100
    return {
        "centers":np.arange(n,dtype=float)*4.0-133.0,
        "high_db":np.full(n,-85.0),
        "rms_db":np.full(n,-65.0),
        "flatness":np.full(n,0.02),
        "zcr":np.full(n,0.01),
        "periodicity":np.full(n,0.95),
    }


def test_sustained_frication_passes_all_gates():
    f=_features()
    f["high_db"][40:70]=-45.0
    f["rms_db"][40:70]=-30.0
    f["flatness"][40:70]=0.2
    f["zcr"][40:70]=0.22
    f["periodicity"][40:70]=0.15
    r=summarize_gates(f,0.0)
    assert r["all_five_pass_frames"]==30
    assert r["longest_all_five_run_frames"]==30


def test_independent_pass_is_not_joint_pass():
    f=_features()
    f["high_db"][10:40]=-45
    f["flatness"][10:40]=0.4
    f["zcr"][10:40]=0.3
    f["periodicity"][10:40]=0.15
    f["rms_db"][60:90]=-20
    r=summarize_gates(f,0.0)
    assert r["all_five_pass_frames"]==0
    assert r["best_gate_count"]==4
