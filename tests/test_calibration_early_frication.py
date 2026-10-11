from phonoweave.calibration_manual_labels import DEV_ANCHOR_TARGETS
from phonoweave.calibration_manual_labeler import HTML
from phonoweave.calibration_early_frication import trace_early_frication
import numpy as np

def test_additive_label_targets_and_focused_navigation():
    assert ("x","rounded") in DEV_ANCHOR_TARGETS
    assert ("sh","plain") in DEV_ANCHOR_TARGETS
    assert "new URLSearchParams(location.search).get('targets')" in HTML

def test_no_onset_does_not_create_candidate():
    assert trace_early_frication({},None,1000.)["status"]=="unavailable"

def test_backtrace_never_advances_valid_core():
    t=np.arange(0,3000,4,dtype=float)
    high=np.full(len(t),-85.)
    high[(t>=980)&(t<=1260)]=-45.
    ratio=np.full(len(t),-5.)
    r=trace_early_frication({"centers_ms":t,"high_dbfs":high,"high_fraction_db":ratio},1088.,1083.333)
    assert r["status"]=="shadow_candidate"
    assert r["onset_ms"]<=1088.
    assert 0<=r["backtrace_ms"]<=100.
