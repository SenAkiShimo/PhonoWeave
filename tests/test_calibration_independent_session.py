from phonoweave.calibration_independent_session import ALLOWED_DEVICES, ALLOWED_ENVIRONMENTS
from phonoweave.calibration_selective_early_onset import select_early_onset

def test_explicit_capture_provenance_categories():
    assert "phone" in ALLOWED_DEVICES
    assert "external_usb" in ALLOWED_DEVICES
    assert "quiet" in ALLOWED_ENVIRONMENTS
    assert "indoor_noise" in ALLOWED_ENVIRONMENTS

def test_selection_does_not_require_reference_labels():
    r=select_early_onset(
        paired_status="paired_candidate",
        baseline_onset_ms=1100.,
        shadow_onset_ms=1076.,
        cv_boundary_ms=1300.,
    )
    assert r["applied"] is True
    assert r["onset_ms"]==1076.
