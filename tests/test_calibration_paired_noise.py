from phonoweave.calibration_paired_noise import test_paired_noise
from phonoweave.calibration_robustness import SCENARIOS
def test_stress_conditions_are_explicit():
    assert SCENARIOS[0]=="original"
    assert "noise_20db_snr" in SCENARIOS
    assert callable(test_paired_noise)
