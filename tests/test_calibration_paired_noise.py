"""Regression checks for the paired-detector evaluation entrypoint."""
from phonoweave import calibration_paired_noise
from phonoweave.calibration_robustness import SCENARIOS


def test_stress_conditions_are_explicit():
    assert SCENARIOS[0] == "original"
    assert "noise_20db_snr" in SCENARIOS
    assert callable(calibration_paired_noise.evaluate_paired_noise)


def test_evaluation_entrypoint_is_not_a_pytest_test():
    assert not hasattr(calibration_paired_noise, "test_paired_noise")
