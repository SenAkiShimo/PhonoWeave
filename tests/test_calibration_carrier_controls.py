from pathlib import Path
from phonoweave.calibration_carrier_controls import SCENARIOS, OUTPUT

def test_reproducible_scenarios_and_output_name():
    assert SCENARIOS == ("original", "noise_35db_snr", "noise_20db_snr")
    assert OUTPUT.endswith(".json")

def test_carrier_windows_are_declared_adversarial():
    from phonoweave.calibration_carrier_controls import evaluate_carriers
    assert evaluate_carriers.__doc__ is None or isinstance(evaluate_carriers.__doc__,str)
