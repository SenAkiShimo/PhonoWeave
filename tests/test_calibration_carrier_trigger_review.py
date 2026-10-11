from phonoweave.calibration_carrier_trigger_review import TRIGGERS, NAME

def test_review_targets():
    assert len(TRIGGERS) == 2
    assert {item[0] for item in TRIGGERS} == {6, 26}
    assert NAME.endswith(".json")
