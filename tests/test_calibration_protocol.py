from phonoweave.calibration_protocol import live_calibration_protocol


def test_live_calibration_protocol_is_frozen_and_single_round() -> None:
    protocol = live_calibration_protocol()
    assert protocol.name == "live_calibration"
    assert protocol.version == "0.1"
    assert protocol.automatic_supplement_rounds == 1
    assert protocol.stop_rule == "reanalyze_once_then_freeze_if_still_unresolved"


def test_live_calibration_protocol_covers_supported_and_unresolved_targets() -> None:
    protocol = live_calibration_protocol()
    keys = {(item.base_unit, item.context_family) for item in protocol.prompts}
    assert ("sh", "plain") in keys
    assert ("sh", "rounded") in keys
    assert ("h", "rounded") in keys
    assert ("h", "other") in keys
    assert ("zh", "plain") in keys
    assert ("zh", "rounded") in keys
    assert ("r", "front") in keys


def test_internal_prompts_place_target_between_fixed_carriers() -> None:
    protocol = live_calibration_protocol()
    prompt = next(item for item in protocol.prompts if item.base_unit == "f" and item.context_family == "rounded")
    assert prompt.spoken_pattern == "a fo a"
    assert prompt.repeats == 3


def test_all_role_prompts_repeat_target_in_one_pattern() -> None:
    protocol = live_calibration_protocol()
    prompt = next(item for item in protocol.prompts if item.base_unit == "zh" and item.context_family == "plain")
    assert prompt.spoken_pattern == "zha a zha a"
    assert prompt.repeats == 3
