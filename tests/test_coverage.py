from phonoweave.coverage import analyzer_for, coverage_status


def test_lateral_is_experimental_coverage() -> None:
    assert coverage_status("l") == "experimental"
    assert analyzer_for("l") == "lateral"


def test_unimplemented_onsets_remain_unsupported() -> None:
    assert coverage_status("m") == "unsupported"
    assert analyzer_for("m") is None
