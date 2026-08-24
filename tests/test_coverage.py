from phonoweave.coverage import analyzer_for, coverage_status


def test_sonorant_analyzers_are_experimental_coverage() -> None:
    for base, analyzer in (("l", "lateral"), ("m", "nasal"), ("n", "nasal")):
        assert coverage_status(base) == "experimental"
        assert analyzer_for(base) == analyzer


def test_unimplemented_onsets_remain_unsupported() -> None:
    assert coverage_status("b") == "unsupported"
    assert analyzer_for("b") is None
