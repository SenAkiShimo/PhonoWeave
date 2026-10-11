"""Read-only evidence eligibility and schema checks."""
import pytest

from phonoweave.calibration_evidence_adapter import (
    _manual_anchor,
    _nonnegative_time,
    adapt_calibration_evidence,
)


def test_accepted_manual_anchor_is_not_adjudication():
    item = {"status": "ok", "source": "manual_v0.3",
            "outside_expected_slot": False, "absolute_anchor_ms": 123.2}
    row, reason = _manual_anchor(item, "consonant_onset", 3000.)
    assert reason is None
    assert row["absolute_ms"] == 123.2
    assert row["reviewed_independently"] is False


@pytest.mark.parametrize("item,expected", [
    (None, "consonant_onset_missing"),
    ({"status": "uncertain"}, "consonant_onset_not_ok"),
    ({"status": "ok", "source": "legacy_reviewed_copy"},
     "consonant_onset_not_independent_manual_source"),
    ({"status": "ok", "source": "manual_v0.3", "outside_expected_slot": True},
     "consonant_onset_slot_review"),
])
def test_manual_not_eligible(item, expected):
    row, reason = _manual_anchor(item, "consonant_onset", 3000.)
    assert row is None and reason == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True, "42", 3001.])
def test_invalid_times(value):
    with pytest.raises(ValueError):
        _nonnegative_time(value, "test", 3000.)


def test_rejects_non_session(tmp_path):
    with pytest.raises(ValueError, match="not a PhonoWeave calibration session"):
        adapt_calibration_evidence(tmp_path)
