from phonoweave.calibration_early_frication_multiseed import (
    summarize_onsets, compare_paired_rows
)

def test_pairwise_error_metrics_and_coverage():
    rows = [
        {"manual_onset_ms": 100.0, "baseline_error_ms": 50., "shadow_error_ms": 20., "backtrace_ms": 30.},
        {"manual_onset_ms": 200.0, "baseline_error_ms": -2., "shadow_error_ms": -8., "backtrace_ms": 6.},
        {"manual_onset_ms": None, "baseline_error_ms": None, "shadow_error_ms": None, "backtrace_ms": 0.},
    ]
    base = summarize_onsets(rows, "baseline")
    assert base["eligible"] == 2
    assert base["matched"] == 2
    assert base["errors_over_40ms"] == 1
    shadow = summarize_onsets(rows, "shadow")
    assert shadow["conditional_mae_ms"] == 14.
    comparison = compare_paired_rows(rows)
    assert comparison["improved"] == 1
    assert comparison["worsened"] == 1

def test_unmatched_results_are_not_scored_as_low_error():
    rows = [{"manual_onset_ms": 100., "baseline_error_ms": None, "shadow_error_ms": None}]
    result = summarize_onsets(rows, "baseline")
    assert result["matched"] == 0
    assert result["coverage"] == 0.
    assert result["conditional_mae_ms"] is None
