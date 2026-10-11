from phonoweave.calibration_selective_early_onset import (
    select_early_onset, score_rows, pairwise_score
)

def choose(base, shadow, cv=1290., status="paired_candidate"):
    return select_early_onset(
        paired_status=status, baseline_onset_ms=base,
        shadow_onset_ms=shadow, cv_boundary_ms=cv
    )

def test_short_shift_preserves_original():
    r=choose(1100.,1096.)
    assert r=={"onset_ms":1100., "applied":False, "reason":"short_precursor"}

def test_full_frame_precursor_is_selected():
    r=choose(1100.,1080.)
    assert r["onset_ms"]==1080. and r["applied"]

def test_not_paired_cannot_create_onset():
    assert choose(1100.,1060.,status="review")["onset_ms"] is None

def test_cv_timing_and_backtrace_limits():
    assert choose(1100.,1076.,cv=1450.)["reason"]=="cv_timing"
    assert choose(1100.,980.)["reason"]=="invalid_backtrace"
    assert choose(1100.,1110.)["reason"]=="invalid_backtrace"

def test_scoring_does_not_count_unmatched_as_accuracy():
    rows=[{"manual_onset_ms":1000.,"selected_error_ms":None},
          {"manual_onset_ms":1100.,"selected_error_ms":10.}]
    r=score_rows(rows,"selected_error_ms")
    assert r["eligible"]==2 and r["matched"]==1 and r["coverage"]==0.5
    assert r["conditional_mae_ms"]==10.

def test_pairwise_harm_recorded():
    rows=[{"baseline_error_ms":3.,"selected_error_ms":-7.},
          {"baseline_error_ms":25.,"selected_error_ms":5.}]
    d=pairwise_score(rows)
    assert d["improved"]==1 and d["worsened"]==1
