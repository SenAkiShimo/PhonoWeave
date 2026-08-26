from phonoweave.h_fricative_relevance import (
    HFricativeOtoSetResult,
    HFricativeRelevanceComparison,
    _holm_adjust,
    _pair_summary,
)


def _comparison(target: str, substitute: str, delta: float, p: float, oto: float):
    return HFricativeRelevanceComparison(
        target_family=target,
        substitution_family=substitute,
        targets=6,
        mean_boundary_spectral_delta=delta,
        boundary_spectral_p=p,
        boundary_spectral_p_holm=p,
        mean_body_spectral_delta=0.2,
        body_spectral_p=0.01,
        oto_sets=(
            HFricativeOtoSetResult("A3", 2, oto, 0.1),
            HFricativeOtoSetResult("C3", 2, oto, 0.1),
            HFricativeOtoSetResult("Gb3", 2, oto, 0.1),
        ),
        target_penalties=(),
    )


def test_holm_adjusts_two_primary_directions():
    assert _holm_adjust([0.01, 0.04]) == [0.02, 0.04]


def test_pair_summary_requires_bidirectional_consistent_support():
    supported = _pair_summary(
        (
            _comparison("rounded", "other", 0.3, 0.01, 0.2),
            _comparison("other", "rounded", 0.2, 0.02, 0.1),
        )
    )
    assert supported.split_supported_under_proxy

    inconsistent = _pair_summary(
        (
            _comparison("rounded", "other", 0.3, 0.01, 0.2),
            _comparison("other", "rounded", 0.2, 0.02, -0.1),
        )
    )
    assert not inconsistent.split_supported_under_proxy
