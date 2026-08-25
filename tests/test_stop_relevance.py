from types import SimpleNamespace

from phonoweave.stop_relevance import _pair_summaries


def test_stop_pair_summary_rejects_one_way_support() -> None:
    forward = SimpleNamespace(
        target_family="u_series",
        substitution_family="other",
        mean_onset_spectral_delta=0.2,
        onset_spectral_p_holm=0.01,
        oto_sets=(SimpleNamespace(mean_onset_spectral_delta=0.2),),
    )
    reverse = SimpleNamespace(
        target_family="other",
        substitution_family="u_series",
        mean_onset_spectral_delta=-0.1,
        onset_spectral_p_holm=1.0,
        oto_sets=(SimpleNamespace(mean_onset_spectral_delta=-0.1),),
    )
    result = _pair_summaries(
        (("u_series", "other"),),
        (forward, reverse),
    )
    assert len(result) == 1
    assert result[0].both_onset_positive is False
    assert result[0].both_onset_holm_significant is False
    assert result[0].all_oto_sets_onset_positive is False


def test_stop_pair_summary_accepts_complete_bidirectional_support() -> None:
    forward = SimpleNamespace(
        target_family="u_series",
        substitution_family="other",
        mean_onset_spectral_delta=0.2,
        onset_spectral_p_holm=0.01,
        oto_sets=(
            SimpleNamespace(mean_onset_spectral_delta=0.1),
            SimpleNamespace(mean_onset_spectral_delta=0.2),
            SimpleNamespace(mean_onset_spectral_delta=0.3),
        ),
    )
    reverse = SimpleNamespace(
        target_family="other",
        substitution_family="u_series",
        mean_onset_spectral_delta=0.15,
        onset_spectral_p_holm=0.02,
        oto_sets=(
            SimpleNamespace(mean_onset_spectral_delta=0.1),
            SimpleNamespace(mean_onset_spectral_delta=0.2),
            SimpleNamespace(mean_onset_spectral_delta=0.15),
        ),
    )
    result = _pair_summaries(
        (("u_series", "other"),),
        (forward, reverse),
    )
    assert len(result) == 1
    assert result[0].both_onset_positive is True
    assert result[0].both_onset_holm_significant is True
    assert result[0].all_oto_sets_onset_positive is True
