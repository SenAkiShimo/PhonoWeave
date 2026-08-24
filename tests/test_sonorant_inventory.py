from types import SimpleNamespace

from phonoweave.inventory import _lateral_split_pairs, _nasal_acoustic_level, _nasal_split_pairs


def test_nasal_split_requires_bidirectional_support() -> None:
    relevance = SimpleNamespace(
        pairs=(
            SimpleNamespace(
                left="i_series",
                right="u_series",
                both_transition_positive=False,
                both_transition_holm_significant=False,
                both_body_positive=False,
                all_oto_sets_transition_positive=False,
            ),
        )
    )
    assert _nasal_split_pairs(relevance) == []


def test_nasal_split_accepts_only_complete_bidirectional_gate() -> None:
    relevance = SimpleNamespace(
        pairs=(
            SimpleNamespace(
                left="i_series",
                right="u_series",
                both_transition_positive=True,
                both_transition_holm_significant=True,
                both_body_positive=True,
                all_oto_sets_transition_positive=True,
            ),
        )
    )
    assert _nasal_split_pairs(relevance) == ["i_series<->u_series"]


def test_nasal_partial_acoustic_evidence_stays_partial() -> None:
    pair = SimpleNamespace(
        left="i_series",
        right="other",
        stratified_permutation_p=0.01,
        stratified_p_holm=0.02,
    )
    acoustic = SimpleNamespace(
        roles=(
            SimpleNamespace(
                role="internal",
                windows=(
                    SimpleNamespace(pairwise=(pair,)),
                    SimpleNamespace(pairwise=(pair,)),
                ),
            ),
        )
    )
    level, eligible, significant = _nasal_acoustic_level(acoustic)
    assert level == "partial"
    assert eligible == {("i_series", "other")}
    assert significant == {("i_series", "other")}


def test_lateral_split_rejects_directional_asymmetry() -> None:
    forward = SimpleNamespace(
        target_family="i_series",
        substitution_family="u_series",
        targets=10,
        mean_boundary_spectral_delta=0.2,
        boundary_spectral_p_holm=0.01,
        mean_body_spectral_delta=0.1,
        oto_sets=(SimpleNamespace(mean_boundary_spectral_delta=0.2),),
    )
    reverse = SimpleNamespace(
        target_family="u_series",
        substitution_family="i_series",
        targets=10,
        mean_boundary_spectral_delta=-0.1,
        boundary_spectral_p_holm=1.0,
        mean_body_spectral_delta=-0.1,
        oto_sets=(SimpleNamespace(mean_boundary_spectral_delta=-0.1),),
    )
    relevance = SimpleNamespace(
        roles=(
            SimpleNamespace(
                role="internal",
                comparisons=(forward, reverse),
            ),
        )
    )
    assert _lateral_split_pairs(relevance) == []
