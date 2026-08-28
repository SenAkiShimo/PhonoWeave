from collections import Counter

from phonoweave.diagnostic_selector import _candidates, _take_with_replicates
from phonoweave.supplement_plan import SupplementRequest, SupplementTarget


def _request(base_unit: str, family: str, count: int, role: str | None = None) -> SupplementRequest:
    return SupplementRequest(
        base_unit=base_unit,
        class_name="rhotic" if base_unit == "r" else "fricative",
        gap_type="coverage_limited",
        priority="high",
        role_scope=role,
        targets=(SupplementTarget(context_family=family, diagnostic_items=count),),
        pitch_policy="repeat_each_item_across_existing_oto_sets",
        automatic_round_limit=1,
        stop_rule="reanalyze_once_then_freeze_if_still_unresolved",
    )


def test_m_does_not_offer_v_series_candidates() -> None:
    request = _request("m", "v_series", 2, "internal")
    assert _candidates(request, Counter(), "v_series") == []


def test_r_front_uses_ri_and_allows_independent_replicates() -> None:
    request = _request("r", "front", 2)
    candidates = _candidates(request, Counter(), "front")
    assert [item.syllable for item in candidates] == ["ri"]
    selected = _take_with_replicates(candidates, 2)
    assert [item.syllable for item in selected] == ["ri", "ri"]
    assert [item.replicate for item in selected] == [1, 2]


def test_selector_prefers_less_observed_legal_syllables() -> None:
    request = _request("f", "rounded", 1, "internal")
    counts = Counter({("f", "u"): 3, ("f", "o"): 1, ("f", "ou"): 2})
    candidates = _candidates(request, counts, "rounded")
    assert candidates[0].syllable == "fo"
    assert candidates[0].existing_observations == 1


def test_jq_rounded_counts_use_orthographic_finals() -> None:
    j_request = _request("j", "rounded", 1)
    q_request = _request("q", "rounded", 1)
    counts = Counter({("j", "u"): 4, ("q", "u"): 5})
    j_candidates = _candidates(j_request, counts, "rounded")
    q_candidates = _candidates(q_request, counts, "rounded")
    ju = next(item for item in j_candidates if item.syllable == "ju")
    qu = next(item for item in q_candidates if item.syllable == "qu")
    assert ju.existing_observations == 4
    assert qu.existing_observations == 5


def test_illegal_tiu_is_not_a_candidate() -> None:
    request = _request("t", "i_series", 10, "internal")
    candidates = _candidates(request, Counter(), "i_series")
    assert "tiu" not in {item.syllable for item in candidates}
