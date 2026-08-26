from pathlib import Path
from types import SimpleNamespace

import phonoweave.fh_fricative_inventory as inventory


def _acoustic(base_unit: str, p_value: float, cross: float):
    return SimpleNamespace(
        base_unit=base_unit,
        samples=30,
        duplicate_observations_removed=0,
        ambiguous_segments_removed=0,
        cross_oto_set_balanced_accuracy=cross,
        stratified_distance=0.8,
        stratified_permutation_p=p_value,
    )


def test_f_stops_before_relevance_when_acoustic_gate_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        inventory,
        "analyze_fh_fricative",
        lambda root, base: _acoustic(base, 0.1152, 0.708),
    )
    result = inventory.fh_fricative_decisions(Path("/tmp/voicebank"))[0]
    assert result.base_unit == "f"
    assert result.acoustic_evidence == "weak_or_inconsistent"
    assert result.synthesis_evidence == "not_tested"
    assert result.decision == "unresolved"


def test_h_requires_relevance_pair_for_split(monkeypatch) -> None:
    monkeypatch.setattr(
        inventory,
        "analyze_fh_fricative",
        lambda root, base: _acoustic(base, 0.0002, 0.798),
    )
    pair = SimpleNamespace(
        both_boundary_positive=True,
        both_boundary_holm_significant=True,
        all_oto_sets_boundary_positive=True,
        split_supported_under_proxy=True,
    )
    monkeypatch.setattr(
        inventory,
        "h_fricative_relevance_test",
        lambda root: SimpleNamespace(samples=60, pair=pair),
    )
    result = inventory.fh_fricative_decisions(Path("/tmp/voicebank"))[1]
    assert result.base_unit == "h"
    assert result.acoustic_evidence == "strongly_supported"
    assert result.synthesis_evidence == "supported_under_proxy"
    assert result.decision == "split_recommended"
