from __future__ import annotations

from types import SimpleNamespace

from phonoweave.inventory import _fricative_acoustic_level, within_layer_supported


def _region(p, accuracy=0.72, distance=0.9):
    return SimpleNamespace(
        permutation_p=p, loo_balanced_accuracy=accuracy, standardized_distance=distance
    )


def _result(*subbanks):
    return SimpleNamespace(
        subbanks=list(subbanks),
        cross_core_balanced_accuracy=None,
        mean_core_distance=None,
    )


def test_one_region_is_enough_but_must_pass_the_split_threshold() -> None:
    assert within_layer_supported((_region(0.3), _region(0.002)))
    assert not within_layer_supported((_region(0.03), _region(0.4)))


def test_accuracy_and_distance_are_also_required() -> None:
    assert not within_layer_supported((_region(0.001, accuracy=0.6),))
    assert not within_layer_supported((_region(0.001, distance=0.4),))


def test_single_layer_rule_applies_only_to_one_layer_and_only_when_asked() -> None:
    layer = SimpleNamespace(core=_region(0.01), late=_region(0.4))
    one = _result(layer)
    two = _result(layer, layer)

    assert _fricative_acoustic_level(one) == "weak"
    assert _fricative_acoustic_level(one, single_layer=True) == "within_layer"
    assert _fricative_acoustic_level(two, single_layer=True) == "weak"


def test_affricate_layer_results_are_accepted() -> None:
    from phonoweave.affricate import AffricateSubbankContrast

    layer = AffricateSubbankContrast(
        subbank="a", plain_count=8, rounded_count=6, distance=0.9,
        loo_balanced_accuracy=0.8, permutation_p=0.004,
        mean_plain={}, mean_rounded={}, effects={},
    )
    weak = AffricateSubbankContrast(
        subbank="a", plain_count=8, rounded_count=6, distance=0.3,
        loo_balanced_accuracy=0.8, permutation_p=0.004,
        mean_plain={}, mean_rounded={}, effects={},
    )

    assert within_layer_supported((layer,))
    assert not within_layer_supported((weak,))
