from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from phonoweave.bank_concordance import compare_with_bank, format_concordance
from phonoweave.reclist import Syllable, UnitMap, build_reclist
from phonoweave.synthesis_inventory import SynthesisInventory, SynthesisUnit
from phonoweave.tolerance import MERGED, apply_tolerance


def _unit(unit_id, base, contexts, decision, confidence="high", acoustic="strongly_supported"):
    return SynthesisUnit(unit_id, base, contexts, unit_id, None, decision, confidence, acoustic, "x")


def _inventory() -> SynthesisInventory:
    return SynthesisInventory(
        "s", "zh", Path("."),
        (
            _unit("sh_plain", "sh", ("plain",), "split_recommended"),
            _unit("sh_rounded", "sh", ("rounded",), "split_recommended"),
            _unit("g_plain", "g", ("plain",), "split_recommended", "moderate", "supported"),
            _unit("g_rounded", "g", ("rounded",), "split_recommended", "moderate", "supported"),
            _unit("k_plain", "k", ("plain",), "split_recommended", "high", "supported"),
            _unit("k_rounded", "k", ("rounded",), "split_recommended", "high", "supported"),
            _unit("m", "m", ("plain", "rounded"), "unresolved", "low", "weak_or_inconsistent"),
        ),
    )


def test_levels_merge_progressively() -> None:
    strict, merged = apply_tolerance(_inventory(), "strict")
    assert merged == () and strict == _inventory()

    _, balanced = apply_tolerance(_inventory(), "balanced")
    assert balanced == ("g",)

    loose_inventory, loose = apply_tolerance(_inventory(), "loose")
    assert loose == ("g", "k")
    assert {u.decision for u in loose_inventory.units if u.base_unit == "k"} == {MERGED}


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError):
        apply_tolerance(_inventory(), "fuzzy")


def test_tolerance_merge_collapses_units() -> None:
    syllables = tuple(
        Syllable(f"g{final}", "g", final) for final in ("a", "u")
    )
    inventory, _ = apply_tolerance(_inventory(), "balanced")
    units = UnitMap(inventory, collapse_unresolved=True)
    assert units.unit(syllables[0], "internal") == units.unit(syllables[1], "internal")
    assert "g" in units.collapsed
    assert build_reclist(syllables, units, 6, 3).lines


def _audit() -> dict:
    return {
        "systematic_labels": [{"label": label, "finals": 20} for label in ("m", "my")],
        "finals": [f"f{i}" for i in range(20)],
    }


def test_bank_merge_candidates_follow_tolerance() -> None:
    syllables = tuple(Syllable(f"m{f}", "m", f) for f in ("a", "i", "u"))
    inventory = _inventory()

    strict = compare_with_bank(_audit(), inventory, syllables, "strict")
    assert not strict[0].merge_labels

    balanced = compare_with_bank(_audit(), inventory, syllables, "balanced")
    assert balanced[0].merge_labels == ("my",)
    assert balanced[0].merge_saving == 20
    assert "merge my -> saves 20 VC entries" in format_concordance(balanced)

    weak_none = replace(
        inventory,
        units=tuple(
            replace(u, acoustic_evidence="supported") if u.base_unit == "m" else u
            for u in inventory.units
        ),
    )
    assert not compare_with_bank(_audit(), weak_none, syllables, "balanced")[0].merge_labels
    assert compare_with_bank(_audit(), weak_none, syllables, "loose")[0].merge_labels == ("my",)
