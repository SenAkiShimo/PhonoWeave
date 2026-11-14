from __future__ import annotations

from pathlib import Path

from phonoweave.reclist import Syllable, UnitMap
from phonoweave.synthesis_inventory import SynthesisInventory, SynthesisUnit
from phonoweave.transition_consistency import TransitionPairResult, TransitionReport
from phonoweave.transition_evidence import apply_transition_evidence


def _unit(unit_id, base, contexts, decision="unresolved"):
    return SynthesisUnit(unit_id, base, contexts, unit_id, None, decision, "low", "x", "x")


def _pairs(a, b, p_holm=0.0007):
    return tuple(
        TransitionPairResult(a, b, level, 4, 4, 20, 30, 1.0, 0.0002, p_holm)
        for level in (0.25, 0.5, 0.75)
    )


def _report(base, pairs, verdict="consistent_transition_difference"):
    return TransitionReport(base, 60, 0, {}, pairs, verdict)


def _inventory():
    return SynthesisInventory(
        "s", "zh", Path("."),
        (
            _unit("m", "m", ("plain", "rounded")),
            _unit("n_plain", "n", ("plain",), "split_recommended"),
            _unit("n_rounded", "n", ("rounded",), "split_recommended"),
            _unit("l", "l", ("plain", "rounded")),
        ),
    )


def _unit_of(inventory, base, final):
    return UnitMap(inventory).unit(Syllable(base + final, base, final), "internal")


def test_supported_axis_replaces_units() -> None:
    reports = {
        "m": _report("m", _pairs("i_series", "other")),
        "n": _report("n", _pairs("i_series", "other")),
        "l": _report("l", (), "insufficient_data"),
    }
    inventory, notes = apply_transition_evidence(_inventory(), reports)

    assert _unit_of(inventory, "m", "i") == "m_front"
    assert _unit_of(inventory, "m", "u") == "m_plain"
    assert _unit_of(inventory, "n", "i") == "n_front"
    assert _unit_of(inventory, "n", "u") == "n_plain"
    assert _unit_of(inventory, "l", "i") == "l"
    assert any(note.startswith("l: insufficient_data") for note in notes)


def test_existing_axis_is_left_alone() -> None:
    reports = {"n": _report("n", _pairs("u_series", "other"))}
    inventory, notes = apply_transition_evidence(_inventory(), reports)

    assert [u.id for u in inventory.units if u.base_unit == "n"] == ["n_plain", "n_rounded"]
    assert "already present" in notes[0]


def test_weak_report_changes_nothing() -> None:
    reports = {"m": _report("m", _pairs("i_series", "other", 0.4), "no_difference_detected")}
    inventory, _ = apply_transition_evidence(_inventory(), reports)

    assert inventory == _inventory()
