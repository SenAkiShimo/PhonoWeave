from __future__ import annotations

from pathlib import Path

from phonoweave.bank_concordance import ConcordanceRow
from phonoweave.bank_supplement import SPLIT_STATUS, BankSupplement
from phonoweave.plan_summary import build_summary
from phonoweave.synthesis_inventory import SynthesisInventory, SynthesisUnit


def _unit(base, decision, confidence, evidence):
    return SynthesisUnit(f"{base}_a", base, ("plain",), f"{base}_a", None, decision, confidence, evidence, "x")


def test_summary_groups_by_evidence_and_lists_what_is_held_back() -> None:
    inventory = SynthesisInventory(
        "s", "zh", Path("."),
        (
            _unit("m", "split_recommended", "moderate", "transition_supported"),
            _unit("sh", "split_recommended", "low", "single_layer_supported"),
            _unit("x", "split_recommended", "low", "single_layer_supported"),
            _unit("c", "unresolved", "low", "weak_or_inconsistent"),
        ),
    )
    rows = [
        ConcordanceRow("m", "m", "single", SPLIT_STATUS, ("my",), 23),
        ConcordanceRow("sh", "sh", "single", SPLIT_STATUS, ("shw",), 23),
        ConcordanceRow("x", "x", "single", SPLIT_STATUS, ("xw", "xy"), 46),
        ConcordanceRow("l", "l", "y:i", "different axis", ("lw",), 23),
        ConcordanceRow("b", "unresolved", "y:i", "voicebank splits, we have no evidence"),
    ]
    supplement = BankSupplement(lines=((),), connections=frozenset(), labels=("my", "shw"))

    text = build_summary(rows, inventory, supplement, {"tolerance": "strict"})

    assert "moderate: m (my), transition analysis" in text
    assert "weak (single layer): sh (shw), within-layer test" in text
    assert "x (xw, xy): left out by --supplement-onsets" in text
    assert "l (lw): different axis" in text
    assert "no evidence either way here: b" in text
    assert "no recording needed): c" in text
    assert "Settings: tolerance strict" in text


def test_summary_warns_when_only_one_layer_is_usable() -> None:
    inventory = SynthesisInventory("s", "zh", Path("."), ())
    supplement = BankSupplement(lines=(), connections=frozenset(), labels=())

    assert "Only one layer holds usable CV entries" in build_summary([], inventory, supplement, {}, 1)
    assert "Only one layer" not in build_summary([], inventory, supplement, {}, 2)
    on = build_summary([], inventory, supplement, {"single-layer evidence": "on"}, 1)
    assert "(h and f)" in on
