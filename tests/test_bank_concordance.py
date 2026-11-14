from __future__ import annotations

from pathlib import Path

from phonoweave.bank_concordance import bank_variants, compare_with_bank, format_concordance
from phonoweave.reclist import Syllable
from phonoweave.synthesis_inventory import SynthesisInventory, SynthesisUnit


def _unit(unit_id: str, base: str, contexts: tuple[str, ...], decision: str) -> SynthesisUnit:
    return SynthesisUnit(
        id=unit_id,
        base_unit=base,
        contexts=contexts,
        source_group=unit_id,
        canonical_context=None,
        decision=decision,
        confidence="high",
        acoustic_evidence="x",
        synthesis_evidence="x",
    )


def _audit(labels: list[str]) -> dict:
    return {
        "systematic_labels": [{"label": label, "finals": 20} for label in labels],
        "finals": [f"f{i}" for i in range(20)],
    }


def _syllables() -> tuple[Syllable, ...]:
    rows = []
    for base in ("sh", "l", "g", "m", "b"):
        for final in ("a", "i", "u"):
            rows.append(Syllable(f"{base}{final}", base, final))
    for final in ("i", "v"):
        rows.append(Syllable(f"x{final}", "x", final))
    return tuple(rows)


def test_bank_variants_parse_context_labels() -> None:
    found = bank_variants(_audit(["sh", "shw", "l", "ly", "b", "w", "y", "xy", "xw"]))
    assert found["sh"] == {"plain", "w"}
    assert found["l"] == {"plain", "y"}
    assert found["b"] == {"plain"}
    assert found["x"] == {"y", "w"}


def test_concordance_statuses() -> None:
    inventory = SynthesisInventory(
        speaker_id="s",
        language="zh",
        source_voicebank=Path("."),
        units=(
            _unit("sh_plain", "sh", ("plain",), "split_recommended"),
            _unit("sh_rounded", "sh", ("rounded",), "split_recommended"),
            _unit("l_plain", "l", ("plain",), "split_recommended"),
            _unit("l_rounded", "l", ("rounded",), "split_recommended"),
            _unit("g_plain", "g", ("plain",), "split_recommended"),
            _unit("g_rounded", "g", ("rounded",), "split_recommended"),
            _unit("m", "m", ("plain", "rounded"), "unresolved"),
            _unit("b", "b", ("plain", "rounded"), "unresolved"),
            _unit("x_plain", "x", ("plain",), "split_recommended"),
            _unit("x_rounded", "x", ("rounded",), "split_recommended"),
        ),
    )
    audit = _audit(["sh", "shw", "l", "ly", "g", "m", "my", "b", "xy", "xw"])

    rows = {row.base: row for row in compare_with_bank(audit, inventory, _syllables())}

    assert rows["sh"].status == "same axis"
    assert rows["l"].status == "different axis"
    assert rows["g"].status == "we split, voicebank does not"
    assert rows["m"].status == "voicebank splits, we have no evidence"
    assert rows["b"].status == "neither splits (ours unresolved)"
    assert rows["x"].status == "same axis"
    assert "sh_plain:i,o | sh_rounded:u" in format_concordance(list(rows.values()))


def test_suggested_additions_use_voicebank_labels() -> None:
    inventory = SynthesisInventory(
        speaker_id="s",
        language="zh",
        source_voicebank=Path("."),
        units=(
            _unit("l_plain", "l", ("plain",), "split_recommended"),
            _unit("l_rounded", "l", ("rounded",), "split_recommended"),
            _unit("g_plain", "g", ("plain",), "split_recommended"),
            _unit("g_rounded", "g", ("rounded",), "split_recommended"),
            _unit("sh_plain", "sh", ("plain",), "split_recommended"),
            _unit("sh_rounded", "sh", ("rounded",), "split_recommended"),
        ),
    )
    audit = _audit(["sh", "shw", "l", "ly", "g"])

    rows = {row.base: row for row in compare_with_bank(audit, inventory, _syllables())}

    assert rows["l"].suggested == ("lw",) and rows["l"].suggested_vc == len({s.tail for s in _syllables()})
    assert rows["g"].suggested == ("gw",) and rows["g"].suggested_vc == 0
    assert rows["sh"].suggested == ()
    text = format_concordance(list(rows.values()))
    assert f"l: add lw -> {rows['l'].suggested_vc} VC entries" in text
    assert "g: add gw -> no VC entries" in text
