from __future__ import annotations

from pathlib import Path

from phonoweave.final_classes import derive_final_classes, summarize_final_classes
from phonoweave.reclist import Syllable, UnitMap, build_reclist, observed_syllables


def _bank(root: Path) -> Path:
    bank = root / "bank"
    bank.mkdir()
    rows = [
        "a.wav=ba,0,0,0,0,0",
        "a.wav=- ba,0,0,0,0,0",
        "a.wav=bua,0,0,0,0,0",
        "an.wav=duan,0,0,0,0,0",
        "an.wav=dan,0,0,0,0,0",
        "x.wav=zhi,0,0,0,0,0",
        "x.wav=zi,0,0,0,0,0",
        "x.wav=bi,0,0,0,0,0",
        "x.wav=ir b,0,0,0,0,0",
        "x.wav=i0 b,0,0,0,0,0",
        "x.wav=i b,0,0,0,0,0",
        "a.wav=a b,0,0,0,0,0",
        "a.wav=a d,0,0,0,0,0",
        "an.wav=an b,0,0,0,0,0",
        "an.wav=an d,0,0,0,0,0",
        "a.wav=a R,0,0,0,0,0",
        "an.wav=an R,0,0,0,0,0",
    ]
    (bank / "oto.ini").write_text("".join(row + "\n" for row in rows), encoding="utf-8")
    return bank


def test_final_classes_follow_wav_grouping(tmp_path) -> None:
    bank = _bank(tmp_path)
    mapping = derive_final_classes(bank)

    assert mapping[("b", "ua")] == "a"
    assert mapping[("d", "uan")] == "an"
    assert mapping[("zh", "i")] == "ir"
    assert mapping[("z", "i")] == "i0"
    assert ("b", "i") not in mapping
    assert summarize_final_classes(mapping)["ua"] == ["a"]

    syllables = {item.text: item for item in observed_syllables(bank, mapping)}
    assert syllables["bua"].tail == "a"
    assert syllables["duan"].tail == "an"
    assert syllables["bua"].final == "ua"


def test_vc_requirements_use_final_classes() -> None:
    plain = (
        Syllable("ba", "b", "a"),
        Syllable("bua", "b", "ua"),
        Syllable("dan", "d", "an"),
        Syllable("duan", "d", "uan"),
    )
    merged = (
        Syllable("ba", "b", "a", "a"),
        Syllable("bua", "b", "ua", "a"),
        Syllable("dan", "d", "an", "an"),
        Syllable("duan", "d", "uan", "an"),
    )

    full = build_reclist(plain, UnitMap(None), 6, 5)
    reduced = build_reclist(merged, UnitMap(None), 6, 5)

    assert len(full.requirements.vc) == 8
    assert len(reduced.requirements.vc) == 4
    assert len(reduced.requirements.end) == 2


def test_stop_connections_ignore_context_splits() -> None:
    from pathlib import Path

    from phonoweave.reclist import STOP_ONSETS
    from phonoweave.synthesis_inventory import SynthesisInventory, SynthesisUnit

    def unit(unit_id, base, contexts):
        return SynthesisUnit(unit_id, base, contexts, unit_id, None, "split_recommended",
                             "high", "x", "x")

    inventory = SynthesisInventory(
        "s", "zh", Path("."),
        (
            unit("g_plain", "g", ("plain",)),
            unit("g_rounded", "g", ("rounded",)),
            unit("s_plain", "s", ("plain",)),
            unit("s_rounded", "s", ("rounded",)),
        ),
    )
    syllables = tuple(
        Syllable(f"{onset}{final}", onset, final)
        for onset in ("g", "s")
        for final in ("a", "u")
    )

    split = build_reclist(syllables, UnitMap(inventory), 8, 5)
    shared = build_reclist(
        syllables, UnitMap(inventory, shared_connection=STOP_ONSETS), 8, 5
    )

    assert len(split.requirements.vc) == 8
    assert len(shared.requirements.vc) == 6


def test_er_is_added_when_the_voicebank_has_it(tmp_path) -> None:
    bank = tmp_path / "er_bank"
    bank.mkdir()
    (bank / "oto.ini").write_text(
        "a.wav=ba,0,0,0,0,0\ner.wav=er b,0,0,0,0,0\n", encoding="utf-8"
    )
    texts = {item.text for item in observed_syllables(bank)}
    assert "er" in texts

    plain = tmp_path / "plain_bank"
    plain.mkdir()
    (plain / "oto.ini").write_text("a.wav=ba,0,0,0,0,0\n", encoding="utf-8")
    assert "er" not in {item.text for item in observed_syllables(plain)}
