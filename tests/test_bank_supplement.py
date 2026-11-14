from __future__ import annotations

from phonoweave.bank_concordance import ConcordanceRow
from phonoweave.bank_supplement import SPLIT_STATUS, build_bank_supplement, uncovered_connections
from phonoweave.reclist import Syllable


def _syllables() -> tuple[Syllable, ...]:
    table = {
        "b": "a o ai ei an en ang u", "m": "a i u", "z": "a i u uo ui uan un",
        "s": "a i u uo",
    }
    classes = {"uo": "o", "ui": "ei", "uan": "an", "un": "en"}
    return tuple(
        Syllable(onset + final, onset, final, classes.get(final, final))
        for onset, finals in table.items()
        for final in finals.split()
    )


def _rows(status: str = SPLIT_STATUS) -> list[ConcordanceRow]:
    return [
        ConcordanceRow("z", "z_plain:i,o | z_rounded:u", "single", status, ("zw",), 8),
        ConcordanceRow("s", "single", "single", "neither splits"),
    ]


def test_every_missing_connection_is_covered_within_the_line_limit() -> None:
    syllables = _syllables()
    rows = _rows()
    supplement = build_bank_supplement(rows, syllables, max_syllables=6)

    tails = {item.tail for item in syllables}
    assert supplement.labels == ("zw",)
    assert supplement.connections == {(tail, "zw") for tail in tails}
    assert not uncovered_connections(supplement, rows, syllables)
    assert all(len(line) <= 6 for line in supplement.lines)


def test_rows_without_a_new_split_give_nothing() -> None:
    syllables = _syllables()
    for status in ("same axis", "different axis", "voicebank splits, we have no evidence"):
        assert build_bank_supplement(_rows(status), syllables).lines == ()


def test_stop_splits_with_no_vc_cost_are_skipped() -> None:
    rows = [ConcordanceRow("b", "b_i:i | b_u:u", "single", SPLIT_STATUS, ("bw",), 0)]
    assert build_bank_supplement(rows, _syllables()).lines == ()


def test_select_rows_limits_the_supplement_to_the_named_onsets() -> None:
    from phonoweave.bank_supplement import select_rows

    rows = [
        ConcordanceRow("z", "z_plain:i,o | z_rounded:u", "single", SPLIT_STATUS, ("zw",), 8),
        ConcordanceRow("s", "s_plain:i,o | s_rounded:u", "single", SPLIT_STATUS, ("sw",), 8),
    ]
    syllables = _syllables()

    assert select_rows(rows, None) == rows
    only = select_rows(rows, ["s"])
    assert [row.base for row in only] == ["s"]
    assert build_bank_supplement(only, syllables).labels == ("sw",)


def test_control_rows_add_unrecommended_onsets_but_not_stops_or_duplicates() -> None:
    from phonoweave.bank_supplement import control_rows

    syllables = _syllables()
    rows = _rows()
    made = control_rows(["s", "z", "b", "m"], syllables, rows)

    assert [row.base for row in made] == ["s", "m"]
    assert made[0].suggested == ("sw",) and made[1].suggested == ("my",)
    supplement = build_bank_supplement(rows + made, syllables)
    assert set(supplement.labels) == {"zw", "sw", "my"}
