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
