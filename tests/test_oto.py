from pathlib import Path

from phonoweave.oto import parse_oto_line


def test_parse_oto_line() -> None:
    entry = parse_oto_line(
        "sample.wav=sh u,12.5,80,20,45,10",
        Path("/tmp/oto.ini"),
        3,
    )

    assert entry.alias == "sh u"
    assert entry.offset == 12.5
    assert entry.consonant == 80
    assert entry.cutoff == 20
    assert entry.preutterance == 45
    assert entry.overlap == 10
    assert entry.line_number == 3
