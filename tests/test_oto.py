from pathlib import Path
import unicodedata

from phonoweave.oto import load_oto, parse_oto_line


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


def test_load_oto_matches_normalized_wav_name(tmp_path: Path) -> None:
    composed = "café.wav"
    decomposed = unicodedata.normalize("NFD", composed)
    (tmp_path / decomposed).write_bytes(b"")
    oto_path = tmp_path / "oto.ini"
    oto_path.write_text(f"{composed}=sh u,0,80,0,40,10\n", encoding="utf-8")

    entries, warnings = load_oto(oto_path)

    assert warnings == []
    assert len(entries) == 1
    assert entries[0].wav_path.exists()
