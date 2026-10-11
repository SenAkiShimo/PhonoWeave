from __future__ import annotations

import json

import pytest

from phonoweave.reclist import load_reclist, parse_reclist
from phonoweave.reclist_cli import main as cli_main


def test_lossless_multilingual_reclist() -> None:
    original = "\ufeff# title\r\na ka ki\r\nあ_か_き\r\n; note\r\n\r\n"
    document = parse_reclist(original)
    assert document.to_text() == original
    assert [line.number for line in document.recording_lines] == [2, 3]
    assert document.recording_lines[0].tokens == ("a", "ka", "ki")
    assert document.recording_lines[1].tokens == ("あ_か_き",)
    assert document.summary()["comment_lines"] == 2
    assert document.summary()["blank_lines"] == 1


def test_explicit_underscore_mode() -> None:
    document = parse_reclist("a_ka_ki   a\n", separator="underscore")
    assert document.recording_lines[0].tokens == ("a", "ka", "ki", "a")
    assert document.to_text() == "a_ka_ki   a\n"


def test_empty_reclist_and_comment_only() -> None:
    assert parse_reclist("").summary()["total_lines"] == 0
    assert parse_reclist("// hello\n # hi\n").summary()["recording_lines"] == 0


def test_utf8_bom_file(tmp_path) -> None:
    path = tmp_path / "list.txt"
    content = "\ufeffし しゃ\n"
    path.write_bytes(content.encode("utf-8"))
    doc = load_reclist(path)
    assert doc.encoding == "utf-8"
    assert doc.to_text() == content
    assert doc.recording_lines[0].tokens == ("し", "しゃ")


def test_cp932_file(tmp_path) -> None:
    path = tmp_path / "list.txt"
    path.write_bytes("か き く\r\n".encode("cp932"))
    doc = load_reclist(path)
    assert doc.encoding == "cp932"
    assert doc.recording_lines[0].tokens == ("か", "き", "く")
    assert doc.to_text().encode(doc.encoding) == path.read_bytes()


def test_invalid_separator() -> None:
    with pytest.raises(ValueError):
        parse_reclist("a b", separator="commas")  # type: ignore[arg-type]


def test_cli_json(tmp_path, capsys) -> None:
    path = tmp_path / "my_reclist.txt"
    path.write_text("a ka ki\n# note\n", encoding="utf-8")
    assert cli_main([str(path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["recording_lines"] == 1
    assert report["token_occurrences"] == 3
    assert report["unique_tokens"] == 3
