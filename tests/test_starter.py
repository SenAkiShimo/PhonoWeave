from __future__ import annotations

import json
import sys

from phonoweave import entrypoint
from phonoweave.starter import starter_inventory, starter_syllables
from phonoweave.starter_cli import main as starter_main


def test_starter_syllables_cover_standard_inventory() -> None:
    syllables = starter_syllables()
    texts = {item.text for item in syllables}

    assert 380 <= len(syllables) <= 420
    assert {"ba", "zhi", "zi", "ju", "xue", "nv", "lve", "wu", "yu"} <= texts
    assert {item.tail for item in syllables} == {
        "a", "ai", "an", "ang", "ao", "e", "e0", "ei", "en", "en0", "eng",
        "i", "i0", "in", "ing", "ir", "o", "ong", "ou", "u", "v", "vn",
    }
    zhi = next(item for item in syllables if item.text == "zhi")
    assert zhi.tail == "ir"
    assert next(item for item in syllables if item.text == "si").tail == "i0"
    assert next(item for item in syllables if item.text == "bi").tail == "i"


def test_starter_inventory_splits_only_where_rules_say() -> None:
    bases = {unit.base_unit for unit in starter_inventory().units}
    assert bases == {"m", "n", "l", "s", "sh", "h", "j", "q", "x"}


def test_starter_cli_writes_reclist(tmp_path, capsys) -> None:
    output = tmp_path / "starter.txt"

    assert starter_main(["-o", str(output), "--restarts", "5"]) == 0

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines and all(len(line.split("_")) <= 10 for line in lines)
    report = json.loads((tmp_path / "starter.report.json").read_text(encoding="utf-8"))
    assert report["unmapped_contexts"] == []
    assert (tmp_path / "starter.units.yaml").exists()
    assert "not tested on any speaker" in capsys.readouterr().out


def test_starter_entrypoint(monkeypatch) -> None:
    called = {}

    def fake_main(argv):
        called["argv"] = argv
        return 0

    monkeypatch.setattr(entrypoint, "starter_main", fake_main)
    monkeypatch.setattr(sys, "argv", ["phonoweave", "starter-reclist", "-o", "/tmp/s.txt"])
    assert entrypoint.main() == 0
    assert called["argv"] == ["-o", "/tmp/s.txt"]
