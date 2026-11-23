from __future__ import annotations

from pathlib import Path

import pytest

from phonoweave import prefixmap
from phonoweave.bank_audit import audit_bank, format_audit
from phonoweave.layer_affixes import infer_layer_affixes

ONSETS = ["b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "z", "c", "s"]


def _write(folder: Path, prefix: str) -> None:
    folder.mkdir(parents=True)
    lines = "".join(f"{prefix}{o}a.wav={prefix}{o}a,0,0,0,0,0\n" for o in ONSETS)
    (folder / "oto.ini").write_text(lines, encoding="utf-8")


@pytest.fixture(autouse=True)
def _reset():
    prefixmap.set_layer_affix_inference(False)
    prefixmap._INFERRED_CACHE.clear()
    yield
    prefixmap.set_layer_affix_inference(False)
    prefixmap._INFERRED_CACHE.clear()


def _bank(tmp_path: Path) -> Path:
    bank = tmp_path / "bank"
    _write(bank / "one", "C")
    _write(bank / "two", "G")
    _write(bank / "plain", "")
    return bank


def test_each_folder_gets_its_own_prefix(tmp_path) -> None:
    found = {row.directory: row for row in infer_layer_affixes(_bank(tmp_path))}

    assert found["one"].prefix == "C"
    assert found["two"].prefix == "G"
    assert found["one"].readable_after > found["one"].readable_before


def test_plain_folder_is_left_alone(tmp_path) -> None:
    found = {row.directory for row in infer_layer_affixes(_bank(tmp_path))}

    assert "plain" not in found


def test_off_by_default(tmp_path) -> None:
    audit = audit_bank(_bank(tmp_path))

    assert audit["inferred_affixes"] == []
    assert audit["mandarin_cv_syllables"]["one"] < audit["mandarin_cv_syllables"]["plain"]


def test_switch_makes_every_folder_readable(tmp_path) -> None:
    prefixmap.set_layer_affix_inference(True)
    audit = audit_bank(_bank(tmp_path))

    counts = audit["mandarin_cv_syllables"]
    assert counts["one"] == counts["two"] == counts["plain"] == len(ONSETS)
    assert audit["usable_layers"] == ["one", "plain", "two"]
    assert "Affixes inferred per layer" in format_audit(audit)


def test_declared_prefix_map_is_not_overridden(tmp_path) -> None:
    bank = _bank(tmp_path)
    (bank / "prefix.map").write_text("C4\tC\t\n", encoding="utf-8")
    prefixmap.set_layer_affix_inference(True)

    found = {row.directory for row in infer_layer_affixes(bank, (("C", ""),))}

    assert "one" not in found
    assert "two" in found
