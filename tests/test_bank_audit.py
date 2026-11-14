from __future__ import annotations

from pathlib import Path

from phonoweave.bank_audit import audit_bank, classify, format_audit


def _bank(root: Path) -> Path:
    bank = root / "bank"
    (bank / "D4").mkdir(parents=True)
    (bank / "D5").mkdir()
    (bank / "prefix.map").write_text("D4\tD4\t\nD5\tD5\t\n", encoding="utf-8")
    d4 = [
        "ba", "- ba", "ma", "- ma", "a m", "a sh", "a shw", "an sh", "an m", "ang R", "a R",
    ]
    d5 = ["ba", "- ba", "ma", "a m", "a sh", "an sh", "an m", "a R"]
    (bank / "D4" / "oto.ini").write_text(
        "".join(f"x.wav=D4{a},0,0,0,0,0\n" for a in d4), encoding="utf-8"
    )
    (bank / "D5" / "oto.ini").write_text(
        "".join(f"x.wav=D5{a},0,0,0,0,0\n" for a in d5), encoding="utf-8"
    )
    return bank


def test_classify_alias_kinds() -> None:
    assert classify("ba") == ("cv", ("ba",))
    assert classify("- ba") == ("initial", ("ba",))
    assert classify("an shw") == ("vc", ("an", "shw"))
    assert classify("ang R") == ("end", ("ang",))
    assert classify("a b c") is None


def test_audit_reports_cross_layer_gaps(tmp_path) -> None:
    audit = audit_bank(_bank(tmp_path))

    assert audit["layers"] == ["D4", "D5"]
    assert audit["census"]["D4"]["vc"] == 5
    assert audit["cross_layer_missing"]["initial"] == {"D5": ["ma"]}
    assert audit["cross_layer_missing"]["vc"] == {"D5": ["a shw"]}
    assert audit["cross_layer_missing"]["end"] == {"D5": ["ang"]}
    assert audit["finals_without_end"] == ["an"]
    assert {row["label"] for row in audit["systematic_labels"]} == {"m", "sh"}
    assert [row["label"] for row in audit["partial_labels"]] == ["shw"]
    assert "D5 vc: 1 missing: a shw" in format_audit(audit)


def test_audit_reports_entries_without_audio(tmp_path) -> None:
    bank = _bank(tmp_path)
    (bank / "D4" / "x.wav").write_bytes(b"")

    audit = audit_bank(bank)

    assert audit["missing_wav"] == {"D5": 8}
    assert audit["missing_wav_examples"] == {"D5": ["x.wav"]}
    assert "audio file is missing" in format_audit(audit)
