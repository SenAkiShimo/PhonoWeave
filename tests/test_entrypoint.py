from __future__ import annotations

import sys
from types import SimpleNamespace

from phonoweave import entrypoint
from phonoweave.cli import build_parser


def test_affricate_cli_accepts_j_and_q() -> None:
    parser = build_parser()
    for base in ("j", "q"):
        args = parser.parse_args(["analyze-affricate", "/tmp/voicebank", "--base", base])
        assert args.base == base


def test_build_profile_entrypoint(monkeypatch, tmp_path, capsys) -> None:
    output = tmp_path / "speaker_profile.yaml"
    called = {}

    def fake_write(voicebank, target):
        called["voicebank"] = voicebank
        called["target"] = target
        target.write_text("schema_version: 2\n", encoding="utf-8")
        return SimpleNamespace(
            speaker_id="TestSinger",
            language="mandarin",
            realizations=(1, 2),
        )

    monkeypatch.setattr(entrypoint, "write_speaker_profile", fake_write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phonoweave",
            "build-profile",
            "/tmp/voicebank",
            "-o",
            str(output),
        ],
    )

    assert entrypoint.main() == 0
    assert output.exists()
    assert str(called["voicebank"]) == "/tmp/voicebank"
    assert called["target"] == output
    stdout = capsys.readouterr().out
    assert "Speaker: TestSinger" in stdout
    assert "Realizations: 2" in stdout


def test_build_synthesis_inventory_entrypoint(monkeypatch, tmp_path, capsys) -> None:
    output = tmp_path / "synthesis_inventory.yaml"
    called = {}

    def fake_write(voicebank, target):
        called["voicebank"] = voicebank
        called["target"] = target
        target.write_text("schema_version: 1\n", encoding="utf-8")
        return SimpleNamespace(
            speaker_id="TestSinger",
            language="mandarin",
            units=(1, 2, 3),
        )

    monkeypatch.setattr(entrypoint, "write_synthesis_inventory", fake_write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phonoweave",
            "build-synthesis-inventory",
            "/tmp/voicebank",
            "-o",
            str(output),
        ],
    )

    assert entrypoint.main() == 0
    assert output.exists()
    assert str(called["voicebank"]) == "/tmp/voicebank"
    assert called["target"] == output
    stdout = capsys.readouterr().out
    assert "Speaker: TestSinger" in stdout
    assert "Synthesis units: 3" in stdout
