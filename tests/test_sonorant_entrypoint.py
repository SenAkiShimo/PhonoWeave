from __future__ import annotations

import sys

from phonoweave import entrypoint


def test_analyze_nasal_entrypoint_dispatch(monkeypatch) -> None:
    called = {}

    def fake_main(argv):
        called["argv"] = argv
        return 7

    monkeypatch.setattr(entrypoint, "nasal_main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        ["phonoweave", "analyze-nasal", "/tmp/voicebank", "--base", "n"],
    )

    assert entrypoint.main() == 7
    assert called["argv"] == ["/tmp/voicebank", "--base", "n"]


def test_sonorant_relevance_entrypoint_dispatch(monkeypatch) -> None:
    called = {}

    def fake_lateral(argv):
        called["lateral"] = argv
        return 3

    def fake_nasal(argv):
        called["nasal"] = argv
        return 5

    monkeypatch.setattr(entrypoint, "lateral_relevance_main", fake_lateral)
    monkeypatch.setattr(entrypoint, "nasal_relevance_main", fake_nasal)

    monkeypatch.setattr(
        sys,
        "argv",
        ["phonoweave", "analyze-lateral-relevance", "/tmp/voicebank"],
    )
    assert entrypoint.main() == 3
    assert called["lateral"] == ["/tmp/voicebank"]

    monkeypatch.setattr(
        sys,
        "argv",
        ["phonoweave", "analyze-nasal-relevance", "/tmp/voicebank", "--base", "n"],
    )
    assert entrypoint.main() == 5
    assert called["nasal"] == ["/tmp/voicebank", "--base", "n"]
