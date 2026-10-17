"""CLI validation for direct voicebank and existing local replication manifest."""
import json
from types import SimpleNamespace

import pytest

from phonoweave import recording_scheme_review_cli as cli


def _manifest(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    path = tmp_path / "voicebanks.json"
    path.write_text(json.dumps({
        "schema": "phonoweave.voicebank_replication_manifest.v1",
        "voicebanks": [
            {"bank_id": "bank05", "speaker_group_id": "speaker05",
             "path": str(bank)},
        ],
    }), encoding="utf-8")
    return path, bank


def test_select_existing_bank_by_id(tmp_path):
    path, bank = _manifest(tmp_path)
    assert cli.voicebank_from_manifest(path, "bank05") == bank.resolve()


def test_nonexistent_id_rejected(tmp_path):
    path, _ = _manifest(tmp_path)
    with pytest.raises(ValueError, match="missing or duplicated"):
        cli.voicebank_from_manifest(path, "bank06")


def test_duplicate_id_rejected(tmp_path):
    path, _ = _manifest(tmp_path)
    data = json.loads(path.read_text())
    data["voicebanks"].append(dict(data["voicebanks"][0]))
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing or duplicated"):
        cli.voicebank_from_manifest(path, "bank05")


def test_unknown_schema_rejected(tmp_path):
    path, _ = _manifest(tmp_path)
    data = json.loads(path.read_text())
    data["schema"] = "unknown"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        cli.voicebank_from_manifest(path, "bank05")


def test_main_selects_existing_cohort_without_overwriting(monkeypatch, tmp_path):
    path, bank = _manifest(tmp_path)
    output = tmp_path / "review"
    called = []
    def fake_build(root):
        called.append(root)
        return SimpleNamespace(summary={
            "speaker_id": "test_singer", "units": 2, "oto_entries": 12,
            "suggested_diagnostic_lines": 3
        })
    def fake_write(package, target):
        target.mkdir()
        return target
    monkeypatch.setattr(cli, "build_from_voicebank", fake_build)
    monkeypatch.setattr(cli, "write_review_package", fake_write)
    argv = ["--manifest", str(path), "--bank-id", "bank05",
            "--output", str(output)]
    assert cli.main(argv) == 0
    assert called == [bank.resolve()]
    assert output.is_dir()
    with pytest.raises(SystemExit):
        cli.main(argv)


def test_main_requires_complete_source_selection(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["--manifest", str(tmp_path / "no.json"),
                  "--output", str(tmp_path / "review")])
