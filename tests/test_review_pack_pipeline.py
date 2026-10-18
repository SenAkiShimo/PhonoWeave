"""Tests for safe, atomic orchestration of the existing recording compilers."""
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from phonoweave import review_pack_pipeline as pipeline


def _populate_fake_pack(stage: Path) -> dict:
    for rel in pipeline.ARTIFACTS:
        artifact = stage / rel
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(rel + "\n", encoding="utf-8")
    return {
        "reference_sequences": 4,
        "selected_lines_per_pitch": 2,
        "pitch_groups": ["A#3", "D#4"],
        "syllable_pitch_checks": 12,
        "source_linked_internal_candidates": 2,
    }


def test_complete_pack_is_atomic_and_has_checksums(monkeypatch, tmp_path):
    bank = tmp_path / "fixture_source"
    bank.mkdir()
    (bank / "untouched.wav").write_bytes(b"unaltered")
    calls = []

    def fake(source, stage, budget, repeat_target):
        calls.append((source, budget, repeat_target))
        return _populate_fake_pack(stage)

    monkeypatch.setattr(pipeline, "_build_stages", fake)
    dest = tmp_path / "results" / "review_01"
    result = pipeline.compile_review_pack(bank, dest, budget=2, repeat_target=3)
    assert len(calls) == 1
    assert calls[0][1:] == (2, 3)
    assert result["export_status"] == "human_review_required"
    assert result["new_recordings_authorized"] == 0
    assert len(result["artifact_sha256"]) == len(pipeline.ARTIFACTS)
    assert json.loads((dest / "review_pack_manifest.json").read_text()) == result
    assert (bank / "untouched.wav").read_bytes() == b"unaltered"
    assert list((tmp_path / "results").glob(".phonoweave-review-pack-*")) == []
    with pytest.raises(FileExistsError):
        pipeline.compile_review_pack(bank, dest)


def test_failed_stage_preserves_absence_of_output(monkeypatch, tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()

    def fail(_source, stage, _budget, _repeat_target):
        (stage / "scheme").mkdir()
        (stage / "scheme" / "partial.txt").write_text("incomplete")
        raise ValueError("invalid OTO test case")

    monkeypatch.setattr(pipeline, "_build_stages", fail)
    dest = tmp_path / "review_out"
    with pytest.raises(ValueError, match="invalid OTO"):
        pipeline.compile_review_pack(bank, dest)
    assert not dest.exists()
    assert not list(tmp_path.glob(".phonoweave-review-pack-*"))


def test_missing_stage_artifact_rejects_publication(monkeypatch, tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    monkeypatch.setattr(pipeline, "_build_stages", lambda *args: {"unit": 1})
    out = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="missing required review artifact"):
        pipeline.compile_review_pack(bank, out)
    assert not out.exists()


def test_output_inside_source_and_invalid_budgets_are_rejected(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    with pytest.raises(ValueError, match="must not be inside"):
        pipeline.compile_review_pack(bank, bank / "derived")
    with pytest.raises(ValueError, match="budget"):
        pipeline.compile_review_pack(bank, tmp_path / "elsewhere", budget=-1)
    with pytest.raises(ValueError, match="repeat_target"):
        pipeline.compile_review_pack(bank, tmp_path / "elsewhere", repeat_target=0)


def test_cli_requires_valid_manifest_bank_pair(monkeypatch, tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    with pytest.raises(SystemExit, match="2"):
        pipeline.main(["--manifest", str(tmp_path / "manifest.json"),
                       "--output", str(tmp_path / "out")])
    captured = []

    def fake(source, output, *, budget, repeat_target):
        captured.append((source, output, budget, repeat_target))
        return {"counts": {"selected_lines_per_pitch": 3,
                           "syllable_pitch_checks": 18}}

    monkeypatch.setattr(pipeline, "compile_review_pack", fake)
    assert pipeline.main(["--voicebank", str(bank), "--output", str(tmp_path / "out")]) == 0
    assert captured[0][2:] == (60, 3)


def test_stage_interface_and_count_consistency(monkeypatch, tmp_path):
    """Exercise actual pipeline API wiring using deterministic stage doubles."""
    observations = []

    def module(name, **methods):
        instance = ModuleType("phonoweave." + name)
        for key, value in methods.items():
            setattr(instance, key, value)
        monkeypatch.setitem(sys.modules, "phonoweave." + name, instance)

    def write_scheme(_source, output):
        output.mkdir()
        (output / "alias_records.jsonl").write_text(json.dumps({"x": 1}) + "\n")
        (output / "recording_units_review.csv").write_text("unit_id\na\n")
        (output / "internal_context_links.csv").write_text("unit_id\na\n")
        observations.append("scheme")

    def write_reference(_reference, output):
        output.mkdir()
        observations.append("reference")

    def plan_budget(_reference, units, links, budget, *, repeat_target):
        assert units == [{"unit_id": "a"}]
        assert len(links) == 1 and budget == 2 and repeat_target == 3
        return {"selection": {"selected_line_count": 2}, "production_ready": False}

    def write_budget(_proposal, output):
        output.mkdir()
        observations.append("budget")

    def make_lines(_reference, _proposal, _units, _links):
        return {"selected_line_count": 2, "source_line_syllable_cells": 12,
                "traceable_internal_candidates": 3, "pitch_groups": ["A#3"],
                "production_ready": False}

    def write_lines(_lines, output):
        output.mkdir()
        observations.append("line")

    def make_directives(_lines, _proposal, _units):
        return {"syllable_pitch_checks": 12,
                "source_linked_internal_candidates": 3,
                "production_ready": False}

    def write_directives(_directives, output):
        output.mkdir()
        observations.append("directives")

    module("recording_scheme_review", build_from_voicebank=lambda _: "package",
           write_review_package=write_scheme)
    module("reference_reclist_blueprint",
           compile_reference_reclist=lambda _: {"all_pitch_reference_sequences": 8},
           write_reference_reclist=write_reference)
    module("speaker_context_budget", compile_speaker_context_budget=plan_budget,
           write_speaker_context_budget=write_budget)
    module("recording_line_handoff", compile_line_handoff=make_lines,
           write_line_handoff=write_lines)
    module("split_decision_compiler", compile_split_decisions=make_directives,
           write_split_decisions=write_directives)

    result = pipeline._build_stages(tmp_path / "bank", tmp_path, 2, 3)
    assert observations == list(pipeline.STAGES)
    assert result["selected_lines_per_pitch"] == 2
    assert result["syllable_pitch_checks"] == 12
