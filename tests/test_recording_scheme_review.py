"""Recording scheme compilation: observed aliases, uncertainty, non-destructive outputs."""
from pathlib import Path
import csv
import json

import pytest

from phonoweave.diagnostic_selector import DiagnosticItem, DiagnosticSelection
from phonoweave.mandarin import MandarinObservation
from phonoweave.oto import OtoEntry
from phonoweave.recording_scheme_review import (
    compile_review_package, write_review_package,
)
from phonoweave.synthesis_inventory import SynthesisInventory, SynthesisUnit


def _unit(base="sh", identifier="sh_plain", context="plain",
          decision="split_recommended"):
    return SynthesisUnit(
        id=identifier, base_unit=base, contexts=(context,),
        source_group=identifier, canonical_context=None,
        decision=decision, confidence="low",
        acoustic_evidence="preliminary",
        synthesis_evidence="not_tested",
    )


def _entry(root, alias, number, wavname=None):
    oto = root / "pitch" / "oto.ini"
    wav = root / "pitch" / (wavname or f"take{number}.wav")
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"test fixture content")
    return OtoEntry(
        oto_path=oto, wav_path=wav, alias=alias,
        offset=10, consonant=60, cutoff=-25,
        preutterance=45, overlap=20, line_number=number,
    )


def _inventory(root, *units):
    return SynthesisInventory(
        speaker_id="Singer", language="mandarin",
        source_voicebank=root, units=tuple(units),
    )


def _obs(entry, base, final):
    return MandarinObservation(entry=entry, base_unit=base,
                              final=final, alias_token=base + final)


def _selection():
    item = DiagnosticItem(
        base_unit="sh", final="ua", syllable="shua",
        context_family="rounded", role_scope=None,
        existing_observations=0, replicate=1,
    )
    return DiagnosticSelection((item,), ())


def test_alias_classification_preserves_original_oto_values(tmp_path):
    a = _entry(tmp_path, "sh a", 1)
    b = _entry(tmp_path, "sh ua", 2)
    inv = _inventory(
        tmp_path, _unit(), _unit(identifier="sh_rounded", context="rounded")
    )
    package = compile_review_package(
        inv, [b, a], [_obs(a, "sh", "a"), _obs(b, "sh", "ua")], _selection()
    )
    assert [r["candidate_unit_id"] for r in package.alias_rows] == [
        "sh_plain", "sh_rounded"
    ]
    assert all(r["mapping_status"] == "candidate_review_only"
               for r in package.alias_rows)
    assert package.oto_rows[0]["offset_ms"] == 10
    assert package.oto_rows[0]["cutoff_ms"] == -25
    assert package.summary["oto_timing_changes"] == "none"
    assert package.supplement_text == "shua_a_shua_a\n"


def test_unresolved_never_becomes_production_decision(tmp_path):
    entry = _entry(tmp_path, "sh a", 1)
    inv = _inventory(tmp_path, _unit(decision="unresolved"))
    package = compile_review_package(inv, [entry], [_obs(entry, "sh", "a")],
                                     DiagnosticSelection((), ()))
    assert package.alias_rows[0]["mapping_status"] == "unresolved_review_only"
    assert package.oto_rows[0]["review_action"] == "manual_boundary_review"
    assert package.summary["status"] == "manual_review_required"


def test_identical_alias_conflicting_contexts_detected(tmp_path):
    a = _entry(tmp_path, "sh a", 1)
    b = _entry(tmp_path, "sh a", 2)
    inv = _inventory(
        tmp_path, _unit(), _unit(identifier="sh_rounded", context="rounded")
    )
    package = compile_review_package(
        inv, [a, b], [_obs(a, "sh", "a"), _obs(b, "sh", "ua")],
        DiagnosticSelection((), ())
    )
    assert package.summary["distinct_colliding_aliases"] == 1
    assert all(row["mapping_status"] == "alias_collision_review"
               for row in package.alias_rows)


def test_role_context_not_guessed_and_missing_wav_flagged(tmp_path):
    entry = _entry(tmp_path, "fa", 1)
    inv = _inventory(tmp_path, _unit("f", "f_internal_rounded",
                                    "internal:rounded"))
    package = compile_review_package(inv, [entry], [_obs(entry, "f", "u")],
                                     DiagnosticSelection((), ()))
    assert package.alias_rows[0]["mapping_status"] == "context_not_inferable"
    entry.wav_path.unlink()
    missing = compile_review_package(inv, [entry], [_obs(entry, "f", "u")],
                                     DiagnosticSelection((), ()))
    assert missing.alias_rows[0]["mapping_status"] == "missing_source_wav"


def test_unclassified_alias_and_x_front_context(tmp_path):
    other = _entry(tmp_path, "unclassified!", 1)
    x = _entry(tmp_path, "xi", 2)
    inv = _inventory(tmp_path, _unit("x", "x_front_unrounded",
                                    "front_unrounded"))
    package = compile_review_package(
        inv, [other, x], [_obs(x, "x", "i")], DiagnosticSelection((), ())
    )
    assert package.alias_rows[0]["mapping_status"] == "unclassified_alias"
    assert package.alias_rows[1]["candidate_unit_id"] == "x_front_unrounded"


def test_publish_complete_package_then_refuse_overwrite(tmp_path):
    root = tmp_path / "bank"
    root.mkdir()
    entry = _entry(root, "=HYPERLINK(1)", 1)
    inv = _inventory(root, _unit())
    package = compile_review_package(
        inv, [entry], [_obs(entry, "sh", "a")], _selection(), oto_warnings=2
    )
    original = entry.wav_path.read_bytes()
    output = tmp_path / "compiled"
    assert write_review_package(package, output) == output
    expected = {
        "README.txt", "summary.json", "synthesis_inventory.yaml",
        "recording_units_review.csv", "observed_alias_review.csv",
        "oto_timing_review.csv", "supplemental_targets.csv",
        "supplemental_reclist.txt",
    }
    assert {p.name for p in output.iterdir()} == expected
    summary = json.loads((output / "summary.json").read_text())
    assert summary["oto_warnings"] == 2
    with (output / "observed_alias_review.csv").open(newline="") as f:
        alias_row = next(csv.DictReader(f))
    assert alias_row["original_alias"].startswith("'=")
    with (output / "oto_timing_review.csv").open(newline="") as f:
        timing_row = next(csv.DictReader(f))
    assert timing_row["cutoff_ms"] == "-25"
    assert entry.wav_path.read_bytes() == original
    with pytest.raises(FileExistsError):
        write_review_package(package, output)


def test_package_can_be_empty_without_fabricated_reclist(tmp_path):
    inv = _inventory(tmp_path)
    package = compile_review_package(inv, [], [], DiagnosticSelection((), ()))
    assert package.plan_rows == []
    assert package.alias_rows == []
    assert package.supplement_rows == []
    assert package.supplement_text == ""
    assert package.summary["status"] == "manual_review_required"
