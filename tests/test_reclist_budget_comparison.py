"""Integration tests for defensible matched-line recording budget comparisons."""
from __future__ import annotations

import csv
from hashlib import sha256
import io
import json

import pytest

from phonoweave.reclist_budget_comparison import (
    compare_literal_budget, load_verified_review_pack, write_comparison,
)
from phonoweave.reclist_budget_comparison_cli import main as cli_main
from phonoweave.speaker_context_budget import compile_speaker_context_budget


def case():
    pitches = ["A#3", "D#4"]
    names = ["a_ba_pa", "a_ba_ma", "o_bo_po", "u_bu_pu"]
    sequences = [
        {
            "sequence": n, "status": "all_pitch_groups_observed",
            "pitch_groups_observed": pitches,
            "pitch_groups_not_observed": [],
            "source_wavs": [
                {"pitch_group": p, "source_wav": f"{p}/_{n}{p}.wav",
                 "oto_files": [f"{p}/oto.ini"]}
                for p in pitches
            ],
        }
        for n in names
    ]
    blueprint = {
        "schema": "phonoweave.reference_reclist_blueprint.v0.1",
        "pitch_groups": pitches, "sequences": sequences,
        "interpretable_sequences": len(names),
        "all_pitch_reference_sequences": len(names),
        "reference_reclist": names,
    }
    units = [
        {"unit_id": "b_internal_other", "base_unit": "b",
         "contexts": "internal:other", "decision": "split_recommended",
         "confidence": "moderate", "synthesis_evidence": "supported_under_proxy"},
        {"unit_id": "b_internal_u", "base_unit": "b",
         "contexts": "internal:u_series", "decision": "unresolved",
         "confidence": "low", "synthesis_evidence": "not_tested"},
    ]
    links = [{
        "unit_id": "b_internal_other", "base_unit": "b",
        "context": "internal:other", "pitch_group": "A#3",
        "source_oto": "A#3/oto.ini",
        "source_wav": "A#3/_a_ba_paA#3.wav", "line_number": "8",
        "boundary_order": "1", "syllable": "ba",
        "previous_syllable": "a", "next_syllable": "ba",
        "evidence": "ordered_reclist_filename_and_oto_boundary",
        "review_status": "candidate_link_not_synthesis_validated",
    }]
    plan = compile_speaker_context_budget(blueprint, units, links, 2)
    return blueprint, plan, units, links


def test_matched_prefix_budget_and_provisional_metrics():
    report = compare_literal_budget(*case())
    speaker = report["policies"]["speaker"]["selected_sequences"]
    literal = report["policies"]["literal"]["selected_sequences"]
    assert set(speaker) <= set(case()[0]["reference_reclist"])
    assert set(literal) <= set(case()[0]["reference_reclist"])
    assert report["matched_curve"][0]["speaker"]["lines_per_pitch"] == 0
    for row in report["matched_curve"]:
        k = row["matched_lines_per_pitch"]
        assert row["speaker"]["lines_per_pitch"] == k
        assert row["literal"]["lines_per_pitch"] == k
        assert row["speaker"]["source_wav_references"] == 2 * k
        assert row["literal"]["source_wav_references"] == 2 * k
        assert (row["speaker_minus_literal_weighted_depth"]
                == row["speaker"]["weighted_provisional_context_depth"]
                - row["literal"]["weighted_provisional_context_depth"])
    assert len(report["matched_curve"]) == min(len(speaker), len(literal)) + 1
    assert report["production_alias_export"] == "disabled"
    assert report["recordings_authorized"] == 0
    assert compare_literal_budget(*case()) == report


def test_forged_speaker_selection_rejected():
    blueprint, speaker, units, links = case()
    speaker["selection"]["selected_lines"] = []
    with pytest.raises(ValueError, match="recomputed"):
        compare_literal_budget(blueprint, speaker, units, links)


def test_bad_link_and_weights_rejected():
    blueprint, speaker, units, links = case()
    links[0]["syllable"] = "not_a_source_token"
    with pytest.raises(ValueError, match="boundary"):
        compare_literal_budget(blueprint, speaker, units, links)
    with pytest.raises(ValueError, match="weight"):
        compare_literal_budget(*case(), pair_weight=0)


def test_zero_budget_and_no_eligible_lines():
    blueprint, _, units, links = case()
    empty = compile_speaker_context_budget(blueprint, units, links, 0)
    report = compare_literal_budget(blueprint, empty, units, links)
    assert len(report["matched_curve"]) == 1
    assert report["policies"]["literal"]["selected_sequences"] == []
    assert report["matched_curve"][0]["speaker"]["weighted_provisional_context_depth"] == 0


def _fixture_pack(tmp_path):
    blueprint, speaker, units, links = case()
    root = tmp_path / "pack"
    root.mkdir()
    files = {
        "reference/reference_reclist_blueprint.json": (
            json.dumps(blueprint, ensure_ascii=False) + "\n"),
        "budget/speaker_context_budget_plan.json": (
            json.dumps(speaker, ensure_ascii=False) + "\n"),
    }
    for key, rows in (("scheme/recording_units_review.csv", units),
                      ("scheme/internal_context_links.csv", links)):
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        files[key] = buf.getvalue()
    hashes = {}
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        hashes[name] = sha256(path.read_bytes()).hexdigest()
    (root / "review_pack_manifest.json").write_text(
        json.dumps({"schema": "phonoweave.recording_review_pack.v0.1",
                    "artifact_sha256": hashes}), encoding="utf-8")
    return root


def test_pack_hash_guard_and_cli_no_overwrite(tmp_path, capsys):
    root = _fixture_pack(tmp_path)
    blueprint, plan, units, links, provenance = load_verified_review_pack(root)
    assert len(provenance["source_artifact_sha256"]) == 4
    assert compare_literal_budget(blueprint, plan, units, links)["matched_curve"]
    out = tmp_path / "comparison"
    assert cli_main(["--pack", str(root), "--output", str(out)]) == 0
    assert (out / "matched_curve.csv").is_file()
    assert (out / "literal_baseline_reclist.txt").is_file()
    assert (out / "matched_budget_comparison.json").is_file()
    with pytest.raises(SystemExit) as exc:
        cli_main(["--pack", str(root), "--output", str(out)])
    assert exc.value.code == 2
    (root / "scheme/internal_context_links.csv").write_text(
        "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_review_pack(root)


def test_no_output_inside_review_pack(tmp_path):
    root = _fixture_pack(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli_main(["--pack", str(root), "--output", str(root / "inside")])
    assert exc.value.code == 2


def test_unsafe_symlink_source_refused(tmp_path):
    root = _fixture_pack(tmp_path)
    bad = root / "scheme/recording_units_review.csv"
    external = tmp_path / "external.csv"
    external.write_bytes(bad.read_bytes())
    bad.unlink()
    bad.symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        load_verified_review_pack(root)
