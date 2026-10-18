"""Self-contained fixtures: no voicebank identities or external WAVs."""
import csv
import json

import pytest

from phonoweave.split_decision_compiler import (
    compile_split_decisions, write_split_decisions,
)


def fixture():
    pitches = ["A#3", "D#4"]
    seq = "sha_ba_pa"
    budget = {
        "schema": "phonoweave.speaker_context_budget_plan.v0.1",
        "pitch_groups": pitches,
        "protected_units": ["sh_plain", "b_internal_other"],
        "unresolved_or_not_proxy_supported_units_not_optimized": ["p_internal_other"],
        "selection": {"selected_lines": [seq], "selected_line_count": 1},
    }
    units = [
        {"unit_id": "sh_plain", "base_unit": "sh", "contexts": "plain",
         "decision": "split_recommended"},
        {"unit_id": "b_internal_other", "base_unit": "b",
         "contexts": "internal:other", "decision": "split_recommended"},
        {"unit_id": "p_internal_other", "base_unit": "p",
         "contexts": "internal:other", "decision": "unresolved"},
    ]
    layers = []
    for pitch in pitches:
        links = [{
            "unit_id": "b_internal_other", "context": "internal:other",
            "source_oto": f"{pitch}/oto.ini", "oto_line_number": 12,
            "original_alias": f"a b{pitch}", "review_status": "provisional_split",
        }]
        layers.append({
            "pitch_group": pitch, "source_wav": f"{pitch}/_{seq}{pitch}.wav",
            "syllables": [
                {"syllable_index": i, "syllable": word,
                 "position": "sequence_initial" if i == 0 else "sequence_internal",
                 "internal_context_candidates": links if i == 1 else []}
                for i, word in enumerate(seq.split("_"))
            ],
        })
    handoff = {
        "schema": "phonoweave.recording_line_handoff.v0.1",
        "pitch_groups": pitches,
        "selected_line_count": 1,
        "source_line_syllable_cells": 6,
        "traceable_internal_candidates": 2,
        "recording_lines": [{"line_id": "RL-001", "sequence": seq, "pitch_layers": layers}],
    }
    return handoff, budget, units


def test_internal_and_initial_contexts_stay_distinct():
    result = compile_split_decisions(*fixture())
    assert result["syllable_pitch_checks"] == 6
    assert result["source_linked_internal_candidates"] == 2
    cells = result["directives"][:3]
    assert cells[0]["evidence_kind"] == "syllable_context_only"
    assert cells[0]["review_status"] == "provisional_split_requires_review"
    assert cells[0]["candidate_realizations"][0]["source_oto"] is None
    assert cells[1]["review_status"] == "provisional_split_requires_review"
    assert cells[1]["candidate_realizations"][0]["source_oto"] == "A#3/oto.ini"
    assert cells[2]["review_status"] == "no_internal_role_link_do_not_infer_absence"
    assert result["automatic_split_or_merge_approvals"] == 0
    assert not result["production_ready"]


def test_deferred_link_remains_unresolved():
    handoff, budget, units = fixture()
    link = handoff["recording_lines"][0]["pitch_layers"][0]["syllables"][2]
    link["internal_context_candidates"].append({
        "unit_id": "p_internal_other", "context": "internal:other",
        "source_oto": "A#3/oto.ini", "oto_line_number": 42,
        "original_alias": "a pA#3", "review_status": "unresolved_no_automatic_merge",
    })
    handoff["traceable_internal_candidates"] = 3
    row = compile_split_decisions(handoff, budget, units)["directives"][2]
    assert row["review_status"] == "unresolved_keep_separate_pending_review"
    assert row["production_routing"] is None


def test_rejects_forged_internal_context_and_bad_counts():
    handoff, budget, units = fixture()
    handoff["recording_lines"][0]["pitch_layers"][0]["syllables"][1][
        "internal_context_candidates"][0]["context"] = "internal:u_series"
    with pytest.raises(ValueError, match="mismatch"):
        compile_split_decisions(handoff, budget, units)
    handoff, budget, units = fixture()
    handoff["traceable_internal_candidates"] = 999
    with pytest.raises(ValueError, match="candidate total mismatch"):
        compile_split_decisions(handoff, budget, units)


def test_ambiguous_candidates_block_routing():
    handoff, budget, units = fixture()
    units.append({"unit_id": "sh_plain_alt", "base_unit": "sh",
                  "contexts": "plain", "decision": "unresolved"})
    budget["unresolved_or_not_proxy_supported_units_not_optimized"].append("sh_plain_alt")
    result = compile_split_decisions(handoff, budget, units)
    assert result["directives"][0]["review_status"] == "ambiguous_context_hold"
    assert result["directives"][0]["production_routing"] is None


def test_files_non_overwriting_and_numeric_csv(tmp_path):
    report = compile_split_decisions(*fixture())
    output = tmp_path / "split"
    write_split_decisions(report, output)
    saved = json.loads((output / "split_decision_directives.json").read_text())
    assert saved["source_linked_internal_candidates"] == 2
    with (output / "ANNOTATOR_SPLIT_DIRECTIVES.csv").open(newline="") as f:
        entries = list(csv.DictReader(f))
    assert len(entries) == 6
    assert entries[1]["source_oto_references"] == "A#3/oto.ini:12"
    assert "OTO" in (output / "ANNOTATOR_SPLIT_GUIDE.md").read_text()
    assert "调教" in (output / "TUNER_CONTEXT_CHECKS.md").read_text()
    with pytest.raises(FileExistsError):
        write_split_decisions(report, output)


def test_rejects_mismatched_budget_lines():
    handoff, budget, units = fixture()
    budget["selection"]["selected_lines"] = ["different"]
    with pytest.raises(ValueError, match="sequence ordering"):
        compile_split_decisions(handoff, budget, units)
