import csv
import json
from pathlib import Path

import pytest

from phonoweave.recording_line_handoff import (
    compile_line_handoff, write_line_handoff,
)


def fixture():
    pitches = ["A#3", "D#4", "G#4"]
    sequence = "a_ba_pa"
    blueprint = {
        "schema": "phonoweave.reference_reclist_blueprint.v0.1",
        "pitch_groups": pitches,
        "sequences": [{
            "sequence": sequence, "status": "all_pitch_groups_observed",
            "source_wavs": [{
                "pitch_group": p,
                "source_wav": f"{p}/_{sequence}{p}.wav",
                "oto_files": [f"{p}/oto.ini"],
            } for p in pitches],
        }],
    }
    budget = {
        "schema": "phonoweave.speaker_context_budget_plan.v0.1",
        "pitch_groups": pitches, "per_pitch_line_budget_max": 1,
        "protected_units": ["b_internal_other"],
        "unresolved_or_not_proxy_supported_units_not_optimized": ["p_internal_other"],
        "selection": {"selected_lines": [sequence], "selected_line_count": 1},
    }
    units = [{
        "unit_id": f"{base}_internal_other", "base_unit": base,
        "contexts": "internal:other",
    } for base in ("b", "p")]
    link = {
        "pitch_group": "A#3", "source_wav": "A#3/_a_ba_paA#3.wav",
        "source_oto": "A#3/oto.ini", "line_number": "12",
        "original_alias": "a bA#3", "boundary_order": "1",
        "syllable": "ba", "previous_syllable": "a", "next_syllable": "ba",
        "unit_id": "b_internal_other", "base_unit": "b",
        "context": "internal:other",
        "evidence": "ordered_reclist_filename_and_oto_boundary",
        "review_status": "candidate_link_not_synthesis_validated",
    }
    return blueprint, budget, units, [link]


def test_position_mapping_and_no_automatic_decisions():
    report = compile_line_handoff(*fixture())
    assert report["selected_line_count"] == 1
    assert report["source_line_syllable_cells"] == 9
    assert report["traceable_internal_candidates"] == 1
    assert report["protected_internal_units_with_candidate"] == ["b_internal_other"]
    assert not report["production_ready"]
    assert report["recordings_authorized"] == 0
    cells = report["recording_lines"][0]["pitch_layers"][0]["syllables"]
    assert cells[0]["internal_context_candidates"] == []
    candidate = cells[1]["internal_context_candidates"][0]
    assert candidate["original_alias"] == "a bA#3"
    assert candidate["review_status"] == "provisional_split"
    assert cells[2]["internal_context_candidates"] == []


def test_source_mismatch_and_unknown_unit_are_rejected():
    b, budget, u, links = fixture()
    links[0]["previous_syllable"] = "pa"
    with pytest.raises(ValueError, match="mismatch"):
        compile_line_handoff(b, budget, u, links)
    links[0]["previous_syllable"] = "a"
    links[0]["unit_id"] = "forged"
    with pytest.raises(ValueError, match="unknown unit"):
        compile_line_handoff(b, budget, u, links)


def test_partial_pitch_and_budget_error():
    b, budget, u, links = fixture()
    b["sequences"][0]["source_wavs"].pop()
    with pytest.raises(ValueError, match="pitch"):
        compile_line_handoff(b, budget, u, links)
    b, budget, u, links = fixture()
    budget["per_pitch_line_budget_max"] = 0
    with pytest.raises(ValueError, match="budget"):
        compile_line_handoff(b, budget, u, links)


def test_write_and_nonoverwrite(tmp_path):
    report = compile_line_handoff(*fixture())
    target = tmp_path / "handoff"
    write_line_handoff(report, target)
    rows = list(csv.DictReader((target / "ANNOTATOR_LINE_TARGETS.csv").open()))
    assert len(rows) == 9
    assert rows[1]["original_vc_aliases"] == "a bA#3"
    assert "ba" in (target / "ANNOTATOR_LINE_GUIDE.md").read_text()
    assert "a bA#3" in (target / "TUNER_LINE_GUIDE.md").read_text()
    assert json.loads((target / "recording_line_handoff.json").read_text()) == report
    with pytest.raises(FileExistsError):
        write_line_handoff(report, target)


def test_nonselected_invalid_link_is_skipped():
    b, budget, u, links = fixture()
    links[0]["source_wav"] = "A#3/_unselectedA#3.wav"
    result = compile_line_handoff(b, budget, u, links)
    assert result["traceable_internal_candidates"] == 0
