import copy
import pytest
from phonoweave.split_handoff import compile_split_handoff, write_split_handoff


def fixture():
    budget = {
        "schema": "phonoweave.speaker_context_budget_plan.v0.1",
        "pitch_groups": ["A#3"], "protected_units": ["sh_plain"],
        "unresolved_or_not_proxy_supported_units_not_optimized": ["n_plain"],
        "selection": {"selected_lines": ["a_sha"], "selected_line_count": 1},
    }
    units = [
        {"unit_id": "sh_plain", "base_unit": "sh", "contexts": "plain",
         "decision": "split_recommended", "confidence": "high"},
        {"unit_id": "n_plain", "base_unit": "n", "contexts": "plain",
         "decision": "unresolved", "confidence": "low"},
    ]
    actions = {
        "schema": "phonoweave.recording_scheme_action_plan.v0.1",
        "speaker_id": "voice", "tasks": [
            {"task_id": "RS-001", "unit_id": "sh_plain", "source_examples": []}
        ],
    }
    key = {"source_oto": "A#3/oto.ini", "line_number": 8,
           "original_alias": "shaA#3"}
    records = [{
        "schema": "phonoweave.lossless_alias_record.v0.1",
        "source": {**key, "source_wav": "A#3/_a_shaA#3.wav"},
        "role": {**key, "pitch_suffix": "A#3",
                 "pitch_suffix_status": "matched", "role": "standalone_cv"},
        "compilation_preview": {**key, "candidate_unit_id": "sh_plain",
                                "compile_status": "roundtrip_review_only"},
    }]
    return budget, units, actions, records


def test_handoff_contains_both_roles_and_abstains(tmp_path):
    args = fixture()
    plan = compile_split_handoff(*args)
    assert plan["unit_count"] == 2
    assert plan["provisional_split_units"] == 1
    assert not plan["production_ready"]
    sh = next(x for x in plan["units"] if x["unit_id"] == "sh_plain")
    assert sh["pitch_evidence"][0]["original_cv_examples"][0]["original_alias"] == "shaA#3"
    n = next(x for x in plan["units"] if x["unit_id"] == "n_plain")
    assert n["status"] == "unresolved_do_not_merge"
    out = tmp_path / "guide"
    write_split_handoff(plan, out)
    assert "shaA#3" in (out / "ANNOTATOR_SPLIT_AND_OTO_GUIDE.md").read_text()
    assert "sh_plain" in (out / "TUNER_ALIAS_AND_CONTEXT_GUIDE.md").read_text()
    with pytest.raises(FileExistsError):
        write_split_handoff(plan, out)


def test_provenance_mismatch_rejected():
    budget, units, actions, records = fixture()
    records[0]["role"]["line_number"] = 100
    with pytest.raises(ValueError, match="provenance"):
        compile_split_handoff(budget, units, actions, records)


def test_unknown_unit_rejected():
    budget, units, actions, records = fixture()
    actions["tasks"][0]["unit_id"] = "forged"
    with pytest.raises(ValueError, match="unknown"):
        compile_split_handoff(budget, units, actions, records)
