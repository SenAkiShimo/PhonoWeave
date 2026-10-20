"""Tests for the offline, checksummed review workspace."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import pytest

from phonoweave.recording_workspace import (
    ARTIFACTS, load_workspace, render_workspace, write_workspace, main,
)


def _pack(tmp_path: Path, *, injected_name="fixture") -> Path:
    pack = tmp_path / "review_pack"
    pack.mkdir()
    names = ["a_ba"]
    action = {
        "schema": "phonoweave.recording_scheme_action_plan.v0.1",
        "speaker_id": injected_name,
        "consolidated_tasks": 1,
        "tasks": [{"task_id": "RS-001", "unit_id": "b_internal_other",
                   "review_action": "review_internal_candidate_mapping",
                   "pitch_groups": ["A#3", "D#4"], "suggested_syllables": ["ba"],
                   "source_examples": [], "priority": 2,
                   "verification": "Review original source"}],
    }
    reference = {
        "schema": "phonoweave.reference_reclist_blueprint.v0.1",
        "all_pitch_reference_sequences": 1,
    }
    budget = {
        "schema": "phonoweave.speaker_context_budget_plan.v0.1",
        "pitch_groups": ["A#3", "D#4"],
        "selection": {"selected_lines": names, "selected_line_count": 1},
    }
    pitches = ["A#3", "D#4"]
    layers = [
        {"pitch_group": p, "source_wav": f"{p}/_a_ba{p}.wav",
         "syllables": [{"syllable": "a", "syllable_index": 0},
                       {"syllable": "ba", "syllable_index": 1}]}
        for p in pitches
    ]
    line = {
        "schema": "phonoweave.recording_line_handoff.v0.1",
        "pitch_groups": pitches, "selected_line_count": 1,
        "recording_lines": [{"line_id": "RL-001", "sequence": "a_ba",
                             "pitch_layers": layers}],
    }
    review_items = []
    for pitch in pitches:
        for position, word in enumerate(["a", "ba"]):
            examples = []
            if position == 1:
                examples.append({"unit_id": "b_internal_other",
                                 "context": "internal:other",
                                 "evidence_kind": "source_linked_internal_VC_candidate",
                                 "original_alias": f"a b{pitch}",
                                 "source_oto": f"{pitch}/oto.ini",
                                 "oto_line_number": 10})
            review_items.append({
                "line_id": "RL-001", "pitch_group": pitch,
                "syllable_index": position, "syllable": word,
                "source_wav": f"{pitch}/_a_ba{pitch}.wav",
                "candidate_realizations": examples,
                "evidence_kind": "source_linked_internal_VC_candidate" if examples else "no_verified_role_link",
                "review_status": "provisional_split_requires_review" if examples else "no_supported_direct_unit_link",
                "annotator_action": "Inspect original OTO", "tuner_action": "Audition original alias",
                "source_oto_references": [f"{pitch}/oto.ini:10"] if examples else [],
            })
    directives = {
        "schema": "phonoweave.split_decision_directives.v0.1",
        "pitch_groups": pitches, "syllable_pitch_checks": 4,
        "source_linked_internal_candidates": 2,
        "directives": review_items,
    }
    objects = {
        ARTIFACTS[0]: action, ARTIFACTS[2]: reference,
        ARTIFACTS[3]: budget, ARTIFACTS[4]: line,
        ARTIFACTS[5]: directives,
    }
    hashes = {}
    for rel in ARTIFACTS:
        path = pack / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".csv"):
            with path.open("w", encoding="utf-8", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=["unit_id", "base_unit", "contexts",
                                                           "decision", "confidence"])
                writer.writeheader()
                writer.writerow({"unit_id": "b_internal_other", "base_unit": "b",
                                 "contexts": "internal:other", "decision": "split_recommended",
                                 "confidence": "moderate"})
        else:
            path.write_text(json.dumps(objects[rel], ensure_ascii=False), encoding="utf-8")
        hashes[rel] = sha256(path.read_bytes()).hexdigest()
    (pack / "review_pack_manifest.json").write_text(json.dumps({
        "schema": "phonoweave.recording_review_pack.v0.1",
        "artifact_sha256": hashes,
    }), encoding="utf-8")
    return pack


def test_workspace_has_real_line_cells_and_grouped_sources(tmp_path):
    data = load_workspace(_pack(tmp_path))
    assert data["summary"]["selected_lines"] == 1
    assert data["summary"]["source_cells"] == 4
    assert data["summary"]["linked_candidates"] == 2
    assert data["summary"]["naming_groups"] == 1
    assert len(data["tasks"]) == 1
    assert data["lines"][0]["sources"]["D#4"].endswith("D#4.wav")
    assert data["summary"]["production_ready"] is False


def test_artifact_tampering_refused(tmp_path):
    pack = _pack(tmp_path)
    path = pack / "directives/split_decision_directives.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_workspace(pack)


def test_mismatched_directive_rejected_even_with_updated_hash(tmp_path):
    pack = _pack(tmp_path)
    path = pack / "directives/split_decision_directives.json"
    source = json.loads(path.read_text())
    source["directives"][0]["source_wav"] = "forged.wav"
    path.write_text(json.dumps(source))
    manifest_path = pack / "review_pack_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"][ARTIFACTS[5]] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatched WAV"):
        load_workspace(pack)


def test_html_escapes_untrusted_script_payload(tmp_path):
    pack = _pack(tmp_path, injected_name="</script><script>evil()</script>")
    html = render_workspace(load_workspace(pack))
    assert "</script><script>evil()" not in html
    assert "\\u003c/script\\u003e" in html
    assert "录音方案工作台" in html
    assert "data-page=\"recordings\"" in html
    assert "phonoweave.workspace_notes.v0.1" in html


def test_export_is_outside_pack_and_never_overwrites(tmp_path):
    pack = _pack(tmp_path)
    output = tmp_path / "workspace.html"
    assert write_workspace(pack, output) == output
    assert output.stat().st_size > 10000
    with pytest.raises(FileExistsError):
        write_workspace(pack, output)
    with pytest.raises(ValueError, match="outside"):
        write_workspace(pack, pack / "workspace.html")


def test_cli_generates_standalone_html(tmp_path, capsys):
    pack = _pack(tmp_path)
    output = tmp_path / "review.html"
    assert main(["--pack", str(pack), "--output", str(output)]) == 0
    assert output.exists()
    assert "Recording workspace:" in capsys.readouterr().out
