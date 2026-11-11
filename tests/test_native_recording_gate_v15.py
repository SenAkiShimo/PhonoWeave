"""Synthetic safety and provenance tests; not phone/synthesis accuracy tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phonoweave.native_recording_gate_v15 import assess, run, V13, V14

SHA = "a" * 64
DIGEST = "b" * 64


def fixture():
    v13 = {
        "schema": V13, "source_archive_sha256": SHA,
        "equal_donor_budget_and_train_only_donors": True,
        "development": {"selection_used_audit": False},
        "verified_phone_identity_or_boundary": False,
        "automatic_unit_split_merge_authorized": False,
        "audit_previously_explored_in_prior_studies_not_confirmatory": True,
        "source_groups_by_split": {"train": 14, "validation": 5, "audit": 4},
        "clip_inventory": {"complete_three_stage_clips": 167},
        "audit_descriptive": {"paired_source_groups": 4, "matched_observations": 28,
                              "pooled_group_mean_mse": 0.5, "context_group_mean_mse": 0.6,
                              "relative_improvement": -0.2},
    }
    v14 = {
        "schema": V14, "v9_archive_sha256": SHA,
        "no_automatic_inventory_split_merge": True,
        "not_perceptual_or_singing_synthesis_quality": True,
        "audit_previous_iterations_already_seen_retrospective_only": True,
        "audio_source": {"source_zip_sha256_verified": True},
        "frozen_rule": {"feature_fit": "train_only",
                        "selection_access_to_target_right": False,
                        "candidate_budget": 4},
        "source_groups_by_partition": {"train": 14, "validation": 5, "audit": 4},
        "complete_clips": 167,
        "audit": {"groups": 4, "observations": 28, "donor_budget_equal": 4,
                  "trajectory_mse": {"pooled": 0.5, "seam_aware": 0.6,
                                     "relative_gain_positive_is_better": -0.2},
                  "predicted_seam_cost": {"pooled": 0.5, "seam_aware": 0.4,
                                          "relative_gain_positive_is_better": 0.2},
                  "splice_max_sample_step_over_rms": {"pooled": 0.5, "seam_aware": 0.6,
                                                      "relative_gain_positive_is_better": -0.2}},
    }
    return v13, v14


def check(a, b):
    return assess(a, b, digest13=DIGEST, digest14=DIGEST)


def test_realistic_negative_proxy_cannot_generate_any_recording():
    x, y = fixture()
    result = check(x, y)
    assert result["finding"] == "current_context_split_not_supported"
    assert result["recording_advice"]["new_units_authorized"] is False
    assert result["recording_advice"]["new_reclist_lines"] == []
    assert result["recording_advice"]["automatic_alias_changes"] == []
    assert result["recording_advice"]["existing_voicebank"] == "unchanged"


def test_positive_proxy_is_still_not_deployable():
    x, y = fixture()
    x["audit_descriptive"].update(context_group_mean_mse=0.4, relative_improvement=0.2)
    y["audit"]["trajectory_mse"].update(seam_aware=0.4, relative_gain_positive_is_better=0.2)
    z = check(x, y)
    assert z["finding"] == "mixed_or_exploratory_proxy_only"
    assert z["recording_advice"]["new_units_authorized"] is False


def test_rejects_audio_source_mismatch():
    x, y = fixture()
    y["v9_archive_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="different audio sources"):
        check(x, y)


def test_rejects_test_split_mismatch():
    x, y = fixture()
    y["source_groups_by_partition"]["audit"] = 3
    with pytest.raises(ValueError, match="different group split"):
        check(x, y)


def test_rejects_audit_denominator_change():
    x, y = fixture()
    y["audit"]["observations"] = 27
    with pytest.raises(ValueError, match="denominator"):
        check(x, y)


def test_rejects_metric_misreporting():
    x, y = fixture()
    y["audit"]["trajectory_mse"]["relative_gain_positive_is_better"] = 0.9
    with pytest.raises(ValueError, match="contradictory"):
        check(x, y)


def test_rejects_nonfinite_losses():
    x, y = fixture()
    y["audit"]["trajectory_mse"]["seam_aware"] = float("nan")
    with pytest.raises(ValueError, match="invalid"):
        check(x, y)


def test_rejects_audit_leakage():
    x, y = fixture()
    x["development"]["selection_used_audit"] = True
    with pytest.raises(ValueError, match="leakage"):
        check(x, y)


def test_rejects_right_hand_target_leakage():
    x, y = fixture()
    y["frozen_rule"]["selection_access_to_target_right"] = True
    with pytest.raises(ValueError, match="leakage"):
        check(x, y)


def test_rejects_claim_of_verified_phone_from_upstream():
    x, y = fixture()
    x["verified_phone_identity_or_boundary"] = True
    with pytest.raises(ValueError, match="provenance"):
        check(x, y)


def test_existing_coverage_workflow_not_overridden():
    x, y = fixture()
    assert check(x, y)["recording_advice"]["separate_existing_coverage_based_supplement_planner"] == "unmodified"


def test_end_to_end_writes_checksum_bound_non_overwriting_json(tmp_path: Path):
    x, y = fixture()
    p = tmp_path / "v13.json"; q = tmp_path / "v14.json"
    p.write_text(json.dumps(x)); q.write_text(json.dumps(y))
    out = tmp_path / "gate.json"
    report = run(p, q, out)
    assert json.loads(out.read_text()) == report
    assert report["input_report_sha256"]["v13"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert report["input_report_sha256"]["v14"] == hashlib.sha256(q.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        run(p, q, out)


def test_invalid_source_does_not_create_output(tmp_path: Path):
    x, y = fixture(); y["v9_archive_sha256"] = "0" * 64
    p = tmp_path / "v13.json"; q = tmp_path / "v14.json"
    p.write_text(json.dumps(x)); q.write_text(json.dumps(y))
    with pytest.raises(ValueError):
        run(p, q, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()
