from __future__ import annotations
import json
import pytest
from phonoweave.calibration_dual_labels import save_dual_label, load_dual_labels, copy_legacy_label, dual_path
from phonoweave.calibration_manual_labels import save_manual_label
from test_calibration_manual_labels import _session


def _beeps(monkeypatch):
    class Event:
        def __init__(self, ms): self.time_ms=ms
    class Sequence:
        events=tuple(Event(250+833.333*i) for i in range(8))
    monkeypatch.setattr("phonoweave.calibration_manual_labels.detect_beep_sequence",
                        lambda samples, rate, token_count: Sequence())


def test_two_anchor_types_saved_independently(tmp_path, monkeypatch):
    _beeps(monkeypatch)
    session=_session(tmp_path)
    save_dual_label(session,12,1,"consonant_onset",-130,"ok")
    p=save_dual_label(session,12,1,"cv_boundary",120,"ok")
    a=p["labels"]["12"]["occurrences"]["1"]
    assert a["consonant_onset"]["anchor_ms_after_cue"] == -130
    assert a["cv_boundary"]["anchor_ms_after_cue"] == 120
    save_dual_label(session,12,1,"cv_boundary",None,"unset")
    assert "consonant_onset" in load_dual_labels(session)["labels"]["12"]["occurrences"]["1"]
    assert "cv_boundary" not in load_dual_labels(session)["labels"]["12"]["occurrences"]["1"]


def test_legacy_is_read_only_and_import_requires_explicit_action(tmp_path, monkeypatch):
    _beeps(monkeypatch)
    session=_session(tmp_path)
    save_manual_label(session,12,1,110,"uncertain")
    old_path=session/"analysis"/"calibration_manual_anchor_labels_v0.2.json"
    old=old_path.read_bytes()
    assert not dual_path(session).exists()
    p=copy_legacy_label(session,12,1,"cv_boundary")
    assert old_path.read_bytes()==old
    entry=p["labels"]["12"]["occurrences"]["1"]["cv_boundary"]
    assert entry["status"]=="uncertain"
    assert entry["source"]=="legacy_reviewed_copy"
    with pytest.raises(ValueError,match="already exists"):
        copy_legacy_label(session,12,1,"cv_boundary")


def test_invalid_anchor_and_nonfinite(tmp_path, monkeypatch):
    _beeps(monkeypatch)
    session=_session(tmp_path)
    with pytest.raises(ValueError,match="anchor type"):
        save_dual_label(session,12,1,"wrong",0,"ok")
    with pytest.raises(ValueError,match="finite"):
        save_dual_label(session,12,1,"consonant_onset",float("nan"),"ok")
