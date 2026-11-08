"""Synthetic capture protocol tests: no singing/split efficacy claims."""
from __future__ import annotations

import csv
from copy import deepcopy
from datetime import date
from hashlib import sha256
import json
import math
from pathlib import Path
import wave

import numpy as np
import pytest

from phonoweave.vc36_capture import (
    SCHEMA, TARGETS, REPEATS, make_plan, audit_manifest,
    ingest_session, write_plan, write_cue_sheet, apply_boundary_csv,
)


def _pcm(path: Path, freq: float, *, rate: int=48000, duration: float=2.5):
    path.parent.mkdir(parents=True,exist_ok=True)
    x = np.arange(round(rate*duration)) / rate
    tone = 0.11*np.sin(2*np.pi*freq*x)
    with wave.open(str(path),"wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((tone*32767).astype("<i2").tobytes())


def _prepared(tmp_path):
    root=tmp_path/"corpus"
    root.mkdir()
    manifest=root/"manifest.json"
    plan=write_plan(manifest,speaker_id="test_singer",pitch_hz=293.66)
    for i,sid in enumerate(("S01","S02")):
        source=tmp_path/f"source_{sid}"
        source.mkdir()
        for j,r in enumerate(sorted([r for r in plan["recordings"] if r["session_id"]==sid], key=lambda t:t["take_id"])):
            _pcm(source/(r["take_id"]+".wav"), 210+j*13+i*2)
    return root,manifest,plan


def _ingest(manifest,src,session,day):
    return ingest_session(manifest,src,session_id=session,capture_date=day,
                          device_id="Rode48K",room_id="same_room",gain_setting="-10dB",
                          distance_cm=18.0,setup_reset=True)


def test_exact_36_grid_and_frozen_checksum(tmp_path):
    a=make_plan(speaker_id="pilot")
    b=make_plan(speaker_id="pilot")
    assert a==b
    assert len(a["recordings"])==36
    assert len({r["take_id"] for r in a["recordings"]})==36
    assert set((r["left_label"],r["right_label"],r["prompt"]) for r in a["recordings"])==set(TARGETS)
    assert set(r["presentation_order"] for r in a["recordings"] if r["session_id"]=="S01")==set(range(1,19))
    changed=deepcopy(a)
    changed["frozen_score"]["pitch_hz"]=220
    assert "frozen_protocol_modified:protocol_sha256" in audit_manifest(changed,tmp_path)["errors"]


def test_plan_and_cues_are_exclusive(tmp_path):
    root,manifest,plan=_prepared(tmp_path)
    with pytest.raises(FileExistsError):
        write_plan(manifest,speaker_id="test_singer")
    cue=tmp_path/"cues.csv"
    write_cue_sheet(plan,cue)
    assert len(list(csv.DictReader(cue.open())))==36
    with pytest.raises(FileExistsError):
        write_cue_sheet(plan,cue)


def test_s02_blocked_until_complete_and_later_day(tmp_path):
    root,manifest,plan=_prepared(tmp_path)
    with pytest.raises(ValueError,match="S01 capture"):
        _ingest(manifest,tmp_path/"source_S02","S02","2026-11-02")
    s01=_ingest(manifest,tmp_path/"source_S01","S01","2026-11-01")
    assert s01["imported"]==18
    with pytest.raises(ValueError,match="later calendar day"):
        _ingest(manifest,tmp_path/"source_S02","S02","2026-11-01")
    s02=_ingest(manifest,tmp_path/"source_S02","S02","2026-11-02")
    assert s02["imported"]==18
    report=audit_manifest(json.loads(manifest.read_text()),root)
    assert report["capture_complete"]
    assert not report["analysis_ready"]
    assert len(report["unannotated"])==36
    assert report["verified_wav_files"]==36
    assert report["outcome_status"]=="not_evaluated"


def test_duplicate_source_blocks_ingestion_with_no_partial_writes(tmp_path):
    root,manifest,plan=_prepared(tmp_path)
    folder=tmp_path/"source_S01"
    names=sorted(folder.glob("*.wav"))
    names[1].write_bytes(names[0].read_bytes())
    with pytest.raises(ValueError,match="duplicated take"):
        _ingest(manifest,folder,"S01","2026-11-01")
    assert not (root/"audio").exists()
    assert all(r["wav_sha256"] is None for r in json.loads(manifest.read_text())["recordings"])


def test_mutated_registered_audio_blocks_resume(tmp_path):
    root,manifest,plan=_prepared(tmp_path)
    _ingest(manifest,tmp_path/"source_S01","S01","2026-11-01")
    r=next(r for r in json.loads(manifest.read_text())["recordings"] if r["session_id"]=="S01")
    (root/r["relative_wav"]).write_bytes(b"modified")
    with pytest.raises(ValueError,match="altered"):
        _ingest(manifest,tmp_path/"source_S01","S01","2026-11-01")
    assert not audit_manifest(json.loads(manifest.read_text()),root)["capture_complete"]


def test_exact_boundary_review_hashes_required(tmp_path):
    root,manifest,plan=_prepared(tmp_path)
    _ingest(manifest,tmp_path/"source_S01","S01","2026-11-01")
    _ingest(manifest,tmp_path/"source_S02","S02","2026-11-02")
    data=json.loads(manifest.read_text())
    output=tmp_path/"annotations.csv"
    fields=["take_id","source_wav_sha256","transition_onset_ms","transition_offset_ms","boundary_reviewer","qc_manual_reviewed"]
    with output.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for r in data["recordings"]:
            writer.writerow({"take_id":r["take_id"],"source_wav_sha256":r["wav_sha256"],
                             "transition_onset_ms":800,"transition_offset_ms":1020,
                             "boundary_reviewer":"human_a","qc_manual_reviewed":"true"})
    result=apply_boundary_csv(manifest,output)
    assert result["annotations_accepted"]==36
    report=audit_manifest(json.loads(manifest.read_text()),root)
    assert report["analysis_ready"]
    assert report["speaker_adaptive_effect"]=="not_assessed"
    with pytest.raises(ValueError,match="existing reviewed boundary"):
        rows=list(csv.DictReader(output.open()))
        rows[0]["transition_onset_ms"]="900"
        with output.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader();writer.writerows(rows)
        apply_boundary_csv(manifest,output)


def test_malicious_path_in_frozen_manifest_rejected(tmp_path):
    root,manifest,plan=_prepared(tmp_path)
    forged=deepcopy(plan)
    forged["recordings"][0]["relative_wav"]="../../private.wav"
    report=audit_manifest(forged,root)
    assert not report["capture_complete"]
    assert any("frozen_slot_modified" in e for e in report["errors"])


def test_missing_or_uncertain_is_never_assessed(tmp_path):
    root,manifest,plan=_prepared(tmp_path)
    report=audit_manifest(plan,root)
    assert not report["analysis_ready"]
    assert report["speaker_adaptive_effect"]=="not_assessed"
    assert len(report["missing"])==36
