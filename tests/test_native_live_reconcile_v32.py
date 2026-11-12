"""v3.2 end-to-end: native supplement decisions against live WAV/OTO evidence.

Tests use synthetic long-WAV voicebank files; real corpus acceptance still pending.
"""
from __future__ import annotations

import json
import threading
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from phonoweave import gui
from phonoweave.native_live_reconcile_v32 import (
    SCHEMA, build_reconciled_native_plan, reconcile_native_live,
)
from phonoweave.native_live_assets_v31 import scan_live_voicebank
from phonoweave.native_recording_plan_v20 import (
    TARGET_SCHEMA, build_native_recording_workflow,
)
from phonoweave.recording_plan_page_v19 import HTML
from phonoweave.recording_workflow_v18 import export_recording_workflow


def write_wav(path: Path, v: int = 100):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(44100)
        w.writeframes(v.to_bytes(2, "little", signed=True) * 800)


def sample(tmp_path):
    root=tmp_path/"voice";d4=root/"D4";d4.mkdir(parents=True)
    write_wav(d4/"long.wav")
    write_wav(d4/"orphan.wav", 300)
    write_wav(d4/"candidate.wav", 350)
    (d4/"broken.wav").write_bytes(b"broken")
    (d4/"oto.ini").write_text(
        "long.wav=D4i a,0,0,0,0,0\n"
        "long.wav=D4ir a,10,0,0,0,0\n"
        "long.wav=D4i0 a,20,0,0,0,0\n"
        "missing.wav=D4i sh,0,0,0,0,0\n"
        "broken.wav=D4ir sh,0,0,0,0,0\n"
        "candidate.wav=D4i m,0,0,0,0,0\n"
        "long.wav=D4i m,0,0,0,0,0\n",
        encoding="utf-8",
    )
    targets=tmp_path/"targets.json"
    targets.write_text(json.dumps({
        "schema":TARGET_SCHEMA, "bank_alias_prefixes":{"D4":"D4"},
        "targets":[
            {"line":"i a","subbanks":["D4"]},
            {"line":"ir a","subbanks":["D4"]},
            {"line":"i0 a","subbanks":["D4"]},
            {"line":"i sh","subbanks":["D4"]},
            {"line":"ir sh","subbanks":["D4"]},
            {"line":"i m","subbanks":["D4"]},
            {"line":"orphan","wav":"orphan.wav","subbanks":["D4"]},
            {"line":"no proof","subbanks":["D4"]},
        ]
    }, ensure_ascii=False),encoding="utf-8")
    return root,targets


def test_true_long_wav_source_agrees_without_duplicate_recordings(tmp_path):
    root,targets=sample(tmp_path)
    result=build_reconciled_native_plan(root,expected_reclist=targets,max_lines=1)
    assert result["source_consistency_review"]["schema"]==SCHEMA
    audit=result["source_consistency_review"]
    assert audit["native_template_check_count"]==8
    assert audit["file_level_agreement_count"]==3
    assert audit["possible_inconsistency_count"]==0
    assert audit["result"]=="no_detected_file_level_discrepancy_but_reviews_remain"
    assert audit["live_oto_entries_read"]==7
    assert audit["live_unreferenced_wav_routes"]==1
    assert result["recording_plan"]["recording_lines"]==["i sh"]
    assert audit["no_new_recording_lines_from_reconciliation"] is True
    assert audit["automatic_unit_merges_or_splits"]==[]


def test_gap_corrupt_ambiguous_and_unregistered_are_not_conflated(tmp_path):
    root,targets=sample(tmp_path)
    result=build_reconciled_native_plan(root,expected_reclist=targets)
    rows={x["original_recording_line"]:x
          for x in result["source_consistency_review"]["checks"]}
    assert rows["i sh"]["result"]=="source_file_missing"
    assert rows["ir sh"]["result"]=="source_repair_review"
    assert rows["i m"]["result"]=="source_choice_review"
    assert rows["orphan"]["result"]=="oto_or_source_review"
    assert rows["no proof"]["result"]=="source_mapping_review"
    assert rows["i sh"]["live_has_explicit_missing_WAV"]
    assert result["recording_plan"]["recording_lines"]==["i sh"]
    assert not any(r["automatic_new_recording_authorized"] for r in rows.values())


def test_missing_actual_pcm_after_plan_is_flagged_not_scheduled(tmp_path):
    root,targets=sample(tmp_path)
    plan=build_native_recording_workflow(root,expected_reclist=targets)
    # Simulate the voicebank changing between the planner and live scan.
    (root/"D4"/"long.wav").unlink()
    audit=reconcile_native_live(plan,scan_live_voicebank(root),targets)
    assert audit["possible_inconsistency_count"]>=3
    assert plan["recording_plan"]["recording_lines"]==["i sh"]
    assert audit["original_voicebank_modified"] is False


def test_different_root_rejected(tmp_path):
    root,targets=sample(tmp_path)
    plan=build_native_recording_workflow(root,expected_reclist=targets)
    other=tmp_path/"other"
    write_wav(other/"D4"/"x.wav")
    with pytest.raises(ValueError,match="different voicebank"):
        reconcile_native_live(plan,scan_live_voicebank(other),targets)


def test_already_scheduled_missing_item_stays_out_of_new_reclist(tmp_path):
    root,targets=sample(tmp_path)
    scheduled=tmp_path/"scheduled.txt";scheduled.write_text("i sh\n",encoding="utf-8")
    result=build_reconciled_native_plan(root,expected_reclist=targets,
                                       existing_reclist=scheduled)
    assert result["recording_plan"]["recording_lines"]==[]
    assert result["source_consistency_review"]["native_template_check_count"]==8
    assert result["recording_plan"]["deferred_or_skipped"][0]["reason"]=="already_scheduled_not_verified_audio"


def test_export_three_existing_files_preserves_consistency_and_no_reclist_change(tmp_path):
    root,targets=sample(tmp_path)
    r=build_reconciled_native_plan(root,expected_reclist=targets)
    folder=export_recording_workflow(r,tmp_path/"export",voicebank=root)
    assert (folder/"supplement_reclist.txt").read_text(encoding="utf-8")=="i sh\n"
    restored=json.loads((folder/"recording_plan.json").read_text(encoding="utf-8"))
    assert restored["source_consistency_review"]==r["source_consistency_review"]
    assert "ir sh" in (folder/"recording_review.txt").read_text(encoding="utf-8")
    assert (root/"D4"/"long.wav").is_file()


def post(base,endpoint,data):
    req=Request(base+endpoint,data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type":"application/json"})
    try:
        with urlopen(req,timeout=12) as response:
            return response.status,json.loads(response.read())
    except HTTPError as e:
        return e.code,json.loads(e.read())


def test_gui_preview_includes_consistency_and_does_not_change_scheduled_lines(tmp_path):
    root,targets=sample(tmp_path)
    with gui._STATE_LOCK:
        gui._STATE.recording_report=None
    server=ThreadingHTTPServer(("127.0.0.1",0),gui._Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        url=f"http://127.0.0.1:{server.server_address[1]}"
        status,result=post(url,"/api/recording-plan/preview",{
            "path":str(root),"native_reclist":str(targets),"max_lines":2})
        assert status==200
        assert result["source_consistency_review"]["file_level_agreement_count"]==3
        assert result["recording_plan"]["recording_lines"]==["i sh"]
        assert 'id="consistency-metrics"' in HTML
        assert 'id="consistency-rows"' in HTML
        assert "innerHTML=" not in HTML
    finally:
        server.shutdown();thread.join(timeout=5);server.server_close()
        with gui._STATE_LOCK:
            gui._STATE.recording_report=None


def test_existing_wav_without_oto_never_proves_new_phone(tmp_path):
    root=tmp_path/"bank"
    write_wav(root/"D4"/"unregistered.wav")
    (root/"D4"/"oto.ini").write_text("",encoding="utf-8")
    target=tmp_path/"targets.txt";target.write_text("unregistered\n",encoding="utf-8")
    report=build_reconciled_native_plan(root,expected_reclist=target)
    assert report["recording_plan"]["recording_lines"]==[]
    assert report["source_consistency_review"]["checks"][0]["result"]=="oto_or_source_review"
    assert report["source_consistency_review"]["no_new_recording_lines_from_reconciliation"]
