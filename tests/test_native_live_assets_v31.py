"""v3.1 current voicebank source routing: actual PCM, OTO and GUI smoke tests."""
from __future__ import annotations

import csv
import io
import json
import threading
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from phonoweave import gui
from phonoweave.native_asset_gui_v30 import asset_routes_csv, asset_ledger_summary
from phonoweave.native_live_assets_v31 import SCHEMA, UNRESOLVED, scan_live_voicebank
from phonoweave.native_recording_plan_v20 import TARGET_SCHEMA
from phonoweave.recording_plan_page_v19 import HTML


def audio(path: Path, value: int = 100):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(value.to_bytes(2, "little", signed=True) * 1600)


def bank(tmp_path):
    root=tmp_path/"voice"
    d4=root/"D4";d4.mkdir(parents=True)
    audio(d4/"long.wav")
    audio(d4/"copy.wav")
    audio(d4/"unused.wav",value=300)
    (d4/"broken.wav").write_bytes(b"not a WAV")
    (d4/"oto.ini").write_text(
        "long.wav=D4i a,0,0,0,0,0\n"
        "long.wav=D4ir a,20,0,0,0,0\n"
        "long.wav=D4i0 a,40,0,0,0,0\n"
        "long.wav=D4i a,50,0,0,0,0\n" # repeated alias, not repeated PCM
        "missing.wav=D4i sh,0,0,0,0,0\n"
        "broken.wav=D4ir sh,0,0,0,0,0\n"
        "other.wav=D4ir k,0,0,0,0,0\n"
        "copy.wav=D4i a,0,0,0,0,0\n",encoding="utf-8")
    m=tmp_path/"targets.json"
    m.write_text(json.dumps({
        "schema":TARGET_SCHEMA,
        "bank_alias_prefixes":{"D4":"D4"},
        "targets":[{"line":"i a","subbanks":["D4"]}]
    }),encoding="utf-8")
    return root,m


def test_live_long_wav_multi_alias_counts_one_pcm(tmp_path):
    root,manifest=bank(tmp_path)
    result=scan_live_voicebank(root,labels=["i","ir","i0"],native_reclist=manifest)
    assert result["schema"]==SCHEMA
    assert result["source_archive_sha256"] is None
    assert result["declared_prefix_banks"]==["D4"]
    assert result["oto_entries_read"]==8
    routes=result["original_alias_routes"]
    r=next(x for x in routes if x["original_unit_label"]=="ir" and x["following_context"]=="a")
    assert r["original_aliases"]==["D4ir a"]
    assert r["routing_status"]=="single_current_oto_source"
    assert r["observed_distinct_pcm_assets"]==1
    assert result["multiple_source_alias_routes"]==1
    assert any(x["original_unit_label"]=="i" and x["following_context"]=="a"
               and x["routing_status"]=="multiple_observed_source_options_review" for x in routes)
    assert len(result["exact_pcm_duplicate_path_groups"])==1
    assert result["source_sound_modified"] is False
    assert result["proposed_new_recording_lines"]==[]


def test_bad_wav_missing_wav_and_unreferenced_are_review_not_rerecord(tmp_path):
    root,manifest=bank(tmp_path)
    r=scan_live_voicebank(root)
    status={tuple(x["original_aliases"]):x["routing_status"]
            for x in r["original_alias_routes"] if x["original_aliases"]}
    assert status[("D4i sh",)]=="missing_referenced_wav"
    assert status[("D4ir sh",)]=="unreadable_referenced_wav"
    assert status[("D4ir k",)]=="missing_referenced_wav"
    assert r["unreferenced_wav_routes"]==1
    assert r["proposed_new_recording_lines"]==[]
    assert all(x["original_unit_label"]==UNRESOLVED
               for x in r["original_alias_routes"])


def test_strict_prefix_is_not_guessed(tmp_path):
    root,_=bank(tmp_path)
    r=scan_live_voicebank(root,labels=["i","ir","i0"])
    assert r["unresolved_alias_routes"]==len(r["original_alias_routes"])
    assert r["original_user_labels"]==["i","ir","i0"]


def test_cannot_declare_bogus_prefix_subbank(tmp_path):
    root,_=bank(tmp_path)
    m=tmp_path/"bogus.json"
    m.write_text(json.dumps({"schema":TARGET_SCHEMA,"bank_alias_prefixes":{"Z9":"Z9"},
                            "targets":[{"line":"x"}]}),encoding="utf-8")
    with pytest.raises(ValueError,match="prefix subbank"):
        scan_live_voicebank(root, native_reclist=m)


def test_cross_bank_reference_flags_review(tmp_path):
    root,_=bank(tmp_path)
    a3=root/"A3";a3.mkdir()
    (a3/"oto.ini").write_text("../D4/long.wav=i,0,0,0,0,0\n",encoding="utf-8")
    r=scan_live_voicebank(root,labels=["i"])
    x=next(x for x in r["original_alias_routes"] if x["bank"]=="A3")
    assert x["routing_status"]=="cross_subbank_wav_reference_review"
    assert x["observed_source_options"][0]["source_pcm_sha256"]


def test_outside_wav_is_not_scanned_or_leaked(tmp_path):
    root,_=bank(tmp_path)
    outside=tmp_path/"outside.wav";audio(outside)
    with (root/"D4"/"oto.ini").open("a",encoding="utf-8") as f:
        f.write("../../outside.wav=D4x sh,0,0,0,0,0\n")
    report=scan_live_voicebank(root)
    route=next(x for x in report["original_alias_routes"] if x["original_aliases"]==["D4x sh"])
    assert route["routing_status"]=="source_outside_voicebank_not_scanned"
    assert route["observed_source_options"][0]["source_pcm_sha256"] is None
    assert "outside.wav" not in route["observed_source_options"][0]["source_wav"]


def test_scanner_can_inventory_unregistered_wavs_without_oto(tmp_path):
    root=tmp_path/"voice"
    audio(root/"D4"/"unregistered.wav")
    r=scan_live_voicebank(root)
    assert r["oto_entries_read"]==0
    assert r["unreferenced_wav_routes"]==1
    assert r["distinct_existing_bank_PCM_source_containers"]==1
    assert r["approved_alias_merges"]==[]


def test_duplicate_PCM_across_different_paths_is_not_independent_take(tmp_path):
    root=tmp_path/"voice"
    audio(root/"A3"/"a.wav")
    audio(root/"A3"/"b.wav")
    r=scan_live_voicebank(root)
    assert r["distinct_existing_bank_PCM_source_containers"]==1
    assert len(r["exact_pcm_duplicate_path_groups"])==1


def test_live_csv_and_summary(tmp_path):
    root,m=bank(tmp_path)
    r=scan_live_voicebank(root, labels=["i","ir","i0"],native_reclist=m)
    summary=asset_ledger_summary(r)
    assert summary["live_voicebank_scan"] is True
    assert summary["historic_index_not_live_voicebank_validation"] is False
    assert summary["unreferenced_wav_routes"]==1
    rows=list(csv.DictReader(io.StringIO(asset_routes_csv(r))))
    assert rows and "source_status" in rows[0]
    assert any(x["original_aliases"]=="D4ir a" and x["original_unit_label"]=="ir" for x in rows)


def send(base,path,body=None):
    req=Request(base+path,data=json.dumps(body).encode() if body is not None else None,
                headers={"Content-Type":"application/json"} if body is not None else {})
    try:
        with urlopen(req,timeout=8) as r:
            return r.status,r.read()
    except HTTPError as e:
        return e.code,e.read()


@pytest.fixture
def server(tmp_path):
    with gui._STATE_LOCK:
        gui._STATE.asset_routing_report=None
        gui._STATE.recording_report={"recording_plan":{"recording_lines":["already planned"]}}
    s=ThreadingHTTPServer(("127.0.0.1",0),gui._Handler)
    worker=threading.Thread(target=s.serve_forever,daemon=True);worker.start()
    try:
        yield f"http://127.0.0.1:{s.server_address[1]}"
    finally:
        s.shutdown();worker.join(timeout=5);s.server_close()
        with gui._STATE_LOCK:
            gui._STATE.asset_routing_report=None
            gui._STATE.recording_report=None


def test_live_gui_preview_and_csv_download_keeps_reclist(tmp_path,server):
    root,m=bank(tmp_path)
    status,body=send(server,"/api/recording-plan/assets/preview",
                     {"mode":"live","voicebank":str(root),
                      "native_reclist":str(m),"labels":["i","ir","i0"]})
    assert status==200
    report=json.loads(body)
    assert report["live_voicebank_scan"] is True
    assert report["multiple_source_alias_routes"] if "multiple_source_alias_routes" in report else report["routes_requiring_source_choice_review"]>=1
    status,csv_bytes=send(server,"/api/recording-plan/assets/download/csv")
    assert status==200
    assert b"D4ir a" in csv_bytes
    with gui._STATE_LOCK:
        assert gui._STATE.recording_report["recording_plan"]["recording_lines"]==["already planned"]


def test_failing_live_scan_clears_old_asset_download(tmp_path,server):
    root,m=bank(tmp_path)
    send(server,"/api/recording-plan/assets/preview",{"mode":"live","voicebank":str(root),"labels":[]})
    status,_=send(server,"/api/recording-plan/assets/preview",
                  {"mode":"live","voicebank":str(tmp_path/"missing"),"labels":[]})
    assert status==400
    status,_=send(server,"/api/recording-plan/assets/download/json")
    assert status==409


def test_gui_has_live_and_historical_modes():
    assert 'id="asset-mode"' in HTML
    assert 'value="live" selected' in HTML
    assert 'value="historical"' in HTML
    assert "native_reclist" in HTML
    assert "innerHTML=" not in HTML
    assert "invalidateAssets" in HTML
