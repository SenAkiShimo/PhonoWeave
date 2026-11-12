"""v3.0 GUI read-only source routing tests, no new recording decisions."""
from __future__ import annotations

import csv
import io
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from phonoweave import gui
from phonoweave import native_asset_routing_v29
from phonoweave.native_asset_gui_v30 import asset_routes_csv, asset_ledger_summary
from phonoweave.recording_plan_page_v19 import HTML


def sample_ledger():
    return native_asset_routing_v29.build_asset_routing_ledger([
        {"bank": bank, "symbol": symbol, "context": context,
         "source_wav": wav, "source_pcm_sha256": digest * 64,
         "source_alias": f"{bank}{symbol} {context}", "features": [0.0] * 12}
        for bank, context, wav, digest in [
            ("D4", "sh", "long.wav", "a"),
            ("C4", "a", "other.wav", "b"),
        ]
        for symbol in ("i", "ir", "i0")
    ] + [
        {"bank": "D4", "symbol": "i", "context": "sh",
         "source_wav": "alternative.wav", "source_pcm_sha256": "c"*64,
         "source_alias": "D4i sh", "features": [1.0]*12},
    ], ["i", "ir", "i0"], archive_sha256="d"*64)


def send(base, path, data=None, origin=None):
    headers={}
    if data is not None: headers["Content-Type"]="application/json"
    if origin is not None: headers["Origin"]=origin
    req=Request(base+path, data=None if data is None else json.dumps(data).encode(),
                headers=headers)
    try:
        with urlopen(req, timeout=12) as response:
            return response.status, response.read(), response.headers
    except HTTPError as exc:
        return exc.code, exc.read(), exc.headers


@pytest.fixture
def running_server(monkeypatch):
    calls=[]
    ledger=sample_ledger()
    def fake_run(archive, labels):
        calls.append((str(archive),list(labels)))
        assert labels==["i","ir","i0"]
        return ledger
    monkeypatch.setattr(native_asset_routing_v29,"run_archive",fake_run)
    with gui._STATE_LOCK:
        gui._STATE.asset_routing_report=None
        gui._STATE.recording_report={"recording_plan":{"recording_lines":["untouched"]}}
    server=ThreadingHTTPServer(("127.0.0.1",0),gui._Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    try:
        yield "http://127.0.0.1:"+str(server.server_address[1]), calls
    finally:
        server.shutdown();worker.join(timeout=5);server.server_close()
        with gui._STATE_LOCK:
            gui._STATE.asset_routing_report=None
            gui._STATE.recording_report=None


def test_asset_ledger_not_available_without_preview(running_server):
    base,_=running_server
    for name in ("json","csv"):
        status,body,_=send(base,"/api/recording-plan/assets/download/"+name)
        assert status==409 and b"Preview" in body


def test_preview_read_only_preserves_original_recording_plan(running_server):
    base,calls=running_server
    status,body,_=send(base,"/api/recording-plan/assets/preview",
        {"archive":"/tmp/old_source.zip","labels":["i","ir","i0"]})
    assert status==200
    result=json.loads(body)
    assert result["historic_index_not_live_voicebank_validation"] is True
    assert result["no_new_recording_or_alias_changes"] is True
    assert result["total_existing_alias_context_routes"]==6
    assert result["distinct_existing_bank_PCM_source_containers"]==3
    assert result["routes_requiring_source_choice_review"]==1
    assert calls==[("/tmp/old_source.zip",["i","ir","i0"])]
    with gui._STATE_LOCK:
        assert gui._STATE.recording_report["recording_plan"]["recording_lines"]==["untouched"]


def test_download_csv_and_json_include_original_aliases(running_server):
    base,_=running_server
    send(base,"/api/recording-plan/assets/preview",
         {"archive":"/tmp/old_source.zip","labels":["i","ir","i0"]})
    status,body,h=send(base,"/api/recording-plan/assets/download/csv")
    assert status==200 and h.get_content_type()=="text/csv"
    records=list(csv.DictReader(io.StringIO(body.decode("utf-8"))))
    assert len(records)==7
    assert records[0]["original_unit_label"] in ["i","ir","i0"]
    assert any(r["original_aliases"]=="D4i sh" for r in records)
    assert "attachment" in h["Content-Disposition"]
    status,body,h=send(base,"/api/recording-plan/assets/download/json")
    assert status==200 and h.get_content_type()=="application/json"
    assert json.loads(body)["approved_unit_substitutions"]==[]
    assert "attachment" in h["Content-Disposition"]


def test_failed_preview_invalidates_old_downloads(running_server):
    base,_=running_server
    send(base,"/api/recording-plan/assets/preview",
         {"archive":"/tmp/old_source.zip","labels":["i","ir","i0"]})
    status,_,_=send(base,"/api/recording-plan/assets/preview",
                     {"archive":"","labels":["i","ir","i0"]})
    assert status==400
    status,_,_=send(base,"/api/recording-plan/assets/download/json")
    assert status==409
    with gui._STATE_LOCK:
        assert gui._STATE.recording_report["recording_plan"]["recording_lines"]==["untouched"]


@pytest.mark.parametrize("labels",[["i"],[],["i","i"],["i","ir",3],"i,ir"])
def test_invalid_label_requests_rejected(running_server, labels):
    base,_=running_server
    status,_,_=send(base,"/api/recording-plan/assets/preview",
                     {"archive":"/tmp/old.zip","labels":labels})
    assert status==400


def test_asset_preview_rejects_foreign_origin(running_server):
    base,_=running_server
    status,_,_=send(base,"/api/recording-plan/assets/preview",
        {"archive":"/tmp/old.zip","labels":["i","ir","i0"]},
        origin="https://malicious.invalid")
    assert status==403


def test_csv_defuses_spreadsheet_formula_injection():
    data=sample_ledger()
    data["original_alias_routes"][0]["original_aliases"]=["=IMPORTXML('https://bad.example')"]
    data["original_alias_routes"][0]["observed_source_options"][0]["observed_original_aliases"]=["@SUM(1,1)"]
    text=asset_routes_csv(data)
    values=list(csv.DictReader(io.StringIO(text)))
    assert any(row["original_aliases"].startswith("'@SUM") for row in values)
    assert "=IMPORTXML" not in text


def test_invalid_or_wrong_schema_csv_rejected():
    with pytest.raises(ValueError,match="expected validated"):
        asset_routes_csv({"schema":"unexpected"})


def test_gui_is_text_only_and_uses_bounded_display():
    assert "/api/recording-plan/assets/preview" in HTML
    assert "/api/recording-plan/assets/download/csv" in HTML
    assert "/api/recording-plan/assets/download/json" in HTML
    assert "list.slice(0,80)" in HTML
    assert "textContent" in HTML
    assert "innerHTML=" not in HTML
    assert "invalidateAssets" in HTML
    assert "历史归档" in HTML


def test_ledger_summary_cannot_be_confused_with_live_voicebank_scan():
    info=asset_ledger_summary(sample_ledger())
    assert info["source_scope"]=="existing_OTO_relative_filtered_source_index_only"
    assert info["historic_index_not_live_voicebank_validation"] is True
    assert info["no_new_recording_or_alias_changes"] is True
