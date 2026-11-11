"""Integration smoke tests for recording-plan GUI endpoints."""
from __future__ import annotations
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pytest
from phonoweave import gui
from phonoweave.recording_plan_page_v19 import HTML

def send(base, path, body=None, origin=None):
    headers = {}
    if body is not None:
        headers['Content-Type'] = 'application/json'
    if origin:
        headers['Origin'] = origin
    req = Request(base + path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
    try:
        with urlopen(req, timeout=8) as response:
            return response.status, response.read(), response.headers
    except HTTPError as exc:
        return exc.code, exc.read(), exc.headers

@pytest.fixture
def local_server(monkeypatch):
    with gui._STATE_LOCK:
        gui._STATE.recording_report = None
    def planner(path, *, max_lines, existing_reclist):
        assert max_lines == 3
        return {'voicebank': {'root': str(path)},
                'coverage': {'recognized_syllable_observations': 2},
                'evidence': {'diagnostic_recording_gaps': 1},
                'recording_plan': {'recording_lines': ['a_fa_a'], 'selected_count': 1,
                                   'selected': [{'line': 'a_fa_a'}]},
                'warnings': ['not independently validated'],
                'restrictions': {'voicebank_unchanged': True}}
    monkeypatch.setattr(gui, 'build_recording_workflow', planner)
    server = ThreadingHTTPServer(('127.0.0.1', 0), gui._Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield 'http://127.0.0.1:%s' % server.server_address[1]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        with gui._STATE_LOCK:
            gui._STATE.recording_report = None

def test_page_and_original_home_link(local_server):
    status, body, _ = send(local_server, '/')
    assert status == 200 and b'/recording-plan' in body
    status, body, _ = send(local_server, '/recording-plan')
    assert status == 200 and b'/api/recording-plan/preview' in body

def test_download_refused_without_preview(local_server):
    for suffix in ('reclist','report'):
        status, _, _ = send(local_server, '/api/recording-plan/download/' + suffix)
        assert status == 409

def test_preview_and_both_downloads(local_server):
    status, body, _ = send(local_server, '/api/recording-plan/preview',
                            {'path':'/tmp/example','max_lines':3})
    assert status == 200
    report = json.loads(body)
    assert report['recording_plan']['recording_lines'] == ['a_fa_a']
    status, reclist, headers = send(local_server, '/api/recording-plan/download/reclist')
    assert status == 200 and reclist == b'a_fa_a\n'
    assert 'attachment' in headers['Content-Disposition']
    status, exported, headers = send(local_server, '/api/recording-plan/download/report')
    assert status == 200 and json.loads(exported) == report
    assert 'attachment' in headers['Content-Disposition']

def test_failed_preview_invalidates_downloads(local_server):
    send(local_server, '/api/recording-plan/preview', {'path':'/tmp/test','max_lines':3})
    status, _, _ = send(local_server, '/api/recording-plan/preview', {'path':'','max_lines':3})
    assert status == 400
    status, _, _ = send(local_server, '/api/recording-plan/download/report')
    assert status == 409

@pytest.mark.parametrize('budget', [True, -1, 10001, '3', 1.5])
def test_invalid_budget_rejected(local_server, budget):
    status, _, _ = send(local_server, '/api/recording-plan/preview',
                         {'path':'/tmp/test','max_lines':budget})
    assert status == 400

def test_foreign_origin_rejected(local_server):
    status, _, _ = send(local_server, '/api/recording-plan/preview',
                        {'path':'/tmp/test','max_lines':3}, origin='https://attacker.example')
    assert status == 403

def test_page_uses_text_only_dom_for_voicebank_data():
    assert 'textContent' in HTML
    assert 'innerHTML=' not in HTML
    assert 'eval(' not in HTML
    assert 'reset()' in HTML
    assert 'max_lines' in HTML
