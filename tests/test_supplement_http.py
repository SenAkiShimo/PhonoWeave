from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from phonoweave import gui

WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 40


@pytest.fixture
def server(tmp_path):
    previous = (gui._STATE.supplement_root, gui._STATE.supplement_session_dir, gui._STATE.supplement_defaults)
    gui._STATE.supplement_root = tmp_path
    gui._STATE.supplement_session_dir = None
    gui._STATE.supplement_defaults = {"lines": "ba_nuo\nzu_zuo\n", "layers": ["G4"], "bpm": 90}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), gui._Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    gui._STATE.supplement_root, gui._STATE.supplement_session_dir, gui._STATE.supplement_defaults = previous


def _post(url, body, content_type="application/json"):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": content_type})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def test_record_and_fetch_a_take(server) -> None:
    defaults = json.loads(urllib.request.urlopen(server + "/api/supplement/defaults").read())
    assert defaults["layers"] == ["G4"]

    session = _post(server + "/api/supplement/session", {"lines": "ba_nuo\nzu_zuo", "layers": "G4", "bpm": 90})
    assert session["completed"] == {"G4": []}

    saved = _post(
        server + "/api/supplement/recording?layer=G4&index=1&sample_rate=48000&beats=120,700",
        WAV,
        "audio/wav",
    )
    assert saved["saved"].endswith("02_zu_zuo.wav")
    fetched = urllib.request.urlopen(server + "/api/supplement/recording?layer=G4&index=1").read()
    assert fetched == WAV

    reopened = _post(server + "/api/supplement/open", {"path": session["directory"]})
    assert reopened["completed"] == {"G4": [1]}


def test_a_bad_take_is_refused(server) -> None:
    _post(server + "/api/supplement/session", {"lines": "ba_nuo", "layers": "G4", "bpm": 90})

    with pytest.raises(urllib.error.HTTPError) as error:
        _post(server + "/api/supplement/recording?layer=G4&index=0&sample_rate=48000&beats=1,2", b"nope", "audio/wav")

    assert error.value.code == 400


def test_page_is_served(server) -> None:
    page = urllib.request.urlopen(server + "/supplement").read().decode()

    assert "Supplement recording" in page
