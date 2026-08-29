from __future__ import annotations

import argparse
import json
import platform
import subprocess
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .calibration_io import (
    create_calibration_session,
    load_calibration_session,
    prompt_from_payload,
    protocol_payload,
    recording_path_for_session,
    save_calibration_recording_to_session,
)
from .calibration_page import HTML as CALIBRATION_HTML
from .calibration_protocol import live_calibration_protocol
from .gui_model import GuiAnalysisSnapshot, analyze_for_gui
from .gui_page_remake import HTML
from .profile import profile_yaml
from .synthesis_inventory import synthesis_inventory_yaml


@dataclass
class _GuiState:
    snapshot: GuiAnalysisSnapshot | None = None
    calibration_root: Path = Path.home() / "Downloads" / "PhonoWeaveCalibration"
    calibration_session_dir: Path | None = None


_STATE = _GuiState()
_STATE_LOCK = threading.Lock()


def _main_page_html() -> str:
    marker = '<button class="toolbtn primary" id="analyze"></button>'
    calibration = (
        marker
        + '<a class="toolbtn" href="/calibration" '
        + 'title="Record a live speaker calibration">'
        + 'Live Calibration / 现场校准</a>'
    )
    return HTML.replace(marker, calibration, 1)


def _calibration_page_html() -> str:
    marker = '<button class="toolbtn primary" id="start"></button>'
    open_button = (
        marker
        + '<button class="toolbtn" id="open-session">'
        + '打开本地会话 / Open Session</button>'
    )
    page = CALIBRATION_HTML.replace(marker, open_button, 1)
    bridge = r"""
<script>
async function loadSavedTake(i){
  if(!session||!done.has(i)){
    if(!takes.has(i)){
      document.getElementById('audio').removeAttribute('src');
      if(typeof clearCanvas==='function') clearCanvas(document.getElementById('take'));
      document.getElementById('take-meta').textContent='—';
    }
    return;
  }
  try{
    const r=await fetch(`/api/calibration/recording?index=${i}`);
    if(!r.ok)return;
    const blob=await r.blob();
    const arr=await blob.arrayBuffer();
    const ac=new (window.AudioContext||window.webkitAudioContext)();
    const decoded=await ac.decodeAudioData(arr.slice(0));
    const samples=new Float32Array(decoded.getChannelData(0));
    const sampleRate=decoded.sampleRate;
    await ac.close();
    const old=takes.get(i);if(old?.url)URL.revokeObjectURL(old.url);
    const url=URL.createObjectURL(blob);
    takes.set(i,{samples,url,path:'local session'});
    if(i===index){
      drawTake(samples);
      document.getElementById('audio').src=url;
      const p=currentPrompt();
      document.getElementById('take-meta').textContent=`${p.syllable} · ${p.context_family} · ${(samples.length/sampleRate).toFixed(2)} s`;
      render();
    }
  }catch(e){setStatus(String(e),true)}
}
async function openLocalSession(){
  setStatus(lang==='zh'?'正在打开本地会话…':'Opening local session…');
  try{
    const r=await fetch('/api/calibration/open',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lang})});
    const d=await r.json();
    if(!r.ok)throw Error(d.error||'open failed');
    if(!d.session_id){setStatus(lang==='zh'?'已取消。':'Canceled.');return}
    protocol=d.protocol;session=d.session_id;index=0;done=new Set(d.completed_indices||[]);
    for(const take of takes.values())if(take?.url)URL.revokeObjectURL(take.url);
    takes.clear();
    document.getElementById('session').textContent=`${session} · ${d.session_dir}`;
    render();
    await loadSavedTake(index);
    setStatus(lang==='zh'?`已打开 ${done.size}/${protocol.prompts.length}`:`Opened ${done.size}/${protocol.prompts.length}`);
  }catch(e){setStatus(e.message,true)}
}
document.getElementById('open-session').onclick=openLocalSession;
document.getElementById('list').addEventListener('click',()=>setTimeout(()=>loadSavedTake(index),0));
document.getElementById('prev').addEventListener('click',()=>setTimeout(()=>loadSavedTake(index),0));
document.getElementById('next').addEventListener('click',()=>setTimeout(()=>loadSavedTake(index),0));
</script>
"""
    return page.replace("</body>", bridge + "</body>", 1)


def _snapshot_payload(snapshot: GuiAnalysisSnapshot) -> dict[str, Any]:
    return {
        "voicebank": str(snapshot.root),
        "summary": {
            "onsets": len(snapshot.rows),
            "synthesis_units": len(snapshot.synthesis_inventory.units),
            "analyzed": snapshot.analyzed_count,
            "experimental": snapshot.experimental_count,
            "unsupported": snapshot.unsupported_count,
        },
        "rows": [
            {
                "base_unit": row.base_unit,
                "class_name": row.class_name,
                "acoustic_evidence": row.acoustic_evidence,
                "synthesis_evidence": row.synthesis_evidence,
                "decision": row.decision,
                "confidence": row.confidence,
                "groups": list(row.groups),
                "contexts": list(row.contexts),
                "notes": list(row.notes),
            }
            for row in snapshot.rows
        ],
    }


def _pick_folder(prompt: str) -> str:
    if platform.system() != "Darwin":
        raise RuntimeError("Folder picker is currently available on macOS only.")
    script = (
        f'set chosenFolder to choose folder with prompt "{prompt}"\n'
        'return POSIX path of chosenFolder'
    )
    completed = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip()
        if "User canceled" in message or "-128" in message:
            return ""
        raise RuntimeError(message or "Could not open the folder picker.")
    return completed.stdout.strip().rstrip("/")


def _pick_voicebank(language: str = "en") -> str:
    prompt = "选择 OpenUtau 声库" if language == "zh" else "Choose OpenUtau voicebank"
    return _pick_folder(prompt)


def _pick_calibration_session(language: str = "en") -> str:
    prompt = (
        "选择 PhonoWeave 校准会话"
        if language == "zh"
        else "Choose PhonoWeave calibration session"
    )
    return _pick_folder(prompt)


class _Handler(BaseHTTPRequestHandler):
    server_version = "PhonoWeaveGuiRemake/0.2"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("request body must be a JSON object")
        return parsed

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send_bytes(
                _main_page_html().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/calibration":
            self._send_bytes(
                _calibration_page_html().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/calibration/protocol":
            self._json(protocol_payload(live_calibration_protocol()))
            return
        if path == "/api/calibration/recording":
            try:
                query = parse_qs(parsed.query)
                index = int(query.get("index", ["-1"])[0])
                with _STATE_LOCK:
                    session_dir = _STATE.calibration_session_dir
                if session_dir is None:
                    raise ValueError("no active calibration session")
                wav = recording_path_for_session(session_dir, index)
                self._send_bytes(wav.read_bytes(), "audio/wav")
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        if path == "/api/export/profile":
            with _STATE_LOCK:
                snapshot = _STATE.snapshot
            if snapshot is None:
                self._json(
                    {"error": "No completed analysis is available."},
                    HTTPStatus.CONFLICT,
                )
                return
            self._send_bytes(
                profile_yaml(snapshot.profile).encode("utf-8"),
                "application/yaml; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="speaker_profile.yaml"'},
            )
            return
        if path == "/api/export/inventory":
            with _STATE_LOCK:
                snapshot = _STATE.snapshot
            if snapshot is None:
                self._json(
                    {"error": "No completed analysis is available."},
                    HTTPStatus.CONFLICT,
                )
                return
            self._send_bytes(
                synthesis_inventory_yaml(snapshot.synthesis_inventory).encode("utf-8"),
                "application/yaml; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="synthesis_inventory.yaml"'},
            )
            return
        self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/pick-folder":
            try:
                payload = self._read_json()
                language = "zh" if payload.get("lang") == "zh" else "en"
                selected = _pick_voicebank(language)
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._json({"path": selected})
            return
        if path == "/api/analyze":
            try:
                payload = self._read_json()
                raw_path = str(payload.get("path", "")).strip()
                if not raw_path:
                    raise ValueError("Choose a voicebank first.")
                root = Path(raw_path).expanduser().resolve()
                if not root.is_dir():
                    raise ValueError("The selected voicebank folder does not exist.")
                snapshot = analyze_for_gui(root)
                with _STATE_LOCK:
                    _STATE.snapshot = snapshot
                self._json(_snapshot_payload(snapshot))
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if path == "/api/calibration/start":
            try:
                protocol = live_calibration_protocol()
                with _STATE_LOCK:
                    _, session_dir = create_calibration_session(
                        protocol,
                        _STATE.calibration_root,
                    )
                    _STATE.calibration_session_dir = session_dir
                self._json(
                    {
                        "session_id": session_dir.name,
                        "session_dir": str(session_dir),
                        "protocol": protocol_payload(protocol),
                        "completed_indices": [],
                    }
                )
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if path == "/api/calibration/open":
            try:
                payload = self._read_json()
                raw_path = str(payload.get("path", "")).strip()
                language = "zh" if payload.get("lang") == "zh" else "en"
                if not raw_path:
                    raw_path = _pick_calibration_session(language)
                if not raw_path:
                    self._json({"session_id": ""})
                    return
                session_dir = Path(raw_path).expanduser().resolve()
                loaded = load_calibration_session(session_dir)
                with _STATE_LOCK:
                    _STATE.calibration_session_dir = session_dir
                self._json(loaded)
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if path == "/api/calibration/recording":
            try:
                query = parse_qs(parsed.query)
                index = int(query.get("index", ["-1"])[0])
                sample_rate = int(query.get("sample_rate", ["0"])[0])
                if sample_rate <= 0:
                    raise ValueError("invalid sample rate")
                with _STATE_LOCK:
                    session_dir = _STATE.calibration_session_dir
                if session_dir is None:
                    raise ValueError("no active calibration session")
                loaded = load_calibration_session(session_dir)
                protocol = loaded["protocol"]
                if not isinstance(protocol, dict):
                    raise ValueError("calibration protocol is invalid")
                prompts = protocol.get("prompts")
                if not isinstance(prompts, list) or index < 0 or index >= len(prompts):
                    raise ValueError("invalid calibration prompt index")
                prompt_payload = prompts[index]
                if not isinstance(prompt_payload, dict):
                    raise ValueError("calibration prompt is invalid")
                prompt = prompt_from_payload(prompt_payload)
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 64 * 1024 * 1024:
                    raise ValueError("invalid recording size")
                wav_bytes = self.rfile.read(length)
                output = save_calibration_recording_to_session(
                    session_dir,
                    index,
                    prompt,
                    wav_bytes,
                    sample_rate,
                )
                self._json({"saved": str(output), "index": index})
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave-gui")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"
    print(f"PhonoWeave GUI remake: {url}")
    print(f"Live calibration: {url}calibration")
    print("Press Ctrl-C to stop.")
    if not args.no_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
