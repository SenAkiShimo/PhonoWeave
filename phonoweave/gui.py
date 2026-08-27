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
from urllib.parse import urlparse

from .gui_model import GuiAnalysisSnapshot, analyze_for_gui
from .profile import profile_yaml
from .synthesis_inventory import synthesis_inventory_yaml


@dataclass
class _GuiState:
    snapshot: GuiAnalysisSnapshot | None = None


_STATE = _GuiState()
_STATE_LOCK = threading.Lock()


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


def _pick_folder() -> str:
    if platform.system() != "Darwin":
        raise RuntimeError("Folder picker is currently available on macOS only.")
    script = (
        'set chosenFolder to choose folder with prompt "Choose OpenUtau voicebank"\n'
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


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PhonoWeave</title>
<style>
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --line: #e3e6eb;
  --text: #17191d;
  --muted: #6f7680;
  --accent: #1e63d6;
  --accent-soft: #eaf1ff;
  --good: #247a4b;
  --good-soft: #eaf6ef;
  --warn: #9b6900;
  --warn-soft: #fff5dc;
  --quiet: #59616c;
  --quiet-soft: #eef0f3;
  --danger: #a33b3b;
  --shadow: 0 10px 30px rgba(20, 24, 32, .06);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button, input { font: inherit; }
.app { max-width: 1240px; margin: 0 auto; padding: 28px; }
.header { display: flex; justify-content: space-between; align-items: end; gap: 20px; margin-bottom: 22px; }
.brand h1 { margin: 0; font-size: 28px; letter-spacing: -.6px; }
.brand p { margin: 4px 0 0; color: var(--muted); }
.status { color: var(--muted); font-size: 13px; text-align: right; }
.toolbar {
  display: grid;
  grid-template-columns: auto 1fr auto auto;
  gap: 10px;
  align-items: center;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px;
  box-shadow: var(--shadow);
}
.toolbar label { color: var(--muted); font-weight: 600; }
input {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 9px;
  padding: 10px 12px;
  background: #fff;
  outline: none;
}
input:focus { border-color: #9bbcf1; box-shadow: 0 0 0 3px var(--accent-soft); }
button, .download {
  border: 1px solid var(--line);
  background: #fff;
  border-radius: 9px;
  padding: 9px 13px;
  cursor: pointer;
  color: var(--text);
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}
button:hover, .download:hover { background: #f8f9fb; }
button.primary { background: var(--accent); color: white; border-color: var(--accent); font-weight: 650; }
button.primary:hover { filter: brightness(.97); }
button:disabled, .download.disabled { opacity: .45; cursor: default; pointer-events: none; }
.cards { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 14px 0; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 13px 15px; }
.card .value { font-size: 22px; font-weight: 700; letter-spacing: -.3px; }
.card .label { color: var(--muted); font-size: 12px; margin-top: 2px; }
.workspace { display: grid; grid-template-columns: minmax(560px, 1.25fr) minmax(350px, .75fr); gap: 14px; min-height: 530px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; box-shadow: var(--shadow); overflow: hidden; }
.panel-title { padding: 14px 16px; border-bottom: 1px solid var(--line); font-weight: 700; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; color: var(--muted); font-size: 12px; font-weight: 650; padding: 10px 13px; border-bottom: 1px solid var(--line); background: #fafbfc; }
td { padding: 10px 13px; border-bottom: 1px solid #eff1f4; }
tbody tr { cursor: pointer; }
tbody tr:hover { background: #f8faff; }
tbody tr.selected { background: var(--accent-soft); }
.onset { font-size: 15px; font-weight: 750; }
.badge { display: inline-block; padding: 3px 8px; border-radius: 99px; font-size: 12px; font-weight: 650; white-space: nowrap; }
.split_recommended { color: var(--good); background: var(--good-soft); }
.unresolved { color: var(--warn); background: var(--warn-soft); }
.merge_supported { color: var(--quiet); background: var(--quiet-soft); }
.three_realizations_provisional { color: var(--good); background: var(--good-soft); }
.details { padding: 18px; overflow: auto; height: 100%; }
.details h2 { margin: 0; font-size: 30px; }
.details .class { color: var(--muted); margin-top: 1px; }
.detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 18px 0; }
.detail-box { border: 1px solid var(--line); border-radius: 10px; padding: 11px 12px; }
.detail-box .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .4px; }
.detail-box .v { margin-top: 3px; font-weight: 650; overflow-wrap: anywhere; }
.section { margin-top: 20px; }
.section h3 { font-size: 13px; margin: 0 0 8px; }
.group { border: 1px solid var(--line); border-radius: 9px; padding: 9px 10px; margin: 7px 0; }
.group strong { display: block; overflow-wrap: anywhere; }
.group span { color: var(--muted); font-size: 12px; }
.notes { margin: 0; padding-left: 19px; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
.notes li { margin: 5px 0; overflow-wrap: anywhere; }
.empty { padding: 48px 24px; color: var(--muted); text-align: center; }
.footer { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-top: 14px; }
.export { display: flex; gap: 8px; }
.spinner { display: inline-block; width: 13px; height: 13px; border: 2px solid rgba(255,255,255,.5); border-top-color: #fff; border-radius: 50%; animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.error { color: var(--danger); }
@media (max-width: 900px) {
  .app { padding: 16px; }
  .toolbar { grid-template-columns: 1fr auto; }
  .toolbar label { grid-column: 1 / -1; }
  .toolbar input { grid-column: 1 / -1; }
  .cards { grid-template-columns: repeat(2, 1fr); }
  .workspace { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <div class="brand">
      <h1>PhonoWeave</h1>
      <p>Speaker realization inventory</p>
    </div>
    <div class="status" id="status">Choose a voicebank to begin.</div>
  </div>

  <div class="toolbar">
    <label for="voicebank">Voicebank</label>
    <input id="voicebank" placeholder="/Users/.../OpenUtau/Singers/Voicebank">
    <button id="browse">Browse</button>
    <button class="primary" id="analyze">Analyze</button>
  </div>

  <div class="cards">
    <div class="card"><div class="value" id="onsets">—</div><div class="label">Onsets</div></div>
    <div class="card"><div class="value" id="units">—</div><div class="label">Synthesis units</div></div>
    <div class="card"><div class="value" id="analyzed">—</div><div class="label">Analyzed</div></div>
    <div class="card"><div class="value" id="experimental">—</div><div class="label">Experimental</div></div>
    <div class="card"><div class="value" id="unsupported">—</div><div class="label">Unsupported</div></div>
  </div>

  <div class="workspace">
    <div class="panel">
      <div class="panel-title">Onset decisions</div>
      <div id="table-wrap" class="empty">Run an analysis to build the speaker realization inventory.</div>
    </div>
    <div class="panel">
      <div class="panel-title">Details</div>
      <div id="details" class="empty">Select an onset after analysis.</div>
    </div>
  </div>

  <div class="footer">
    <div id="voicebank-name" class="status"></div>
    <div class="export">
      <a id="profile" class="download disabled" href="/api/export/profile">Export Speaker Profile</a>
      <a id="inventory" class="download disabled" href="/api/export/inventory">Export Synthesis Inventory</a>
    </div>
  </div>
</div>
<script>
let current = null;
let selected = null;
const $ = id => document.getElementById(id);

function setStatus(text, error=false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function renderSummary(summary) {
  for (const key of ['onsets','synthesis_units','analyzed','experimental','unsupported']) {
    const id = key === 'synthesis_units' ? 'units' : key;
    $(id).textContent = summary[key];
  }
}

function renderTable(rows) {
  const html = [`<table><thead><tr><th>Onset</th><th>Class</th><th>Decision</th><th>Confidence</th></tr></thead><tbody>`];
  rows.forEach(row => {
    html.push(`<tr data-base="${escapeHtml(row.base_unit)}">
      <td class="onset">${escapeHtml(row.base_unit)}</td>
      <td>${escapeHtml(row.class_name)}</td>
      <td><span class="badge ${escapeHtml(row.decision)}">${escapeHtml(row.decision)}</span></td>
      <td>${escapeHtml(row.confidence)}</td>
    </tr>`);
  });
  html.push('</tbody></table>');
  $('table-wrap').className = '';
  $('table-wrap').innerHTML = html.join('');
  document.querySelectorAll('tbody tr').forEach(tr => tr.addEventListener('click', () => selectRow(tr.dataset.base)));
}

function selectRow(base) {
  if (!current) return;
  const row = current.rows.find(item => item.base_unit === base);
  if (!row) return;
  selected = base;
  document.querySelectorAll('tbody tr').forEach(tr => tr.classList.toggle('selected', tr.dataset.base === base));
  const groups = row.groups.map((group, i) => `<div class="group"><strong>${escapeHtml(group)}</strong><span>${escapeHtml(row.contexts[i] || '')}</span></div>`).join('');
  const notes = row.notes.map(note => `<li>${escapeHtml(note)}</li>`).join('');
  $('details').className = 'details';
  $('details').innerHTML = `
    <h2>${escapeHtml(row.base_unit)}</h2>
    <div class="class">${escapeHtml(row.class_name)}</div>
    <div class="detail-grid">
      <div class="detail-box"><div class="k">Decision</div><div class="v">${escapeHtml(row.decision)}</div></div>
      <div class="detail-box"><div class="k">Confidence</div><div class="v">${escapeHtml(row.confidence)}</div></div>
      <div class="detail-box"><div class="k">Acoustic evidence</div><div class="v">${escapeHtml(row.acoustic_evidence)}</div></div>
      <div class="detail-box"><div class="k">Synthesis evidence</div><div class="v">${escapeHtml(row.synthesis_evidence)}</div></div>
    </div>
    <div class="section"><h3>Realization groups</h3>${groups}</div>
    <div class="section"><h3>Analysis notes</h3><ul class="notes">${notes}</ul></div>`;
}

async function browse() {
  setStatus('Opening folder picker…');
  try {
    const response = await fetch('/api/pick-folder', {method: 'POST'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Folder picker failed.');
    if (data.path) $('voicebank').value = data.path;
    setStatus(data.path ? 'Voicebank selected.' : 'Folder selection canceled.');
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function analyze() {
  const path = $('voicebank').value.trim();
  if (!path) { setStatus('Choose a voicebank first.', true); return; }
  const button = $('analyze');
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span> Analyzing';
  $('browse').disabled = true;
  $('profile').classList.add('disabled');
  $('inventory').classList.add('disabled');
  setStatus('Analyzing voicebank…');
  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({path})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis failed.');
    current = data;
    renderSummary(data.summary);
    renderTable(data.rows);
    $('voicebank-name').textContent = data.voicebank;
    $('profile').classList.remove('disabled');
    $('inventory').classList.remove('disabled');
    if (data.rows.length) selectRow(data.rows[0].base_unit);
    setStatus(`${data.summary.onsets} onsets · ${data.summary.synthesis_units} synthesis units`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = 'Analyze';
    $('browse').disabled = false;
  }
}

$('browse').addEventListener('click', browse);
$('analyze').addEventListener('click', analyze);
$('voicebank').addEventListener('keydown', event => { if (event.key === 'Enter') analyze(); });
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "PhonoWeaveGui/0.1"

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
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/export/profile":
            with _STATE_LOCK:
                snapshot = _STATE.snapshot
            if snapshot is None:
                self._json({"error": "No completed analysis is available."}, HTTPStatus.CONFLICT)
                return
            body = profile_yaml(snapshot.profile).encode("utf-8")
            self._send_bytes(
                body,
                "application/yaml; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="speaker_profile.yaml"'},
            )
            return
        if path == "/api/export/inventory":
            with _STATE_LOCK:
                snapshot = _STATE.snapshot
            if snapshot is None:
                self._json({"error": "No completed analysis is available."}, HTTPStatus.CONFLICT)
                return
            body = synthesis_inventory_yaml(snapshot.synthesis_inventory).encode("utf-8")
            self._send_bytes(
                body,
                "application/yaml; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="synthesis_inventory.yaml"'},
            )
            return
        self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/pick-folder":
            try:
                selected = _pick_folder()
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
    print(f"PhonoWeave GUI: {url}")
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
