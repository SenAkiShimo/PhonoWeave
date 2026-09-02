from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import gui
from .calibration_anchor_evaluation import OUTPUT_NAME, evaluate_manual_anchors


def _evaluation_page_html() -> str:
    page = gui._calibration_page_html()
    toolbar_marker = '<button class="toolbtn" id="open-session">打开本地会话 / Open Session</button>'
    toolbar = (
        toolbar_marker
        + '<button class="toolbtn" id="anchor-eval">运行 Anchor 评估 / Run Anchor Evaluation</button>'
    )
    page = page.replace(toolbar_marker, toolbar, 1)

    panel = r"""
<div id="anchor-eval-panel" style="display:none;position:fixed;left:248px;right:342px;bottom:25px;max-height:46vh;overflow:auto;background:#f7f7f4;border:1px solid #8f918c;box-shadow:0 -2px 8px rgba(0,0,0,.12);z-index:20;padding:10px 12px;font:12px/1.4 -apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
    <strong>Manual Anchor Evaluation v0.4</strong>
    <span id="anchor-eval-session" style="font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#666"></span>
    <span style="flex:1"></span>
    <button id="anchor-eval-close" class="toolbtn">×</button>
  </div>
  <div style="padding:7px 8px;background:#fff5cf;border:1px solid #c9b56c;margin-bottom:9px">Development data only — this 46-recording session was used during detector development and is not independent confirmatory validation.</div>
  <div id="anchor-eval-summary"></div>
  <div id="anchor-eval-classes" style="margin-top:10px"></div>
  <div id="anchor-eval-worst" style="margin-top:10px"></div>
</div>
"""
    page = page.replace('<div class="statusbar">', panel + '<div class="statusbar">', 1)

    bridge = r"""
<script>
function fmtMs(v){return v===null||v===undefined?'—':Number(v).toFixed(1)+' ms'}
function metricCards(title,m){return `<div style="margin-bottom:8px"><div style="font-weight:700;margin-bottom:4px">${title}</div><div style="display:grid;grid-template-columns:repeat(5,minmax(90px,1fr));gap:6px"><div style="background:white;border:1px solid #ccc;padding:6px"><div style="font-size:10px;color:#666">N</div><b>${m.n}</b></div><div style="background:white;border:1px solid #ccc;padding:6px"><div style="font-size:10px;color:#666">MAE</div><b>${fmtMs(m.mae_ms)}</b></div><div style="background:white;border:1px solid #ccc;padding:6px"><div style="font-size:10px;color:#666">Median |error|</div><b>${fmtMs(m.median_absolute_error_ms)}</b></div><div style="background:white;border:1px solid #ccc;padding:6px"><div style="font-size:10px;color:#666">P90 |error|</div><b>${fmtMs(m.p90_absolute_error_ms)}</b></div><div style="background:white;border:1px solid #ccc;padding:6px"><div style="font-size:10px;color:#666">Median signed</div><b>${fmtMs(m.median_signed_error_ms)}</b></div></div></div>`}
function renderAnchorEval(d){
  const panel=document.getElementById('anchor-eval-panel');panel.style.display='block';
  document.getElementById('anchor-eval-session').textContent=d.session_id||'';
  const a=d.absolute_anchor.primary_metrics,r=d.relative_repeat_alignment.primary_metrics,manual=d.manual_labels;
  document.getElementById('anchor-eval-summary').innerHTML=`<div style="margin-bottom:8px">Manual labels: <b>${manual.ok}</b> primary / ${manual.total} total / ${manual.uncertain} uncertain</div>${metricCards('Absolute paired anchor vs manual',a)}${metricCards('Relative repeat alignment vs manual repeat lag',r)}`;
  const cls=d.absolute_anchor.by_class||{};
  let rows=Object.entries(cls).map(([name,m])=>`<tr><td>${name}</td><td>${m.n}</td><td>${fmtMs(m.mae_ms)}</td><td>${fmtMs(m.median_absolute_error_ms)}</td><td>${fmtMs(m.median_signed_error_ms)}</td></tr>`).join('');
  document.getElementById('anchor-eval-classes').innerHTML=`<div style="font-weight:700;margin-bottom:4px">Absolute error by onset class</div><table style="width:100%;border-collapse:collapse;background:white"><thead><tr><th style="text-align:left">Class</th><th>N</th><th>MAE</th><th>Median |error|</th><th>Median signed</th></tr></thead><tbody>${rows}</tbody></table>`;
  rows=(d.absolute_anchor.worst_cases||[]).map(x=>`<tr><td>${x.syllable}</td><td>${x.occurrence}/2</td><td>${x.base_unit}</td><td>${x.context_family}</td><td>${fmtMs(x.manual_ms_after_cue)}</td><td>${fmtMs(x.auto_ms_after_cue)}</td><td>${fmtMs(x.error_ms)}</td><td>${x.diagnosis}</td></tr>`).join('');
  document.getElementById('anchor-eval-worst').innerHTML=`<div style="font-weight:700;margin-bottom:4px">Worst absolute-anchor cases</div><table style="width:100%;border-collapse:collapse;background:white"><thead><tr><th style="text-align:left">Syllable</th><th>Occ.</th><th>Onset</th><th>Context</th><th>Manual</th><th>Auto</th><th>Error</th><th>Diagnosis</th></tr></thead><tbody>${rows}</tbody></table>`;
}
async function loadAnchorEvalExisting(){
  try{const r=await fetch('/api/calibration/evaluation',{cache:'no-store'});if(r.status===404)return;const d=await r.json();if(r.ok)renderAnchorEval(d)}catch(e){}
}
async function runAnchorEval(){
  setStatus(lang==='zh'?'正在运行人工 Anchor 评估…':'Running manual anchor evaluation…');
  const b=document.getElementById('anchor-eval');b.disabled=true;
  try{const r=await fetch('/api/calibration/evaluate',{method:'POST'}),d=await r.json();if(!r.ok)throw Error(d.error||'evaluation failed');renderAnchorEval(d);setStatus(lang==='zh'?'Anchor 评估完成。':'Anchor evaluation complete.')}catch(e){setStatus(e.message,true)}finally{b.disabled=false}
}
document.getElementById('anchor-eval').onclick=runAnchorEval;
document.getElementById('anchor-eval-close').onclick=()=>document.getElementById('anchor-eval-panel').style.display='none';
const oldOpenLocalSession=openLocalSession;
openLocalSession=async function(){await oldOpenLocalSession();if(session)await loadAnchorEvalExisting()};
if(session)loadAnchorEvalExisting();
</script>
"""
    return page.replace("</body>", bridge + "</body>", 1)


class _EvaluationHandler(gui._Handler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/calibration":
            self._send_bytes(_evaluation_page_html().encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/calibration/evaluation":
            with gui._STATE_LOCK:
                session_dir = gui._STATE.calibration_session_dir
            if session_dir is None:
                self._json({"error": "no active calibration session"}, HTTPStatus.CONFLICT)
                return
            path = session_dir / "analysis" / OUTPUT_NAME
            if not path.is_file():
                self._json({"error": "evaluation not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("evaluation file is invalid")
                self._json(payload)
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/calibration/evaluate":
            with gui._STATE_LOCK:
                session_dir = gui._STATE.calibration_session_dir
            if session_dir is None:
                self._json({"error": "no active calibration session"}, HTTPStatus.CONFLICT)
                return
            try:
                self._json(evaluate_manual_anchors(session_dir))
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        super().do_POST()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave-calibration-eval-gui")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--calibration-session", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.calibration_session is not None:
        session_dir = args.calibration_session.expanduser().resolve()
        load = gui.load_calibration_session(session_dir)
        if not isinstance(load, dict):
            raise SystemExit("invalid calibration session")
        with gui._STATE_LOCK:
            gui._STATE.calibration_session_dir = session_dir

    server = ThreadingHTTPServer(("127.0.0.1", args.port), _EvaluationHandler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"
    print(f"PhonoWeave GUI remake: {url}", flush=True)
    print(f"Live calibration: {url}calibration", flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    if not args.no_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url + "calibration")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
