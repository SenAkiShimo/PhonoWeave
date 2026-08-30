from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .calibration_io import load_calibration_session, recording_path_for_session
from .calibration_manual_labels import save_manual_label, setup_payload


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhonoWeave — Manual Anchor Development</title>
<style>
:root{--bg:#efefec;--panel:#fff;--line:#9b9d98;--ink:#181917;--muted:#62645f;--sel:#1d4f91;--warn:#8a5b17;--ok:#26633f;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}.app{height:100vh;display:grid;grid-template-rows:34px 1fr 26px}.bar,.status{background:#dedfdb;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 9px;gap:10px}.status{border-top:1px solid var(--line);border-bottom:0;font:11px var(--mono)}.brand{font-weight:700}.muted{color:var(--muted)}.workspace{min-height:0;display:grid;grid-template-columns:250px 1fr}.list{overflow:auto;border-right:1px solid var(--line);background:#e7e8e4}.item{padding:7px 9px;border-bottom:1px solid #cfd0cc;cursor:pointer;display:grid;grid-template-columns:34px 1fr 42px}.item.selected{background:var(--sel);color:#fff}.item .sub{font-size:10px;color:#696b66}.item.selected .sub{color:#d9e2ef}.item .state{text-align:right;font:10px var(--mono)}.main{min-width:0;overflow:auto;padding:16px}.head{display:flex;align-items:flex-start;gap:14px;margin-bottom:10px}.title{font:700 23px var(--mono)}.instruction{margin-top:5px;color:#444}.cards{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card{background:var(--panel);border:1px solid var(--line);padding:10px}.cardhead{display:flex;align-items:center;gap:8px;margin-bottom:8px;font:12px var(--mono)}.cardhead .spacer{flex:1}.wave{width:100%;height:220px;border:1px solid var(--line);background:#fafafa;cursor:crosshair;display:block}.readout{font:11px var(--mono);margin-top:7px;min-height:17px}.controls{display:flex;gap:6px;margin-top:7px}.controls button,.bar button{font:inherit;border:1px solid #7e807b;background:#fff;padding:4px 9px;cursor:pointer}.controls button.active{background:#d5c6aa}.legend{font:10px var(--mono);color:var(--muted);margin-top:10px}.status .spacer{flex:1}.ok{color:var(--ok)}.warn{color:var(--warn)}</style>
</head><body><div class="app">
<div class="bar"><span class="brand">PHONOWEAVE</span><span>MANUAL ANCHOR DEVELOPMENT</span><span class="muted" id="session"></span><span style="flex:1"></span><button id="zh">中文</button><button id="en">EN</button></div>
<div class="workspace"><div class="list" id="list"></div><main class="main"><div class="head"><div><div class="title" id="title">—</div><div class="instruction" id="instruction"></div></div></div><div class="cards"><section class="card"><div class="cardhead"><span id="occ1"></span><span class="spacer"></span><span id="state1"></span></div><canvas class="wave" id="wave1"></canvas><div class="readout" id="readout1">—</div><div class="controls"><button data-play="1" id="play1"></button><button data-uncertain="1" id="uncertain1"></button><button data-clear="1" id="clear1"></button></div></section><section class="card"><div class="cardhead"><span id="occ2"></span><span class="spacer"></span><span id="state2"></span></div><canvas class="wave" id="wave2"></canvas><div class="readout" id="readout2">—</div><div class="controls"><button data-play="2" id="play2"></button><button data-uncertain="2" id="uncertain2"></button><button data-clear="2" id="clear2"></button></div></section></div><div class="legend" id="legend"></div></main></div>
<div class="status"><span id="status"></span><span class="spacer"></span><span id="progress"></span></div></div>
<script>
const $=id=>document.getElementById(id);let lang='zh',data=null,index=0,audioCtx=null,buffers=new Map();
const T={zh:{occ:'第',occTail:'遍',play:'试听局部',uncertain:'标为不确定',clear:'清除',ready:'点击波形设置 anchor。',legend:'蓝线 = 实际 beep cue；红线 = 你点击的 anchor。页面不显示自动算法建议。'},en:{occ:'Occurrence ',occTail:'',play:'Play window',uncertain:'Mark uncertain',clear:'Clear',ready:'Click the waveform to set the anchor.',legend:'Blue line = detected beep cue; red line = your manual anchor. Automatic suggestions are intentionally hidden.'}};
function t(k){return T[lang][k]}
function current(){return data.prompts[index]}
function labelRow(){return data.labels[String(current().prompt_index)]||null}
function labelFor(o){const r=labelRow();return r?.occurrences?.[String(o)]||null}
function applyLang(){document.documentElement.lang=lang==='zh'?'zh-CN':'en';$('zh').disabled=lang==='zh';$('en').disabled=lang==='en';$('play1').textContent=$('play2').textContent=t('play');$('uncertain1').textContent=$('uncertain2').textContent=t('uncertain');$('clear1').textContent=$('clear2').textContent=t('clear');$('legend').textContent=t('legend');render()}
function stateText(o){const x=labelFor(o);if(!x)return '—';return x.status==='uncertain'?'UNCERTAIN':'OK'}
function doneCount(){let n=0;for(const p of data.prompts){const r=data.labels[String(p.prompt_index)];if(r?.occurrences?.['1']&&r?.occurrences?.['2'])n++}return n}
function render(){if(!data)return;let h='';data.prompts.forEach((p,i)=>{const r=data.labels[String(p.prompt_index)],done=!!(r?.occurrences?.['1']&&r?.occurrences?.['2']);h+=`<div class="item ${i===index?'selected':''}" data-i="${i}"><span>${String(i+1).padStart(2,'0')}</span><span><b>${p.base_unit}</b> · ${p.context_family}<div class="sub">${p.syllable} · ${p.class_name}</div></span><span class="state">${done?'2/2':r?.occurrences?Object.keys(r.occurrences).length+'/2':'0/2'}</span></div>`});$('list').innerHTML=h;document.querySelectorAll('.item').forEach(x=>x.onclick=()=>{index=Number(x.dataset.i);render();loadCurrent()});const p=current();$('title').textContent=`${p.base_unit} · ${p.context_family} · ${p.syllable}`;$('instruction').textContent=p.instruction[lang];$('occ1').textContent=`${t('occ')}1${t('occTail')}`;$('occ2').textContent=`${t('occ')}2${t('occTail')}`;$('state1').textContent=stateText(1);$('state2').textContent=stateText(2);$('progress').textContent=`${doneCount()}/${data.prompts.length}`;for(let o=1;o<=2;o++){const x=labelFor(o);$(`readout${o}`).textContent=x?`${x.anchor_ms_after_cue.toFixed(1)} ms after cue · ${x.status}`:'—';$(`uncertain${o}`).classList.toggle('active',x?.status==='uncertain')}draw(1);draw(2)}
async function getBuffer(){const p=current(),key=p.prompt_index;if(buffers.has(key))return buffers.get(key);const r=await fetch(`/audio?prompt_index=${p.prompt_index}`),arr=await r.arrayBuffer();if(!audioCtx)audioCtx=new (window.AudioContext||window.webkitAudioContext)();const b=await audioCtx.decodeAudioData(arr);buffers.set(key,b);return b}
function windowBounds(o,b){const p=current(),cue=p.cues_ms[o-1],s=data.window.start_ms_after_cue,e=data.window.end_ms_after_cue;return {cue,start:(cue+s)/1000,end:Math.min(b.duration,(cue+e)/1000),s,e}}
async function loadCurrent(){await getBuffer();render()}
function draw(o){const c=$(`wave${o}`),g=c.getContext('2d'),d=window.devicePixelRatio||1,w=Math.max(1,Math.floor(c.clientWidth*d)),h=Math.max(1,Math.floor(c.clientHeight*d));if(c.width!==w||c.height!==h){c.width=w;c.height=h}g.clearRect(0,0,w,h);const b=buffers.get(current().prompt_index);if(!b)return;const ch=b.getChannelData(0),bounds=windowBounds(o,b),a=Math.max(0,Math.floor(bounds.start*b.sampleRate)),z=Math.min(ch.length,Math.floor(bounds.end*b.sampleRate)),step=Math.max(1,Math.floor((z-a)/w));g.beginPath();for(let x=0;x<w;x++){let m=0;for(let j=0;j<step;j++){const k=a+x*step+j;if(k<z)m=Math.max(m,Math.abs(ch[k]))}g.moveTo(x,h/2-m*h*.44);g.lineTo(x,h/2+m*h*.44)}g.stroke();g.beginPath();g.moveTo(0,0);g.lineTo(0,h);g.strokeStyle='#1d4f91';g.stroke();const lab=labelFor(o);if(lab){const frac=(lab.anchor_ms_after_cue-bounds.s)/(bounds.e-bounds.s),x=Math.max(0,Math.min(w,frac*w));g.beginPath();g.moveTo(x,0);g.lineTo(x,h);g.strokeStyle='#a02b2b';g.lineWidth=2;g.stroke()}g.strokeStyle='#111';g.lineWidth=1}
async function play(o){const b=await getBuffer();if(!audioCtx)audioCtx=new (window.AudioContext||window.webkitAudioContext)();await audioCtx.resume();const bounds=windowBounds(o,b),src=audioCtx.createBufferSource();src.buffer=b;src.connect(audioCtx.destination);src.start(0,bounds.start,Math.max(.01,bounds.end-bounds.start))}
async function save(o,value,status){const p=current();const r=await fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt_index:p.prompt_index,occurrence:o,anchor_ms_after_cue:value,status})}),d=await r.json();if(!r.ok){$('status').textContent=d.error||'error';return}data.labels=d.labels||{};$('status').textContent=t('ready');render()}
for(let o=1;o<=2;o++){const c=$(`wave${o}`);c.onclick=async e=>{const r=c.getBoundingClientRect(),frac=(e.clientX-r.left)/r.width,s=data.window.start_ms_after_cue,end=data.window.end_ms_after_cue,value=s+frac*(end-s),old=labelFor(o),status=old?.status==='uncertain'?'uncertain':'ok';await save(o,value,status)};$(`play${o}`).onclick=()=>play(o);$(`uncertain${o}`).onclick=async()=>{const old=labelFor(o);if(!old){$('status').textContent=lang==='zh'?'请先点击一个 anchor。':'Click an anchor first.';return}await save(o,old.anchor_ms_after_cue,old.status==='uncertain'?'ok':'uncertain')};$(`clear${o}`).onclick=()=>save(o,null,'unset')}
$('zh').onclick=()=>{lang='zh';applyLang()};$('en').onclick=()=>{lang='en';applyLang()};
(async()=>{const r=await fetch('/setup'),d=await r.json();if(!r.ok){$('status').textContent=d.error||'error';return}data=d;$('session').textContent=d.session_id;applyLang();await loadCurrent();$('status').textContent=t('ready')})();
</script></body></html>"""


class _State:
    session_dir: Path | None = None


_STATE = _State()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if _STATE.session_dir is None:
            self._json({"error": "no active session"}, HTTPStatus.CONFLICT)
            return
        if parsed.path == "/setup":
            try:
                self._json(setup_payload(_STATE.session_dir))
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path == "/audio":
            try:
                q = parse_qs(parsed.query)
                prompt_index = int(q.get("prompt_index", ["-1"])[0])
                load_calibration_session(_STATE.session_dir)
                wav = recording_path_for_session(_STATE.session_dir, prompt_index)
                self._bytes(wav.read_bytes(), "audio/wav")
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/label":
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        if _STATE.session_dir is None:
            self._json({"error": "no active session"}, HTTPStatus.CONFLICT)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = save_manual_label(
                _STATE.session_dir,
                int(payload["prompt_index"]),
                int(payload["occurrence"]),
                None if payload.get("anchor_ms_after_cue") is None else float(payload["anchor_ms_after_cue"]),
                str(payload["status"]),
            )
            self._json({"labels": result.get("labels", {})})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave-manual-anchor")
    parser.add_argument("session", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    session_dir = args.session.expanduser().resolve()
    load_calibration_session(session_dir)
    setup_payload(session_dir)
    _STATE.session_dir = session_dir
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"
    print(f"Manual anchor development: {url}")
    print(f"Session: {session_dir}")
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
