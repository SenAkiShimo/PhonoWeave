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
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhonoWeave — Manual Anchor Development</title>
<style>
:root{--bg:#eceee9;--panel:#fff;--line:#979b94;--ink:#171816;--muted:#62665f;--blue:#245b97;--red:#ad3434;--gray:#dadcd7;--gray2:#eceeea;--warn:#8a5b17;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}.app{min-height:100vh;display:grid;grid-template-rows:38px 1fr 28px}.bar,.status{background:#dde0da;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 12px;gap:10px}.status{border-top:1px solid var(--line);border-bottom:0;font:11px var(--mono)}.brand{font-weight:800}.muted{color:var(--muted)}.spacer{flex:1}.main{max-width:1100px;width:100%;margin:auto;padding:24px}.step{font:12px var(--mono);color:var(--muted);margin-bottom:8px}.title{font-size:30px;font-weight:800}.subtitle{font:14px var(--mono);color:#444;margin:6px 0 16px}.instruction{background:#fff8e8;border:1px solid #bca879;padding:13px 15px;margin-bottom:14px;line-height:1.55}.instruction b{display:block;font-size:16px;margin-bottom:3px}.panel{background:var(--panel);border:1px solid var(--line);padding:14px}.slotinfo{display:flex;gap:14px;align-items:center;font:11px var(--mono);color:var(--muted);margin-bottom:8px}.wave{width:100%;height:330px;border:1px solid var(--line);background:#fbfbfa;display:block;cursor:crosshair}.readout{font:12px var(--mono);margin-top:8px;min-height:18px}.controls,.nav{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:12px}.controls button,.nav button,.bar button{font:inherit;border:1px solid #777b74;background:#fff;padding:7px 12px;cursor:pointer}.controls .primary{background:#234f86;color:#fff;border-color:#234f86}.controls .warn{background:#f6ead1}.controls button:disabled,.nav button:disabled{opacity:.45;cursor:default}.legend{margin-top:10px;color:var(--muted);font-size:12px}.warntext{color:var(--warn)}
</style></head><body><div class="app">
<div class="bar"><span class="brand">PHONOWEAVE</span><span>人工校准标注</span><span class="muted" id="session"></span><span class="spacer"></span><button id="zh">中文</button><button id="en">EN</button></div>
<main class="main"><div class="step" id="step">—</div><div class="title" id="title">—</div><div class="subtitle" id="subtitle"></div><div class="instruction"><b id="short"></b><div id="instruction"></div><div class="warntext" id="dont"></div></div><section class="panel"><div class="slotinfo"><span id="slot"></span><span id="mode"></span></div><canvas class="wave" id="wave" style="height:190px"></canvas><div class="slotinfo"><span>频谱 / Spectrogram · 0–8 kHz</span><span id="viewtime" style="margin-left:auto"></span></div><canvas class="wave" id="spectrum" style="height:220px"></canvas><div class="slotinfo"><label for="pan">时间轴 / Timeline</label><input id="pan" type="range" min="0" max="1000" value="0" step="1" style="flex:1"><label for="zoom">缩放 / Zoom</label><input id="zoom" type="range" min="0" max="100" value="45" step="1" style="width:150px"><button id="resetview" type="button">复位 / Reset</button></div><div class="readout" id="readout">—</div><div class="controls"><button id="play" class="primary"></button><button id="playWhole"></button><button id="playClick"></button><button id="uncertain" class="warn"></button><button id="clear"></button></div><div class="legend" id="legend"></div></section><div class="nav"><button id="prev"></button><span class="spacer"></span><button id="next"></button></div></main>
<div class="status"><span id="status"></span><span class="spacer"></span><span id="progress"></span></div></div>
<script>
const $=id=>document.getElementById(id);let lang='zh',data=null,stepIndex=0,audioCtx=null,buffers=new Map(),viewStart=0,viewWidth=1,specCache=new Map();
const T={zh:{play:'▶ 播放这一拍',playWhole:'▶ 播放整条录音',playClick:'▶ 从标记附近播放',uncertain:'我不确定这个位置',clear:'清除这个点',prev:'← 上一个',next:'下一个 →',ready:'请参考波形和频谱标记目标辅音起点。',dont:'无法确定辅音起点时，请使用“不确定”标记；偏离预计节拍的标记将提示复核。',legend:'深灰 = 当前 beep 附近，不要点；浅灰 = 下一拍 beep 附近；蓝线 = 当前 beep；紫线 = 下一拍 beep；红线 = 你的标记。'},en:{play:'▶ Play this beat',playWhole:'▶ Play full recording',playClick:'▶ Play around marker',uncertain:'I am uncertain',clear:'Clear this point',prev:'← Previous',next:'Next →',ready:'Use the waveform and spectrogram to mark onset.',dont:'If onset location is uncertain, mark it for review. Out-of-slot anchors are flagged.',legend:'Dark gray = current cue region; light gray = next cue region; blue = current cue; purple = next cue; red = your mark.'}};function t(k){return T[lang][k]}
function flat(){const a=[];for(const p of data.prompts){a.push({p,o:1});a.push({p,o:2})}return a}function cur(){return flat()[stepIndex]}function row(p){return data.labels[String(p.prompt_index)]||null}function lab(p,o){return row(p)?.occurrences?.[String(o)]||null}function doneCount(){let n=0;for(const x of flat())if(lab(x.p,x.o))n++;return n}
function applyLang(){document.documentElement.lang=lang==='zh'?'zh-CN':'en';$('zh').disabled=lang==='zh';$('en').disabled=lang==='en';for(const k of ['play','playWhole','playClick','uncertain','clear','prev','next'])$(k).textContent=t(k);$('legend').textContent=t('legend');render()}
async function getBuffer(){const p=cur().p,k=p.prompt_index;if(buffers.has(k))return buffers.get(k);const r=await fetch(`/audio?prompt_index=${k}`),arr=await r.arrayBuffer();if(!audioCtx)audioCtx=new(window.AudioContext||window.webkitAudioContext)();const b=await audioCtx.decodeAudioData(arr);buffers.set(k,b);return b}
function bounds(b){const x=cur(),p=x.p,o=x.o,d=data.display,cue=p.cues_ms[o-1],next=p.next_cues_ms[o-1],start=viewStart,end=Math.min(b.duration*1000,start+viewWidth);return{cue,next,start:start/1000,end:end/1000,displayStart:start-cue,displayEnd:end-cue,labelStart:-cue,labelEnd:b.duration*1000-cue,gray1a:d.cue_gray_start_ms,gray1b:d.cue_gray_end_ms,gray2a:(next-cue)+d.cue_gray_start_ms,gray2b:(next-cue)+d.cue_gray_end_ms}}
function setView(b,reset=false){const p=cur().p,o=cur().o,full=b.duration*1000,slot=Math.max(250,p.next_cues_ms[o-1]-p.prev_cues_ms[o-1]+180);if(reset){viewWidth=Math.min(full,Math.max(900,slot));viewStart=Math.max(0,Math.min(full-viewWidth,p.prev_cues_ms[o-1]-80));}else{viewWidth=Math.max(350,Math.min(viewWidth,full));viewStart=Math.max(0,Math.min(viewStart,full-viewWidth));}const maxPan=Math.max(0,full-viewWidth);$('pan').value=String(maxPan?Math.round(1000*viewStart/maxPan):0);$('zoom').value=String(Math.round(100*Math.log(full/viewWidth)/Math.log(Math.max(1.001,full/350))));$('viewtime').textContent=(viewStart/1000).toFixed(2)+'–'+((viewStart+viewWidth)/1000).toFixed(2)+' s';}
function spectrumFor(b){const x=cur().p.prompt_index;if(specCache.has(x))return specCache.get(x);const y=b.getChannelData(0),sr=b.sampleRate,nfft=512,hop=Math.max(1,Math.round(sr*.006)),cols=Math.max(1,Math.floor((y.length-nfft)/hop)+1),bins=128,grid=new Uint8Array(cols*bins),window=new Float64Array(nfft);for(let i=0;i<nfft;i++)window[i]=.5-.5*Math.cos(2*Math.PI*i/(nfft-1));const cos=new Float64Array(nfft/2),sin=new Float64Array(nfft/2);for(let k=0;k<nfft/2;k++){cos[k]=Math.cos(2*Math.PI*k/nfft);sin[k]=Math.sin(2*Math.PI*k/nfft);}for(let col=0;col<cols;col++){const start=col*hop;let sum=0;for(let k=0;k<bins;k++){const fftBin=Math.round((k/bins)*8000*nfft/sr);if(fftBin>nfft/2){continue;}let re=0,im=0;for(let j=0;j<nfft;j++){const angle=2*Math.PI*fftBin*j/nfft;const v=y[start+j]*window[j];re+=v*Math.cos(angle);im-=v*Math.sin(angle);}const db=20*Math.log10(Math.hypot(re,im)+1e-8);grid[col*bins+k]=Math.max(0,Math.min(255,Math.round((db+65)*3.4)));}}const result={grid,cols,bins,hop,sr,nfft};specCache.set(x,result);return result;}
function drawSpectrum(b,bd){const c=$('spectrum'),g=c.getContext('2d'),w=Math.max(1,Math.round(c.clientWidth)),h=Math.max(1,Math.round(c.clientHeight));if(c.width!==w||c.height!==h){c.width=w;c.height=h;}const sp=spectrumFor(b),pix=g.createImageData(w,h),start=bd.start,span=Math.max(.001,bd.end-bd.start);for(let px=0;px<w;px++){const col=Math.min(sp.cols-1,Math.max(0,Math.round((start+px/w*span)*sp.sr/sp.hop)));for(let py=0;py<h;py++){const bin=Math.min(sp.bins-1,Math.floor((1-py/h)*sp.bins));const v=sp.grid[col*sp.bins+bin],i=(py*w+px)*4;pix.data[i]=Math.round(v*.95);pix.data[i+1]=Math.round(v*.74);pix.data[i+2]=Math.round(v*.58);pix.data[i+3]=255;}}g.putImageData(pix,0,0);const cue=cur().p.cues_ms[cur().o-1],next=cur().p.next_cues_ms[cur().o-1];const line=(v,color)=>{const xx=(v/1000-start)/span*w;g.strokeStyle=color;g.lineWidth=2;g.beginPath();g.moveTo(xx,0);g.lineTo(xx,h);g.stroke();};line(cue,'#4aa7ff');line(next,'#ad6aff');const l=lab(cur().p,cur().o);if(l)line(cue+l.anchor_ms_after_cue,'#ff4545');}
function render(){if(!data)return;const x=cur(),p=x.p,o=x.o,l=lab(p,o),total=flat().length;$('step').textContent=lang==='zh'?`第 ${stepIndex+1} / ${total} 个标记`:`Mark ${stepIndex+1} / ${total}`;$('title').textContent=lang==='zh'?`现在标：${p.syllable} · 第 ${o} 遍`:`Now mark: ${p.syllable} · occurrence ${o}`;$('subtitle').textContent=`${p.base_unit} · ${p.context_family} · ${p.class_name}`;$('short').textContent=p.instruction[lang==='zh'?'short_zh':'short_en'];$('instruction').textContent=p.instruction[lang];$('dont').textContent=t('dont');$('slot').textContent=lang==='zh'?`这一拍长度：${p.slot_ms[o-1].toFixed(1)} ms`:`Beat slot: ${p.slot_ms[o-1].toFixed(1)} ms`;$('mode').textContent=lang==='zh'?'显示的是整拍，不是固定 520 ms 截窗':'Full beat slot shown; no fixed 520 ms crop';$('readout').textContent=l?(lang==='zh'?`你标在当前 beep 后 ${l.anchor_ms_after_cue.toFixed(1)} ms · ${l.status==='uncertain'?'不确定':'确定'}`:`Marker ${l.anchor_ms_after_cue.toFixed(1)} ms after cue · ${l.status}`):'—';$('playClick').disabled=!l;$('clear').disabled=!l;$('prev').disabled=stepIndex===0;$('next').disabled=stepIndex===total-1;$('progress').textContent=`${doneCount()}/${total}`;draw()}
function fillRange(g,a,b,bd,w,h,color){const range=bd.displayEnd-bd.displayStart,x1=(a-bd.displayStart)/range*w,x2=(b-bd.displayStart)/range*w;g.fillStyle=color;g.fillRect(Math.max(0,x1),0,Math.max(0,Math.min(w,x2)-Math.max(0,x1)),h)}
function lineAt(g,value,bd,w,h,color,width=2){const x=(value-bd.displayStart)/(bd.displayEnd-bd.displayStart)*w;g.strokeStyle=color;g.lineWidth=width;g.beginPath();g.moveTo(x,0);g.lineTo(x,h);g.stroke();g.lineWidth=1}
function draw(){const c=$('wave'),g=c.getContext('2d'),dpr=window.devicePixelRatio||1,w=Math.max(1,Math.floor(c.clientWidth*dpr)),h=Math.max(1,Math.floor(c.clientHeight*dpr));if(c.width!==w||c.height!==h){c.width=w;c.height=h}g.clearRect(0,0,w,h);const x=cur(),b=buffers.get(x.p.prompt_index);if(!b)return;const bd=bounds(b),ch=b.getChannelData(0),a=Math.max(0,Math.floor(bd.start*b.sampleRate)),z=Math.min(ch.length,Math.floor(bd.end*b.sampleRate)),step=Math.max(1,Math.floor((z-a)/w));fillRange(g,bd.gray1a,bd.gray1b,bd,w,h,'#d7d9d4');fillRange(g,bd.gray2a,bd.gray2b,bd,w,h,'#eceeea');g.strokeStyle='#1e211e';g.beginPath();for(let px=0;px<w;px++){let m=0;for(let j=0;j<step;j++){const k=a+px*step+j;if(k<z)m=Math.max(m,Math.abs(ch[k]))}g.moveTo(px,h/2-m*h*.43);g.lineTo(px,h/2+m*h*.43)}g.stroke();lineAt(g,0,bd,w,h,'#245b97',2);lineAt(g,bd.next-bd.cue,bd,w,h,'#6f4b8b',2);const l=lab(x.p,x.o);if(l)lineAt(g,l.anchor_ms_after_cue,bd,w,h,'#ad3434',3);drawSpectrum(b,bd)}
async function playRange(start,end){const b=await getBuffer();if(!audioCtx)audioCtx=new(window.AudioContext||window.webkitAudioContext)();await audioCtx.resume();const src=audioCtx.createBufferSource();src.buffer=b;src.connect(audioCtx.destination);src.start(0,Math.max(0,start),Math.max(.02,end-start))}
async function playBeat(){const b=await getBuffer(),bd=bounds(b);await playRange(bd.start,bd.end)}async function playWhole(){const b=await getBuffer();await playRange(0,b.duration)}async function playAround(){const b=await getBuffer(),x=cur(),l=lab(x.p,x.o);if(!l)return;const abs=(x.p.cues_ms[x.o-1]+l.anchor_ms_after_cue)/1000;await playRange(abs-.12,Math.min(b.duration,abs+.20))}
async function save(value,status){const x=cur();const r=await fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt_index:x.p.prompt_index,occurrence:x.o,anchor_ms_after_cue:value,status})}),d=await r.json();if(!r.ok){$('status').textContent=d.error||'error';return}data.labels=d.labels||{};$('status').textContent=t('ready');render()}
const clickPlot=async e=>{const b=await getBuffer(),bd=bounds(b),c=$('wave'),r=c.getBoundingClientRect(),frac=(e.clientX-r.left)/r.width,value=bd.displayStart+frac*(bd.displayEnd-bd.displayStart);if(value<bd.labelStart||value>bd.labelEnd){$('status').textContent=lang==='zh'?'标记必须位于录音的有效时间范围内。':'Markers must be within the recording duration.';return}const old=lab(cur().p,cur().o);await save(value,old?.status==='uncertain'?'uncertain':'ok')};$('wave').onclick=clickPlot;$('spectrum').onclick=clickPlot;$('pan').oninput=async()=>{const b=await getBuffer();viewStart=(b.duration*1000-viewWidth)*Number($('pan').value)/1000;setView(b);render()};$('zoom').oninput=async()=>{const b=await getBuffer(),full=b.duration*1000,center=viewStart+viewWidth/2;viewWidth=full/Math.pow(Math.max(1.001,full/350),Number($('zoom').value)/100);viewStart=center-viewWidth/2;setView(b);render()};$('resetview').onclick=async()=>{setView(await getBuffer(),true);render()};$('play').onclick=playBeat;$('playWhole').onclick=playWhole;$('playClick').onclick=playAround;$('uncertain').onclick=async()=>{const x=cur(),l=lab(x.p,x.o);if(!l){$('status').textContent=lang==='zh'?'请先标记参考位置，再设为“不确定”。':'Mark a provisional onset first.';return}await save(l.anchor_ms_after_cue,l.status==='uncertain'?'ok':'uncertain')};$('clear').onclick=()=>save(null,'unset');$('prev').onclick=()=>{if(stepIndex>0){stepIndex--;loadCurrent()}};$('next').onclick=()=>{if(stepIndex<flat().length-1){stepIndex++;loadCurrent()}};$('zh').onclick=()=>{lang='zh';applyLang()};$('en').onclick=()=>{lang='en';applyLang()};async function loadCurrent(){const b=await getBuffer();setView(b,true);render()}(async()=>{const r=await fetch('/setup'),d=await r.json();if(!r.ok){$('status').textContent=d.error||'error';return}data=d;$('session').textContent=d.session_id;applyLang();await loadCurrent();$('status').textContent=t('ready')})();
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
    print(f"PhonoWeave manual anchor development: {url}")
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
