from __future__ import annotations

HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PhonoWeave — Supplement recording</title>
<style>
:root{--line:#c9cac6;--ink:#171816;--muted:#60635e;--sel:#1d4f91;--ok:#24653f;--danger:#8d2f2f;--bg:#f4f4f1}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}
header{display:flex;gap:16px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:#ecece8}
header a{color:var(--sel);text-decoration:none}
header .spacer{flex:1}
header button.lang{border:1px solid var(--line);background:#fff;padding:2px 8px;cursor:pointer}
header button.lang.active{background:var(--sel);color:#fff}
main{max-width:1000px;margin:0 auto;padding:16px}
h2{font-size:15px;margin:0 0 8px}
.panel{background:#fff;border:1px solid var(--line);padding:14px;margin-bottom:14px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0}
textarea,input[type=text],input[type=number],select{font:inherit;padding:5px 7px;border:1px solid var(--line);background:#fff}
textarea{width:100%;height:150px;font-family:ui-monospace,Menlo,monospace}
button{font:inherit;padding:6px 12px;border:1px solid #8a8c88;background:#fff;cursor:pointer}
button.primary{background:var(--sel);border-color:var(--sel);color:#fff}
button:disabled{opacity:.45;cursor:default}
.hint{color:var(--muted);font-size:13px}
.work{display:grid;grid-template-columns:260px 1fr;gap:14px}
.list{max-height:520px;overflow:auto;border:1px solid var(--line);background:#fff}
.item{padding:6px 10px;border-bottom:1px solid #eee;cursor:pointer;display:flex;gap:8px;font-family:ui-monospace,Menlo,monospace;font-size:12px}
.item.sel{background:#e3ecf8}
.item .no{color:var(--muted);width:22px}
.item .txt{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.item .ok{color:var(--ok)}
.tokens{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0;min-height:56px}
.token{font-size:26px;padding:4px 12px;border:2px solid var(--line);border-radius:6px;background:#fafaf8}
.token.hot{border-color:var(--sel);background:#e3ecf8}
canvas{width:100%;height:70px;background:#fff;border:1px solid var(--line);display:block;margin:6px 0}
.meter{height:10px;background:#e4e4e0;margin:6px 0}
.meter div{height:100%;width:0;background:var(--ok)}
.meter div.hot{background:var(--danger)}
#status{color:var(--muted);margin-top:8px;min-height:20px}
#status.error{color:var(--danger)}
</style>
</head>
<body>
<header><strong>PHONOWEAVE</strong><a href="/">Workbench</a><span id="title"></span><span class="spacer"></span>
<button class="lang" id="zh">中文</button><button class="lang" id="en">EN</button></header>
<main>
<section class="panel" id="setup">
<h2 id="t-setup"></h2>
<div class="hint" id="t-lines-hint"></div>
<textarea id="lines" spellcheck="false"></textarea>
<div class="row"><label><span id="t-layers"></span> <input type="text" id="layers" size="22"></label>
<label><span id="t-bpm"></span> <input type="number" id="bpm" min="30" max="240" value="90" style="width:80px"></label>
<label><input type="checkbox" id="reftone" checked> <span id="t-reftone"></span></label></div>
<div class="row"><button class="primary" id="create"></button>
<span class="hint" id="t-or"></span>
<input type="text" id="open-path" size="40" placeholder="/Users/.../session_20260101_120000">
<button id="open"></button></div>
</section>
<section id="session" hidden>
<div class="panel"><div class="row"><span id="session-dir" class="hint"></span></div>
<div class="row"><label><span id="t-layer"></span> <select id="layer"></select></label>
<button id="mic"></button><button id="tone"></button><span class="hint" id="t-phones"></span></div></div>
<div class="work">
<div><div class="list" id="list"></div><div class="hint" id="count"></div></div>
<div class="panel">
<div class="hint" id="line-label"></div>
<div class="tokens" id="tokens"></div>
<div class="meter"><div id="level"></div></div>
<div class="row"><button class="primary" id="record"></button><button id="play"></button><button id="prev"></button><button id="next"></button></div>
<canvas id="take"></canvas>
<audio id="audio" hidden></audio>
<div class="hint" id="take-meta"></div>
</div>
</div>
</section>
<div id="status"></div>
</main>
<script>
const T={
zh:{title:'补录',setup:'新建录音会话',linesHint:'每行是一串用下划线连接的音节，按顺序在节拍上唱出来。',layers:'音高层（空格分隔）',bpm:'速度 BPM',reftone:'录前播放参考音',create:'新建会话',or:'或打开已有会话：',open:'打开',layer:'音高层',mic:'启用麦克风',tone:'参考音',phones:'请戴耳机，避免节拍声被录进去。',record:'录制',play:'回放',prev:'上一行',next:'下一行',counts:'已录',micReady:'麦克风已启用。',micFail:'无法使用麦克风：',saved:'已保存：',noLines:'请先填入要录的行。'},
en:{title:'Supplement recording',setup:'New recording session',linesHint:'Each line is a string of syllables joined by underscores, sung in order on the beat.',layers:'Pitch layers (space separated)',bpm:'Tempo (BPM)',reftone:'Play a reference tone before recording',create:'New session',or:'or open an existing session:',open:'Open',layer:'Pitch layer',mic:'Enable microphone',tone:'Reference tone',phones:'Use headphones so the click is not recorded.',record:'Record',play:'Play',prev:'Previous',next:'Next',counts:'Recorded',micReady:'Microphone enabled.',micFail:'Could not use the microphone: ',saved:'Saved: ',noLines:'Enter the lines to record first.'}};
const $=id=>document.getElementById(id);
let lang=localStorage.getItem('phonoweave.language')||'zh',S=null,layer='',index=0,stream=null,ctx=null,src=null,proc=null,analyser=null,chunks=[],capStart=null,busy=false,ever=null;
const t=k=>T[lang][k];
function setStatus(s,err=false){$('status').textContent=s;$('status').classList.toggle('error',err)}
function applyLang(){
  document.documentElement.lang=lang==='zh'?'zh-CN':'en';localStorage.setItem('phonoweave.language',lang);
  $('zh').classList.toggle('active',lang==='zh');$('en').classList.toggle('active',lang==='en');
  [['title','title'],['t-setup','setup'],['t-lines-hint','linesHint'],['t-layers','layers'],['t-bpm','bpm'],['t-reftone','reftone'],['create','create'],['t-or','or'],['open','open'],['t-layer','layer'],['mic','mic'],['tone','tone'],['t-phones','phones'],['record','record'],['play','play'],['prev','prev'],['next','next']].forEach(([id,k])=>$(id).textContent=t(k));
  if(S)render();
}
async function api(path,body){
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json();if(!r.ok)throw new Error(d.error||'error');return d;
}
async function loadDefaults(){
  try{const d=await (await fetch('/api/supplement/defaults')).json();
    $('lines').value=d.lines||'';$('layers').value=(d.layers||['G3','D4','G4','D5']).join(' ');if(d.bpm)$('bpm').value=d.bpm}catch(e){}
}
function enter(d){
  S=d;layer=S.layers[0];index=0;
  $('setup').hidden=true;$('session').hidden=false;
  $('session-dir').textContent=S.directory;
  $('layer').innerHTML=S.layers.map(l=>`<option>${l}</option>`).join('');
  render();showTake();
}
function isDone(l,i){return S.completed[l].includes(i)}
function render(){
  $('list').innerHTML=S.lines.map((line,i)=>`<div class="item ${i===index?'sel':''}" data-i="${i}"><span class="no">${i+1}</span><span class="txt">${line.join('_')}</span><span class="ok">${isDone(layer,i)?'✓':''}</span></div>`).join('');
  document.querySelectorAll('.item').forEach(x=>x.onclick=()=>{if(!busy){index=Number(x.dataset.i);render();showTake()}});
  $('count').textContent=`${t('counts')} ${S.completed[layer].length}/${S.lines.length}`;
  $('line-label').textContent=`${index+1} / ${S.lines.length} · ${layer} · ${S.bpm} BPM`;
  $('tokens').innerHTML=S.lines[index].map((x,i)=>`<span class="token" id="tok${i}">${x}</span>`).join('');
  $('record').disabled=!stream||busy;$('play').disabled=!isDone(layer,index)||busy;
  $('prev').disabled=busy||index<=0;$('next').disabled=busy||index>=S.lines.length-1;
}
async function enableMic(){
  try{
    stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
    ctx=new (window.AudioContext||window.webkitAudioContext)();
    src=ctx.createMediaStreamSource(stream);analyser=ctx.createAnalyser();analyser.fftSize=1024;src.connect(analyser);
    meter();setStatus(t('micReady'));render();
  }catch(e){setStatus(t('micFail')+e.message,true)}
}
function meter(){
  const a=new Float32Array(analyser.fftSize);analyser.getFloatTimeDomainData(a);
  let p=0;for(const v of a)p=Math.max(p,Math.abs(v));
  const el=$('level');el.style.width=Math.min(100,p*100)+'%';el.classList.toggle('hot',p>0.98);
  requestAnimationFrame(meter);
}
function freqOf(name){
  const m=/([A-G])([#b]?)(-?\d)$/.exec(name);if(!m)return null;
  const base={C:0,D:2,E:4,F:5,G:7,A:9,B:11}[m[1]]+(m[2]==='#'?1:m[2]==='b'?-1:0);
  const midi=(Number(m[3])+1)*12+base;return 440*Math.pow(2,(midi-69)/12);
}
function tone(freq,when,len){
  const o=ctx.createOscillator(),g=ctx.createGain();o.frequency.value=freq;
  g.gain.setValueAtTime(0.0001,when);g.gain.exponentialRampToValueAtTime(0.12,when+0.05);
  g.gain.setValueAtTime(0.12,when+len-0.08);g.gain.exponentialRampToValueAtTime(0.0001,when+len);
  o.connect(g);g.connect(ctx.destination);o.start(when);o.stop(when+len+0.02);
}
function click(when,accent){
  const o=ctx.createOscillator(),g=ctx.createGain();o.frequency.value=accent?1000:700;
  g.gain.setValueAtTime(0.06,when);g.gain.exponentialRampToValueAtTime(0.0001,when+0.06);
  o.connect(g);g.connect(ctx.destination);o.start(when);o.stop(when+0.07);
}
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function startCapture(){
  chunks=[];capStart=null;
  proc=ctx.createScriptProcessor(4096,1,1);
  proc.onaudioprocess=e=>{
    if(capStart===null)capStart=e.playbackTime-e.inputBuffer.length/ctx.sampleRate;
    chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  src.connect(proc);proc.connect(ctx.destination);
}
function stopCapture(){
  proc.disconnect();src.disconnect(proc);
  const n=chunks.reduce((a,c)=>a+c.length,0),out=new Float32Array(n);let o=0;
  for(const c of chunks){out.set(c,o);o+=c.length}return out;
}
function encodeWav(samples,sr){
  const b=new ArrayBuffer(44+samples.length*2),v=new DataView(b);
  const s=(o,x)=>{for(let i=0;i<x.length;i++)v.setUint8(o+i,x.charCodeAt(i))};
  s(0,'RIFF');v.setUint32(4,36+samples.length*2,true);s(8,'WAVE');s(12,'fmt ');
  v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,sr,true);
  v.setUint32(28,sr*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);s(36,'data');v.setUint32(40,samples.length*2,true);
  let o=44;for(let i=0;i<samples.length;i++,o+=2){const x=Math.max(-1,Math.min(1,samples[i]));v.setInt16(o,x<0?x*32768:x*32767,true)}
  return b;
}
async function playReference(){
  const f=freqOf(layer);if(!f)return;
  await ctx.resume();tone(f,ctx.currentTime+0.05,1.2);await sleep(1500);
}
async function record(){
  if(busy||!stream)return;
  busy=true;render();
  try{
    await ctx.resume();
    if($('reftone').checked)await playReference();
    const line=S.lines[index],beat=60/S.bpm;
    startCapture();
    await sleep(250);
    const first=ctx.currentTime+0.3,times=[];
    click(first,true);                       // count-in
    for(let i=0;i<line.length;i++){const when=first+(i+1)*beat;times.push(when);click(when,i===0)}
    const end=first+(line.length+2)*beat;click(end,true);
    await new Promise(resolve=>{
      const tick=()=>{
        const now=ctx.currentTime;
        times.forEach((w,i)=>{const el=$('tok'+i);if(el)el.classList.toggle('hot',now>=w&&now<w+beat)});
        if(now<end+beat*0.5)requestAnimationFrame(tick);else resolve();
      };tick();
    });
    document.querySelectorAll('.token').forEach(x=>x.classList.remove('hot'));
    const samples=stopCapture(),sr=ctx.sampleRate;
    const beats=times.map(w=>Math.round((w-capStart)*1000));
    const q=new URLSearchParams({layer,index,sample_rate:sr,beats:beats.join(',')});
    const r=await fetch('/api/supplement/recording?'+q,{method:'POST',headers:{'Content-Type':'audio/wav'},body:encodeWav(samples,sr)});
    const d=await r.json();if(!r.ok)throw new Error(d.error||'save failed');
    if(!S.completed[layer].includes(index))S.completed[layer].push(index);
    setStatus(t('saved')+d.saved);
  }catch(e){setStatus(e.message,true);try{stopCapture()}catch(_){}}
  busy=false;render();showTake();
}
async function showTake(){
  const c=$('take'),d=window.devicePixelRatio||1;c.width=c.clientWidth*d;c.height=c.clientHeight*d;
  const g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
  $('audio').removeAttribute('src');$('take-meta').textContent='';
  if(!isDone(layer,index))return;
  const key=layer+'|'+index;ever=key;
  const r=await fetch(`/api/supplement/recording?layer=${encodeURIComponent(layer)}&index=${index}&t=${Date.now()}`);
  if(!r.ok||ever!==key)return;
  const blob=await r.blob();$('audio').src=URL.createObjectURL(blob);
  try{
    const ac=new (window.AudioContext||window.webkitAudioContext)();
    const buf=await ac.decodeAudioData(await blob.arrayBuffer());await ac.close();
    const x=buf.getChannelData(0),step=Math.max(1,Math.floor(x.length/c.width));
    g.beginPath();
    for(let px=0;px<c.width;px++){let m=0;for(let j=0;j<step;j++){const i=px*step+j;if(i<x.length)m=Math.max(m,Math.abs(x[i]))}
      g.moveTo(px,c.height/2-m*c.height*.46);g.lineTo(px,c.height/2+m*c.height*.46)}
    g.stroke();$('take-meta').textContent=`${buf.duration.toFixed(2)} s`;
  }catch(e){}
}
$('create').onclick=async()=>{
  if(!$('lines').value.trim()){setStatus(t('noLines'),true);return}
  try{enter(await api('/api/supplement/session',{lines:$('lines').value,layers:$('layers').value,bpm:Number($('bpm').value)}))}
  catch(e){setStatus(e.message,true)}
};
$('open').onclick=async()=>{
  try{enter(await api('/api/supplement/open',{path:$('open-path').value.trim()}))}catch(e){setStatus(e.message,true)}
};
$('layer').onchange=()=>{layer=$('layer').value;render();showTake()};
$('mic').onclick=enableMic;
$('tone').onclick=async()=>{if(!ctx){setStatus(t('mic'),true);return}await playReference()};
$('record').onclick=record;
$('play').onclick=()=>{if($('audio').src)$('audio').play()};
$('prev').onclick=()=>{if(index>0){index--;render();showTake()}};
$('next').onclick=()=>{if(index<S.lines.length-1){index++;render();showTake()}};
$('zh').onclick=()=>{lang='zh';applyLang()};$('en').onclick=()=>{lang='en';applyLang()};
applyLang();loadDefaults();
</script>
</body>
</html>
"""
