from __future__ import annotations

HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PhonoWeave — Negative Controls</title>
<style>
:root{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;color:#222;background:#f4f4f2}
body{max-width:820px;padding:24px;margin:auto;font-size:14px}
h1{font-size:23px;margin-bottom:4px}
main{display:grid;gap:16px}
section{border:1px solid #bbb;background:white;padding:18px}
label{font-weight:600;display:block;margin-bottom:7px}
select,button{font:inherit;padding:9px;border:1px solid #888;background:#fff;max-width:100%}
select{width:100%}
button{cursor:pointer;background:#292a29;color:white;border-color:#292a29}
button:disabled{opacity:.45;cursor:default}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}
.buttons{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
#status{font-family:ui-monospace,monospace;white-space:pre-wrap}
audio{width:100%}
.small{font-size:12px;color:#666}
a{color:#23507f}
</style>
</head>
<body>
<a href="/calibration">← Live Calibration</a>
<h1>Negative Control Recording · 负对照录音</h1>

<main>
<section>
<div class="grid">
<div><label for="category">类别 / Category</label>
<select id="category">
<option value="vowel_a">单独元音 /a/</option>
<option value="vowel_i">单独元音 /i/</option>
<option value="breath">呼气声 / Breath</option>
<option value="silence">静音 / Silence</option>
<option value="room_noise">环境声 / Ambient noise</option>
<option value="handling_noise">操作噪声 / Handling noise</option>
</select></div>
<div><label for="device">输入设备类别 / Input type</label>
<select id="device"><option value="unknown">未指定 / Unknown</option>
<option value="phone">Phone</option><option value="built_in">Built-in microphone</option>
<option value="external_usb">External / USB microphone</option>
<option value="other">Other</option></select></div>
<div><label for="environment">环境 / Environment</label>
<select id="environment"><option value="quiet">安静 / Quiet</option>
<option value="indoor">室内有噪声 / Indoor noise</option>
<option value="outdoor">室外 / Outdoor</option>
<option value="other">其他 / Other</option></select></div>
</div>
</section>
<section>
<div class="buttons">
<button id="enable">Enable microphone · 启用麦克风</button>
<button id="record" disabled>Record 3 s · 录制 3 秒</button>
</div>

<div id="status" role="status">Microphone not enabled.</div>
<audio id="preview" controls></audio>
<div class="small" id="saved"></div>
</section>

</main>
<script>
const $=id=>document.getElementById(id);
let stream=null,ctx=null,source=null,busy=false,previewUrl=null;
function status(s){$('status').textContent=s}
function encodeWav(samples,sr){
 const b=new ArrayBuffer(44+samples.length*2),v=new DataView(b);
 function put(pos,s){for(let i=0;i<s.length;i++)v.setUint8(pos+i,s.charCodeAt(i))}
 put(0,'RIFF');v.setUint32(4,36+samples.length*2,true);put(8,'WAVEfmt ');
 v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
 v.setUint32(24,sr,true);v.setUint32(28,sr*2,true);v.setUint16(32,2,true);
 v.setUint16(34,16,true);put(36,'data');v.setUint32(40,samples.length*2,true);
 for(let i=0;i<samples.length;i++){const x=Math.max(-1,Math.min(1,samples[i]));v.setInt16(44+i*2,x<0?x*32768:x*32767,true)}
 return b;
}
$('enable').onclick=async()=>{
 try {
  if(stream){stream.getTracks().forEach(t=>t.stop());if(ctx)await ctx.close()}
  stream=await navigator.mediaDevices.getUserMedia({audio:{
   channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false
  }});
  ctx=new (window.AudioContext||window.webkitAudioContext)();
  source=ctx.createMediaStreamSource(stream);
  await ctx.resume();
  $('record').disabled=false;status('Microphone enabled. Audio rate: '+ctx.sampleRate+' Hz.');
 }catch(e){status('Microphone error: '+String(e));}
};
$('record').onclick=async()=>{
 if(busy||!stream)return;
 busy=true;$('record').disabled=true;$('enable').disabled=true;
 const category=$('category').value,device=$('device').value,environment=$('environment').value;
 const track=stream.getAudioTracks()[0],s=track?.getSettings?.()||{},chunks=[];
 const processor=ctx.createScriptProcessor(4096,1,1);
 const muted=ctx.createGain();muted.gain.value=0;
 processor.onaudioprocess=e=>chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
 source.connect(processor);processor.connect(muted);muted.connect(ctx.destination);
 status('● REC  ·  3 seconds');
 try {
  await new Promise(resolve=>setTimeout(resolve,3000));
  source.disconnect(processor);processor.disconnect();muted.disconnect();
  const count=chunks.reduce((a,b)=>a+b.length,0),samples=new Float32Array(count);
  let pos=0;for(const block of chunks){samples.set(block,pos);pos+=block.length}
  const wav=encodeWav(samples,ctx.sampleRate);
  const params=new URLSearchParams({category,device_category:device,environment});
  for(const key of ['echoCancellation','noiseSuppression','autoGainControl']){
   if(typeof s[key]==='boolean')params.set(key,String(s[key]));
  }
  const r=await fetch('/api/negative-controls/recording?'+params,{
   method:'POST',headers:{'Content-Type':'audio/wav'},body:wav
  });
  const data=await r.json();
  if(!r.ok)throw Error(data.error||('HTTP '+r.status));
  if(previewUrl)URL.revokeObjectURL(previewUrl);
  previewUrl=URL.createObjectURL(new Blob([wav],{type:'audio/wav'}));
  $('preview').src=previewUrl;
  $('saved').textContent=data.saved;
  status('Saved: '+category+' · '+(samples.length/ctx.sampleRate).toFixed(2)+' s');
 }catch(e){status('Recording error: '+String(e))}
 finally{
  try{source.disconnect(processor);processor.disconnect();muted.disconnect()}catch(e){}
  busy=false;$('record').disabled=false;$('enable').disabled=false;
 }
};
</script>
</body>
</html>"""
