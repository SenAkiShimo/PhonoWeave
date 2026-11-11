"""Bilingual, read-only recording-plan page for the existing local GUI."""
HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhonoWeave — 录音规划</title><style>
*{box-sizing:border-box}body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;background:#f3f5f2;color:#243028;margin:0}
header{background:#e5eae5;padding:14px 24px;display:flex;justify-content:space-between}
main{max-width:960px;margin:20px auto;padding:0 14px}section{background:white;border:1px solid #cbd4ca;border-radius:8px;padding:20px;margin:14px 0}
h1{font-size:25px}h2{font-size:17px}p{color:#58635b}label{display:block;margin-top:12px;font-weight:600;font-size:12px}
input{width:100%;min-width:0;padding:9px;border:1px solid #aeb9ae;border-radius:4px;font:inherit}
button,a.download{display:inline-block;border:1px solid #788779;border-radius:5px;padding:9px 13px;background:white;color:#27352c;text-decoration:none;cursor:pointer;margin:8px 6px 4px 0;font:inherit}
button.primary{background:#263e32;color:white}button:disabled{opacity:.4}.row{display:flex;align-items:end;gap:10px;flex-wrap:wrap}.grow{flex:1;min-width:220px}.small{width:130px}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #e2e6e1;text-align:left;padding:9px;vertical-align:top}
th{background:#f4f6f3}td:first-child{font-family:monospace}#status.error{color:#ad2626}.metric{display:inline-block;padding:12px;background:#f3f5f3;border-radius:5px;margin:5px}
.warning{color:#815700}.note{font-size:12px;color:#626c64}
</style></head><body>
<header><strong>PHONOWEAVE</strong><a href="/">← 分析界面 / Analysis</a></header><main>
<h1>录音规划 / Recording Plan</h1><p>基于现有音频来源和发音覆盖缺口，生成少量补录建议。不会改动原始声库。</p>
<section><div class="row"><div class="grow"><label for="voicebank">声库文件夹 / Voicebank directory</label><input id="voicebank" placeholder="/Users/.../OpenUtau/Singers/Voicebank"></div><button type="button" id="browse">选择文件夹…</button></div>
<div class="row"><div class="small"><label for="budget">最多录音行 / Max lines</label><input type="number" id="budget" min="0" max="10000" step="1" value="8"></div><div class="grow"><label for="existing">已有计划表路径（可选） / Scheduled reclist path</label><input id="existing" placeholder="/path/to/scheduled.txt"></div></div>
<button class="primary" type="button" id="preview">分析并预览 / Preview</button><p id="status" role="status" aria-live="polite">请先选择声库。</p></section>
<section id="result" hidden><h2>建议摘要 / Summary</h2><div id="metrics"></div>
<a class="download" href="/api/recording-plan/download/reclist">下载补录表 (.txt)</a>
<a class="download" href="/api/recording-plan/download/report">下载解释报告 (.json)</a>
<h2>补录内容 / Suggested recording lines</h2><div style="overflow:auto"><table><thead><tr><th>录音行</th><th>上下文</th><th>理由</th><th>已有 PCM</th></tr></thead><tbody id="rows"></tbody></table></div>
<p id="empty" hidden>暂时没有足够证据要求新增录音；这并不代表声库已完整。</p>
<h2>注意事项 / Warnings</h2><div id="warnings"></div>
<p class="note">当前仅支持现有普通话 OTO 分析。PCM 来源数不等于独立演唱次数。补录表仅供核对；不会生成新音素、alias 或修改 oto.ini。</p></section></main>
<script>
const $=id=>document.getElementById(id);let busy=false;
function status(text,err=false){$('status').textContent=text;$('status').classList.toggle('error',err)}
function node(parent,tag,text){const n=document.createElement(tag);n.textContent=String(text??'');parent.appendChild(n);return n}
function reset(){ $('result').hidden=true }
for(const id of ['voicebank','budget','existing'])$(id).addEventListener('input',reset);
async function send(url,payload){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw Error(d.error||'请求失败');return d}
$('browse').onclick=async()=>{if(busy)return;try{const x=await send('/api/pick-folder',{lang:'zh'});if(x.path){$('voicebank').value=x.path;reset();status('声库已选择')}else status('已取消')}catch(e){status(e.message,true)}};
function render(d){const plan=d.recording_plan||{},coverage=d.coverage||{},evidence=d.evidence||{};for(const id of ['metrics','rows','warnings'])$(id).replaceChildren();
for(const [k,v] of [['音节观测',coverage.recognized_syllable_observations],['补录证据缺口',evidence.diagnostic_recording_gaps],['建议新行数',plan.selected_count],['已安排的行数',plan.already_scheduled_count]]){const box=node($('metrics'),'div',k+': '+(v??0));box.className='metric'}
const rows=Array.isArray(plan.selected)?plan.selected:[];$('empty').hidden=rows.length>0;
for(const x of rows){const tr=node($('rows'),'tr','');for(const cell of [x.line,(x.base_unit??'')+' / '+(x.context_family??''),x.gap_type??'',x.existing_distinct_pcm_sources??'—'])node(tr,'td',cell)}
for(const w of (d.warnings||[])){const row=node($('warnings'),'p',w);row.className='warning'}
$('result').hidden=false}
$('preview').onclick=async()=>{if(busy)return;const path=$('voicebank').value.trim(),raw=$('budget').value,max=Number(raw),existing=$('existing').value.trim();
if(!path){status('先选择声库',true);return}
if(!raw||!Number.isSafeInteger(max)||max<0||max>10000){status('录音行数必须是 0–10000 的整数',true);return}
busy=true;$('preview').disabled=true;reset();status('正在检查声库…');
try{render(await send('/api/recording-plan/preview',{path,max_lines:max,existing_reclist:existing}));status('已完成，可下载补录表和解释报告')}
catch(e){status(e.message,true)}finally{busy=false;$('preview').disabled=false}};
</script></body></html>'''