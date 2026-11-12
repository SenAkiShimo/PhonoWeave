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
<label for="native-reclist">原有录音模板文件（可选） / Existing native recording template (optional)</label>
<input id="native-reclist" placeholder="/path/to/existing_reclist.txt or targets.json">
<p class="note">填写后按原始音节/录音名检查各音高子库缺失的 WAV/OTO；留空继续使用普通话分析。不会推断新音素。</p>
<button class="primary" type="button" id="preview">分析并预览 / Preview</button><p id="status" role="status" aria-live="polite">请先选择声库。</p></section>
<section id="result" hidden><h2>建议摘要 / Summary</h2><div id="metrics"></div>
<a class="download" href="/api/recording-plan/download/reclist">下载补录表 (.txt)</a>
<a class="download" href="/api/recording-plan/download/report">下载解释报告 (.json)</a>
<a class="download" href="/api/recording-plan/download/review">下载待检查清单 (.txt)</a>
<h2>补录内容 / Suggested recording lines</h2><div style="overflow:auto"><table><thead><tr><th>录音行</th><th>上下文</th><th>理由</th><th>已有 PCM</th></tr></thead><tbody id="rows"></tbody></table></div>
<p id="empty" hidden>暂时没有足够证据要求新增录音；这并不代表声库已完整。</p>
<h2>检查 OTO / OTO configuration review</h2><div style="overflow:auto"><table><thead><tr><th>录音名</th><th>子声库</th><th>原因</th></tr></thead><tbody id="oto-review"></tbody></table></div>
<h2>核对来源或音频 / Source verification</h2><div style="overflow:auto"><table><thead><tr><th>录音名</th><th>子声库</th><th>原因</th></tr></thead><tbody id="source-review"></tbody></table></div>
<p class="note">损坏的 WAV 先检查或恢复原始文件，不会自动要求重新演唱。配置问题和不确定来源均不会进入补录 TXT。</p>
<h2>注意事项 / Warnings</h2><div id="warnings"></div>
<p class="note">默认使用普通话 OTO 分析；填写原有录音模板后，保留原始名称进行文件覆盖检查。PCM 来源数不等于独立演唱次数。不会生成新音素、alias 或修改 oto.ini。</p></section>
<section id="asset-input">
<h2>已有音频素材 / Existing source assets</h2>
<p>默认直接扫描上方选择的当前声库 WAV/OTO，查看 alias、来源文件和重复或缺失状态；也可改用历史研究 ZIP。两种模式都不会改动声库。</p>
<label for="asset-mode">数据来源 / Source</label>
<select id="asset-mode"><option value="live" selected>当前声库 WAV / OTO（推荐）</option><option value="historical">历史来源索引 ZIP</option></select>
<p id="live-asset-note" class="note">当前声库模式直接使用上方的声库目录。未提供显式映射时，保留原始 alias，不猜测发音标签。</p>
<div id="historic-asset-input">
<label for="asset-archive">来源索引 ZIP / Historical source-index archive</label>
<input id="asset-archive" placeholder="/path/to/PhonoWeave_User_Native_Batch_Discovery_v2.zip"></div>
<label for="asset-labels">原始发音名称（逗号分隔；当前声库可留空） / Original labels (optional in live mode)</label>
<input id="asset-labels" value="i, ir, i0">
<button type="button" id="asset-preview">查看已有素材 / Show existing assets</button>
<p id="asset-status" role="status" aria-live="polite">尚未载入来源索引。</p>
<div id="asset-result" hidden>
<p id="asset-provenance" class="note">原始 alias 的 WAV 文件关联不代表发音单元可以互相替代。</p>
<div id="asset-metrics"></div>
<a class="download" href="/api/recording-plan/assets/download/csv">下载映射 CSV</a>
<a class="download" href="/api/recording-plan/assets/download/json">下载完整来源 JSON</a>
<div class="row">
<div class="grow"><label for="asset-bank">子声库筛选 / Bank</label><select id="asset-bank"><option value="">全部</option></select></div>
<div class="grow"><label for="asset-context">后续环境筛选 / Context</label><input id="asset-context" placeholder="例：sh"></div>
<div class="grow"><label for="asset-unit">发音标签 / Unit</label><select id="asset-unit"><option value="">全部</option></select></div>
<div class="grow"><label for="asset-only-ambiguous"><input id="asset-only-ambiguous" type="checkbox" style="width:auto"> 只看多来源映射 / Ambiguous only</label></div>
</div>
<p id="asset-visible-count" class="note"></p>
<div style="overflow:auto"><table><thead><tr><th>子声库</th><th>原始单元</th><th>后续环境</th><th>原始 alias</th><th>已有 WAV 来源</th><th>判定</th></tr></thead><tbody id="asset-rows"></tbody></table></div>
<p class="note">为保持页面流畅，只显示当前筛选结果的前 80 项；下载文件包含全部映射。来源检查不改变补录表。</p>
</div>
</section></main>
<script>
const $=id=>document.getElementById(id);let busy=false;
function status(text,err=false){$('status').textContent=text;$('status').classList.toggle('error',err)}
function node(parent,tag,text){const n=document.createElement(tag);n.textContent=String(text??'');parent.appendChild(n);return n}
function reset(){ $('result').hidden=true }
for(const id of ['voicebank','budget','existing','native-reclist'])$(id).addEventListener('input',reset);
async function send(url,payload){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw Error(d.error||'请求失败');return d}
$('browse').onclick=async()=>{if(busy)return;try{const x=await send('/api/pick-folder',{lang:'zh'});if(x.path){$('voicebank').value=x.path;reset();status('声库已选择')}else status('已取消')}catch(e){status(e.message,true)}};
function render(d){const plan=d.recording_plan||{},coverage=d.coverage||{},evidence=d.evidence||{};for(const id of ['metrics','rows','warnings','oto-review','source-review'])$(id).replaceChildren();
for(const [k,v] of [[d.scope==='explicit_user_native_recording_templates_no_language_inference'?'原有录音模板':'音节观测',coverage.expected_template_count??coverage.recognized_syllable_observations],['待补录模板',evidence.diagnostic_recording_gaps],['建议新行数',plan.selected_count],['已安排的行数',plan.already_scheduled_count]]){const box=node($('metrics'),'div',k+': '+(v??0));box.className='metric'}
const rows=Array.isArray(plan.selected)?plan.selected:[];$('empty').hidden=rows.length>0;
for(const x of rows){const tr=node($('rows'),'tr','');for(const cell of [x.line,(x.base_unit??'')+' / '+(x.context_family??''),x.gap_type??'',x.existing_distinct_pcm_sources??'—'])node(tr,'td',cell)}
for(const task of (plan.review_tasks||[])){const id=task.category==='oto_configuration'?'oto-review':'source-review';const tr=node($(id),'tr','');for(const v of [task.line,task.bank,(task.reason||task.status)])node(tr,'td',v)}
for(const w of (d.warnings||[])){const row=node($('warnings'),'p',w);row.className='warning'}
$('result').hidden=false}
$('preview').onclick=async()=>{if(busy)return;const path=$('voicebank').value.trim(),raw=$('budget').value,max=Number(raw),existing=$('existing').value.trim(),native=$('native-reclist').value.trim();
if(!path){status('先选择声库',true);return}
if(!raw||!Number.isSafeInteger(max)||max<0||max>10000){status('录音行数必须是 0–10000 的整数',true);return}
busy=true;$('preview').disabled=true;reset();status('正在检查声库…');
try{render(await send('/api/recording-plan/preview',{path,max_lines:max,existing_reclist:existing,native_reclist:native}));status('已完成，可下载补录表和解释报告')}
catch(e){status(e.message,true)}finally{busy=false;$('preview').disabled=false}};

let assetReport=null, assetGeneration=0, assetBusy=false;
const assetIds=['asset-archive','asset-labels','asset-mode','voicebank','native-reclist'];
function updateAssetMode(){const live=$('asset-mode').value==='live';$('historic-asset-input').hidden=live;$('live-asset-note').hidden=!live;}
updateAssetMode();
function invalidateAssets(){assetGeneration++;assetReport=null;$('asset-result').hidden=true;$('asset-status').textContent='输入已修改，请重新检查来源。';updateAssetMode();}
for(const id of assetIds)$(id).addEventListener('input',invalidateAssets);
function assetOptions(id,values){const el=$(id);el.replaceChildren();const opt=node(el,'option','全部');opt.value='';for(const value of [...new Set(values)].sort()){const child=node(el,'option',value);child.value=value}}
function renderAssetRows(){
  if(!assetReport)return;
  const bank=$('asset-bank').value,context=$('asset-context').value.trim(),unit=$('asset-unit').value,only=$('asset-only-ambiguous').checked;
  const list=assetReport.original_alias_routes.filter(x=>
    (!bank||x.bank===bank)&&(!unit||x.original_unit_label===unit)&&
    (!context||x.following_context.includes(context))&&
    (!only||x.routing_status!=='single_current_oto_source'&&x.routing_status!=='single_observed_source_in_filtered_index'));
  $('asset-rows').replaceChildren();
  for(const x of list.slice(0,80)){
    const tr=node($('asset-rows'),'tr','');
    const sources=(x.observed_source_options||[]).map(y=>y.source_wav).join(' | ');
    for(const value of [x.bank,x.original_unit_label,x.following_context,(x.original_aliases||[]).join(' | '),sources,x.routing_status])node(tr,'td',value);
  }
  $('asset-visible-count').textContent='匹配 '+list.length+' 项；当前展示 '+Math.min(list.length,80)+' 项。多来源项需要人工核实，不能直接合并。';
}
for(const id of ['asset-bank','asset-context','asset-unit','asset-only-ambiguous'])$(id).addEventListener('input',renderAssetRows);
$('asset-preview').onclick=async()=>{
  if(assetBusy)return;
  const mode=$('asset-mode').value;
  const archive=$('asset-archive').value.trim();
  const labels=$('asset-labels').value.split(',').map(x=>x.trim()).filter(Boolean);
  const voicebank=$('voicebank').value.trim();
  if((mode==='live'&&!voicebank)||(mode==='historical'&&!archive)||
     labels.length>5||new Set(labels).size!==labels.length||
     (mode==='historical'&&labels.length<2)){
    $('asset-status').textContent='请选择声库（或历史 ZIP），并填写有效的原始发音标签。';return;
  }
  const request=++assetGeneration;
  assetBusy=true;$('asset-preview').disabled=true;$('asset-result').hidden=true;
  $('asset-status').textContent=mode==='live'?'正在读取当前 WAV/OTO…':'正在读取历史来源索引…';
  try{
    const report=await send('/api/recording-plan/assets/preview',{mode,archive,labels,voicebank,native_reclist:$('native-reclist').value.trim()});
    if(request!==assetGeneration)return;
    assetReport=report;
    $('asset-metrics').replaceChildren();
    for(const [name,value] of [['原始 alias / 环境映射',report.total_existing_alias_context_routes],['已识别 PCM 来源',report.distinct_existing_bank_PCM_source_containers],['待核实来源',report.routes_requiring_source_choice_review]]){
      const el=node($('asset-metrics'),'div',name+': '+value);el.className='metric';
    }
    assetOptions('asset-bank',report.per_bank.map(x=>x.bank));
    assetOptions('asset-unit',report.original_user_labels.length?report.original_user_labels:report.original_alias_routes.map(x=>x.original_unit_label));
    $('asset-context').value='';$('asset-only-ambiguous').checked=false;
    $('asset-result').hidden=false;renderAssetRows();
    $('asset-provenance').textContent=report.live_voicebank_scan?'已扫描当前声库 WAV/OTO；未解析标签保留为原始 alias，来源检查不等于发音正确性验证。':'历史来源索引，仅用于检查先前记录的素材关系，不代表当前声库状况。';
    $('asset-status').textContent=report.live_voicebank_scan?'已完成当前声库来源检查；补录建议不变。':'历史来源映射已载入；补录建议不变。';
  }catch(e){if(request===assetGeneration){assetReport=null;$('asset-status').textContent=e.message;$('asset-result').hidden=true}}
  finally{assetBusy=false;$('asset-preview').disabled=false}
};
</script></body></html>'''