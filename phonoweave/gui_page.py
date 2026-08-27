from __future__ import annotations


HTML = r"""<!doctype html>
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
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif;
}
button, input { font: inherit; }
.app { max-width: 1240px; margin: 0 auto; padding: 28px; }
.header { display: flex; justify-content: space-between; align-items: end; gap: 20px; margin-bottom: 22px; }
.header-right { display: flex; flex-direction: column; align-items: end; gap: 9px; }
.brand h1 { margin: 0; font-size: 28px; letter-spacing: -.6px; }
.brand p { margin: 4px 0 0; color: var(--muted); }
.status { color: var(--muted); font-size: 13px; text-align: right; }
.lang-switch { display: inline-flex; border: 1px solid var(--line); border-radius: 9px; padding: 2px; background: var(--panel); }
.lang-switch button { border: 0; border-radius: 6px; padding: 5px 9px; color: var(--muted); background: transparent; }
.lang-switch button.active { color: var(--text); background: var(--accent-soft); font-weight: 700; }
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
.raw { color: var(--muted); font: 11px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace; margin-top: 3px; overflow-wrap: anywhere; }
.section { margin-top: 20px; }
.section h3 { font-size: 13px; margin: 0 0 8px; }
.interpretation { border: 1px solid var(--line); border-radius: 10px; padding: 11px 12px; background: #fafbfc; }
.interpretation p { margin: 0; }
.interpretation .caution { margin-top: 7px; color: var(--muted); font-size: 12px; }
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
      <p id="subtitle"></p>
    </div>
    <div class="header-right">
      <div class="lang-switch" aria-label="Language">
        <button id="lang-zh" type="button">中文</button>
        <button id="lang-en" type="button">EN</button>
      </div>
      <div class="status" id="status"></div>
    </div>
  </div>

  <div class="toolbar">
    <label id="voicebank-label" for="voicebank"></label>
    <input id="voicebank" placeholder="/Users/.../OpenUtau/Singers/Voicebank">
    <button id="browse"></button>
    <button class="primary" id="analyze"></button>
  </div>

  <div class="cards">
    <div class="card"><div class="value" id="onsets">—</div><div class="label" id="onsets-label"></div></div>
    <div class="card"><div class="value" id="units">—</div><div class="label" id="units-label"></div></div>
    <div class="card"><div class="value" id="analyzed">—</div><div class="label" id="analyzed-label"></div></div>
    <div class="card"><div class="value" id="experimental">—</div><div class="label" id="experimental-label"></div></div>
    <div class="card"><div class="value" id="unsupported">—</div><div class="label" id="unsupported-label"></div></div>
  </div>

  <div class="workspace">
    <div class="panel">
      <div class="panel-title" id="decisions-title"></div>
      <div id="table-wrap" class="empty"></div>
    </div>
    <div class="panel">
      <div class="panel-title" id="details-title"></div>
      <div id="details" class="empty"></div>
    </div>
  </div>

  <div class="footer">
    <div id="voicebank-name" class="status"></div>
    <div class="export">
      <a id="profile" class="download disabled" href="/api/export/profile"></a>
      <a id="inventory" class="download disabled" href="/api/export/inventory"></a>
    </div>
  </div>
</div>
<script>
let current = null;
let selected = null;
const $ = id => document.getElementById(id);

const I18N = {
  en: {
    subtitle: 'Speaker realization inventory',
    chooseVoicebank: 'Choose a voicebank to begin.',
    voicebank: 'Voicebank', browse: 'Browse', analyze: 'Analyze', analyzing: 'Analyzing',
    onsets: 'Onsets', units: 'Synthesis units', analyzed: 'Analyzed', experimental: 'Experimental', unsupported: 'Unsupported',
    decisions: 'Onset decisions', details: 'Details', emptyTable: 'Run an analysis to build the speaker realization inventory.', emptyDetails: 'Select an onset after analysis.',
    exportProfile: 'Export Speaker Profile', exportInventory: 'Export Synthesis Inventory',
    onset: 'Onset', class: 'Class', decision: 'Decision', confidence: 'Confidence', acoustic: 'Acoustic evidence', synthesis: 'Synthesis evidence',
    groups: 'Realization groups', notes: 'Analysis notes', interpretation: 'Interpretation',
    openingPicker: 'Opening folder picker…', selected: 'Voicebank selected.', canceled: 'Folder selection canceled.', chooseFirst: 'Choose a voicebank first.', running: 'Analyzing voicebank…',
    analysisFailed: 'Analysis failed.', pickerFailed: 'Folder picker failed.',
    caution: 'Signal-level synthesis relevance is a proxy, not a perceptual listening result.',
    summary: (o, u) => `${o} onsets · ${u} synthesis units`,
  },
  zh: {
    subtitle: '说话人实现库存',
    chooseVoicebank: '请选择一个声库开始分析。',
    voicebank: '声库', browse: '浏览', analyze: '分析', analyzing: '分析中',
    onsets: '声母类型', units: '合成单元', analyzed: '已分析', experimental: '实验性', unsupported: '未支持',
    decisions: '声母判断', details: '详情', emptyTable: '运行分析后，这里会生成说话人的语音实现库存。', emptyDetails: '分析完成后选择一个声母查看详情。',
    exportProfile: '导出说话人配置', exportInventory: '导出合成库存',
    onset: '声母', class: '类别', decision: '判断', confidence: '置信度', acoustic: '声学证据', synthesis: '合成证据',
    groups: '实现分组', notes: '分析记录', interpretation: '结果解释',
    openingPicker: '正在打开文件夹选择器…', selected: '已选择声库。', canceled: '已取消文件夹选择。', chooseFirst: '请先选择声库。', running: '正在分析声库…',
    analysisFailed: '分析失败。', pickerFailed: '文件夹选择失败。',
    caution: '合成相关性目前使用信号层代理指标，不等同于主观听感实验结果。',
    summary: (o, u) => `${o} 个声母 · ${u} 个合成单元`,
  }
};

const TERMS = {
  class_name: {
    fricative: {en: 'Fricative', zh: '擦音'}, affricate: {en: 'Affricate', zh: '塞擦音'}, rhotic: {en: 'Rhotic', zh: 'R 音 / 卷舌音'},
    lateral: {en: 'Lateral', zh: '边音'}, nasal: {en: 'Nasal', zh: '鼻音'}, stop: {en: 'Stop', zh: '塞音'}
  },
  decision: {
    split_recommended: {en: 'Split recommended', zh: '建议区分录制'},
    unresolved: {en: 'Unresolved', zh: '暂不决定'},
    merge_supported: {en: 'Merge supported', zh: '支持合并'},
    three_realizations_provisional: {en: 'Three realizations, provisional', zh: '暂定三种实现'}
  },
  confidence: {
    low: {en: 'Low', zh: '低'}, moderate: {en: 'Moderate', zh: '中等'}, high: {en: 'High', zh: '高'}
  },
  acoustic: {
    weak_or_inconsistent: {en: 'Weak or inconsistent', zh: '较弱或不一致'},
    supported: {en: 'Supported', zh: '得到支持'}, strongly_supported: {en: 'Strongly supported', zh: '得到强支持'},
    partial_coverage_limited: {en: 'Partial, coverage limited', zh: '部分支持，但覆盖受限'},
    mixed: {en: 'Mixed', zh: '证据混合'}, front_distinct_plain_rounded_mixed: {en: 'Front distinct; plain/rounded mixed', zh: '前化实现可区分；普通/圆唇证据混合'},
    front_distinct: {en: 'Front realization distinct', zh: '前化实现可区分'}
  },
  synthesis: {
    not_tested: {en: 'Not tested', zh: '未测试'}, supported_under_proxy: {en: 'Supported under proxy', zh: '当前代理指标支持'},
    split_not_supported_under_proxy: {en: 'Split not supported under proxy', zh: '当前代理指标未支持拆分'},
    unresolved: {en: 'Unresolved', zh: '暂不决定'},
    plain_rounded_split_supported_front_unresolved: {en: 'Plain/rounded split supported; front unresolved', zh: '普通/圆唇拆分得到支持；前化实现未定'},
    three_way_split_supported_under_proxy: {en: 'Three-way split supported under proxy', zh: '当前代理指标支持三分'}
  }
};

const EXPLANATIONS = {
  split_recommended: {
    en: 'The current acoustic and synthesis-relevance evidence supports keeping separate context-conditioned realizations in the synthesis inventory. This does not by itself establish separate linguistic phonemes.',
    zh: '当前声学证据与合成相关性证据支持在合成库存中保留不同的语境实现。这并不等同于证明它们是语言学上的独立音位。'
  },
  unresolved: {
    en: 'The current evidence is not sufficient to justify either a forced split or a merge. PhonoWeave therefore keeps the candidate contexts visible without making a stronger claim.',
    zh: '现有证据不足以支持强制拆分，也不足以支持合并。因此 PhonoWeave 保留候选语境，但暂不作更强的判断。'
  },
  merge_supported: {
    en: 'The current evidence supports sharing one synthesis realization across the tested contexts. This is a synthesis decision, not a claim that the linguistic categories are identical.',
    zh: '当前证据支持在已测试语境之间共用一个合成实现。这是合成层面的判断，并不表示语言学类别本身完全相同。'
  },
  three_realizations_provisional: {
    en: 'Three context-conditioned realizations are provisionally retained. The result remains limited to the tested coverage and current synthesis proxy.',
    zh: '目前暂时保留三种语境实现。该结果仍受现有测试覆盖范围与合成代理指标限制。'
  }
};

function initialLanguage() {
  const stored = localStorage.getItem('phonoweave.language');
  if (stored === 'zh' || stored === 'en') return stored;
  return navigator.language && navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en';
}
let lang = initialLanguage();
function t(key) { return I18N[lang][key]; }
function term(group, value) {
  const item = TERMS[group] && TERMS[group][value];
  return item ? item[lang] : value;
}
function rawTerm(localized, raw) {
  return `<div class="v">${escapeHtml(localized)}</div><div class="raw">${escapeHtml(raw)}</div>`;
}
function contextLabel(raw) {
  if (lang === 'en') return raw;
  const direct = {plain: '普通语境', rounded: '圆唇语境', front: '前化语境', front_unrounded: '前化非圆唇语境'};
  if (direct[raw]) return `${direct[raw]} · ${raw}`;
  const parts = raw.split(':');
  if (parts.length === 2) {
    const role = {internal: '内部', initial: '词首'}[parts[0]] || parts[0];
    const family = {rounded: '圆唇', other: '其他', i_series: 'i 系', u_series: 'u 系', v_series: 'ü 系'}[parts[1]] || parts[1];
    return `${role}：${family} · ${raw}`;
  }
  return raw;
}

function setStatus(text, error=false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
}
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}
function applyLanguage() {
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  localStorage.setItem('phonoweave.language', lang);
  $('lang-zh').classList.toggle('active', lang === 'zh');
  $('lang-en').classList.toggle('active', lang === 'en');
  $('subtitle').textContent = t('subtitle');
  $('voicebank-label').textContent = t('voicebank');
  $('browse').textContent = t('browse');
  $('analyze').textContent = t('analyze');
  $('onsets-label').textContent = t('onsets'); $('units-label').textContent = t('units');
  $('analyzed-label').textContent = t('analyzed'); $('experimental-label').textContent = t('experimental'); $('unsupported-label').textContent = t('unsupported');
  $('decisions-title').textContent = t('decisions'); $('details-title').textContent = t('details');
  $('profile').textContent = t('exportProfile'); $('inventory').textContent = t('exportInventory');
  if (!current) {
    setStatus(t('chooseVoicebank'));
    $('table-wrap').textContent = t('emptyTable');
    $('details').textContent = t('emptyDetails');
  } else {
    renderTable(current.rows);
    if (selected) selectRow(selected);
    setStatus(t('summary')(current.summary.onsets, current.summary.synthesis_units));
  }
}

function renderSummary(summary) {
  for (const key of ['onsets','synthesis_units','analyzed','experimental','unsupported']) {
    const id = key === 'synthesis_units' ? 'units' : key;
    $(id).textContent = summary[key];
  }
}
function renderTable(rows) {
  const html = [`<table><thead><tr><th>${escapeHtml(t('onset'))}</th><th>${escapeHtml(t('class'))}</th><th>${escapeHtml(t('decision'))}</th><th>${escapeHtml(t('confidence'))}</th></tr></thead><tbody>`];
  rows.forEach(row => {
    html.push(`<tr data-base="${escapeHtml(row.base_unit)}">
      <td class="onset">${escapeHtml(row.base_unit)}</td>
      <td>${escapeHtml(term('class_name', row.class_name))}</td>
      <td><span class="badge ${escapeHtml(row.decision)}">${escapeHtml(term('decision', row.decision))}</span></td>
      <td>${escapeHtml(term('confidence', row.confidence))}</td>
    </tr>`);
  });
  html.push('</tbody></table>');
  $('table-wrap').className = '';
  $('table-wrap').innerHTML = html.join('');
  document.querySelectorAll('tbody tr').forEach(tr => tr.addEventListener('click', () => selectRow(tr.dataset.base)));
  if (selected) document.querySelectorAll('tbody tr').forEach(tr => tr.classList.toggle('selected', tr.dataset.base === selected));
}
function selectRow(base) {
  if (!current) return;
  const row = current.rows.find(item => item.base_unit === base);
  if (!row) return;
  selected = base;
  document.querySelectorAll('tbody tr').forEach(tr => tr.classList.toggle('selected', tr.dataset.base === base));
  const groups = row.groups.map((group, i) => `<div class="group"><strong>${escapeHtml(group)}</strong><span>${escapeHtml(contextLabel(row.contexts[i] || ''))}</span></div>`).join('');
  const notes = row.notes.map(note => `<li>${escapeHtml(note)}</li>`).join('');
  const explanation = EXPLANATIONS[row.decision] ? EXPLANATIONS[row.decision][lang] : row.decision;
  $('details').className = 'details';
  $('details').innerHTML = `
    <h2>${escapeHtml(row.base_unit)}</h2>
    <div class="class">${escapeHtml(term('class_name', row.class_name))} · ${escapeHtml(row.class_name)}</div>
    <div class="detail-grid">
      <div class="detail-box"><div class="k">${escapeHtml(t('decision'))}</div>${rawTerm(term('decision', row.decision), row.decision)}</div>
      <div class="detail-box"><div class="k">${escapeHtml(t('confidence'))}</div>${rawTerm(term('confidence', row.confidence), row.confidence)}</div>
      <div class="detail-box"><div class="k">${escapeHtml(t('acoustic'))}</div>${rawTerm(term('acoustic', row.acoustic_evidence), row.acoustic_evidence)}</div>
      <div class="detail-box"><div class="k">${escapeHtml(t('synthesis'))}</div>${rawTerm(term('synthesis', row.synthesis_evidence), row.synthesis_evidence)}</div>
    </div>
    <div class="section"><h3>${escapeHtml(t('interpretation'))}</h3><div class="interpretation"><p>${escapeHtml(explanation)}</p><p class="caution">${escapeHtml(t('caution'))}</p></div></div>
    <div class="section"><h3>${escapeHtml(t('groups'))}</h3>${groups}</div>
    <div class="section"><h3>${escapeHtml(t('notes'))}</h3><ul class="notes">${notes}</ul></div>`;
}

async function browse() {
  setStatus(t('openingPicker'));
  try {
    const response = await fetch('/api/pick-folder', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({lang})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || t('pickerFailed'));
    if (data.path) $('voicebank').value = data.path;
    setStatus(data.path ? t('selected') : t('canceled'));
  } catch (error) { setStatus(error.message, true); }
}
async function analyze() {
  const path = $('voicebank').value.trim();
  if (!path) { setStatus(t('chooseFirst'), true); return; }
  const button = $('analyze');
  button.disabled = true;
  button.innerHTML = `<span class="spinner"></span> ${escapeHtml(t('analyzing'))}`;
  $('browse').disabled = true;
  $('profile').classList.add('disabled'); $('inventory').classList.add('disabled');
  setStatus(t('running'));
  try {
    const response = await fetch('/api/analyze', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({path})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || t('analysisFailed'));
    current = data;
    renderSummary(data.summary); renderTable(data.rows);
    $('voicebank-name').textContent = data.voicebank;
    $('profile').classList.remove('disabled'); $('inventory').classList.remove('disabled');
    if (data.rows.length) selectRow(data.rows[0].base_unit);
    setStatus(t('summary')(data.summary.onsets, data.summary.synthesis_units));
  } catch (error) { setStatus(error.message, true); }
  finally { button.disabled = false; button.textContent = t('analyze'); $('browse').disabled = false; }
}

$('lang-zh').addEventListener('click', () => { lang = 'zh'; applyLanguage(); });
$('lang-en').addEventListener('click', () => { lang = 'en'; applyLanguage(); });
$('browse').addEventListener('click', browse);
$('analyze').addEventListener('click', analyze);
$('voicebank').addEventListener('keydown', event => { if (event.key === 'Enter') analyze(); });
applyLanguage();
</script>
</body>
</html>
"""
