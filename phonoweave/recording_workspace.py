"""Generate an offline, human-readable recording review workspace.

Reads the checksummed outputs of review_pack_pipeline; never writes to the
source voicebank or modifies review decisions. Reviewer notes in the browser
are separate, non-authoritative notes which the user explicitly exports.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import webbrowser

PACK_SCHEMA = "phonoweave.recording_review_pack.v0.1"
WORKSPACE_SCHEMA = "phonoweave.recording_workspace.v0.1"
ARTIFACTS = (
    "scheme/recording_action_plan.json",
    "scheme/recording_units_review.csv",
    "reference/reference_reclist_blueprint.json",
    "budget/speaker_context_budget_plan.json",
    "line/recording_line_handoff.json",
    "directives/split_decision_directives.json",
)


def _checked_artifact(root: Path, relative: str, digests: dict) -> bytes:
    source = (root / relative).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError(f"review artifact escapes pack: {relative}")
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != digests.get(relative):
        raise ValueError(f"review artifact checksum mismatch: {relative}")
    return raw


def _json_bytes(blob: bytes) -> dict:
    parsed = json.loads(blob.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object in review pack")
    return parsed


def load_workspace(pack: Path) -> dict:
    """Validate source provenance and assemble a minimal, offline UI model."""
    root = pack.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("review pack is not a directory")
    manifest_path = (root / "review_pack_manifest.json").resolve(strict=True)
    if not manifest_path.is_relative_to(root):
        raise ValueError("manifest escapes review pack")
    manifest_raw = manifest_path.read_bytes()
    manifest = _json_bytes(manifest_raw)
    if manifest.get("schema") != PACK_SCHEMA:
        raise ValueError("unsupported recording review pack schema")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("missing review pack checksum manifest")
    files = {name: _checked_artifact(root, name, hashes) for name in ARTIFACTS}
    action, reference, budget, line, directives = (
        _json_bytes(files[name]) for name in ARTIFACTS if name.endswith(".json")
    )
    units = list(csv.DictReader(files[ARTIFACTS[1]].decode("utf-8").splitlines()))
    expected_schema = (
        (action, "phonoweave.recording_scheme_action_plan.v0.1"),
        (reference, "phonoweave.reference_reclist_blueprint.v0.1"),
        (budget, "phonoweave.speaker_context_budget_plan.v0.1"),
        (line, "phonoweave.recording_line_handoff.v0.1"),
        (directives, "phonoweave.split_decision_directives.v0.1"),
    )
    if any(obj.get("schema") != schema for obj, schema in expected_schema):
        raise ValueError("incompatible recording review artifact schemas")
    chosen = budget.get("selection", {}).get("selected_lines")
    lines = line.get("recording_lines")
    pitches = budget.get("pitch_groups")
    if not (isinstance(chosen, list) and isinstance(lines, list)
            and isinstance(pitches, list) and pitches and len(set(pitches)) == len(pitches)
            and len(chosen) == len(lines) == line.get("selected_line_count")
            and chosen == [record.get("sequence") for record in lines]
            and pitches == line.get("pitch_groups") == directives.get("pitch_groups")
            and len(set(chosen)) == len(chosen)):
        raise ValueError("inconsistent recording line selection")
    index = {}
    overview_lines = []
    for record in lines:
        layers = record.get("pitch_layers", [])
        if len(layers) != len(pitches) or {x.get("pitch_group") for x in layers} != set(pitches):
            raise ValueError("recording line has incomplete pitch layers")
        sequence = record["sequence"]
        line_id = record["line_id"]
        if line_id in index:
            raise ValueError("duplicate recording line ID")
        by_pitch = {layer["pitch_group"]: layer for layer in layers}
        for pitch in pitches:
            if len(by_pitch[pitch].get("syllables", [])) != len(sequence.split("_")):
                raise ValueError("recording line has incomplete syllables")
        index[line_id] = record
        overview_lines.append({
            "id": line_id,
            "sequence": sequence,
            "syllables": sequence.split("_"),
            "sources": {pitch: by_pitch[pitch]["source_wav"] for pitch in pitches},
        })
    directive_rows = directives.get("directives", [])
    if (not isinstance(directive_rows, list)
            or len(directive_rows) != directives.get("syllable_pitch_checks")):
        raise ValueError("split directives are inconsistent")
    seen_cells = set()
    thin_directives = []
    naming = {}
    linked_candidates = 0
    for row in directive_rows:
        line_id = row.get("line_id")
        pitch = row.get("pitch_group")
        offset = row.get("syllable_index")
        cell_id = (line_id, pitch, offset)
        if (line_id not in index or pitch not in pitches
                or not isinstance(offset, int) or isinstance(offset, bool)
                or offset not in range(len(index[line_id]["sequence"].split("_")))
                or cell_id in seen_cells):
            raise ValueError("split directive position has invalid or duplicate identity")
        seen_cells.add(cell_id)
        word = index[line_id]["sequence"].split("_")[offset]
        original = next(layer for layer in index[line_id]["pitch_layers"]
                        if layer["pitch_group"] == pitch)
        if (row.get("syllable") != word or
                row.get("source_wav") != original["source_wav"]):
            raise ValueError("split directive has mismatched WAV or syllable")
        candidates = row.get("candidate_realizations", [])
        if not isinstance(candidates, list):
            raise ValueError("candidate realizations must be a list")
        for candidate in candidates:
            if candidate.get("evidence_kind") == "source_linked_internal_VC_candidate":
                linked_candidates += 1
                name = candidate.get("unit_id", "")
                context = candidate.get("context", "")
                key = (name, context)
                if key not in naming:
                    naming[key] = {"unit_id": name, "context": context,
                                   "count": 0, "pitches": set(), "examples": []}
                group = naming[key]
                group["count"] += 1
                group["pitches"].add(pitch)
                if len(group["examples"]) < 8:
                    group["examples"].append({
                        "pitch": pitch, "wav": row["source_wav"],
                        "original_alias": candidate.get("original_alias"),
                        "source_oto": candidate.get("source_oto"),
                        "oto_line_number": candidate.get("oto_line_number"),
                    })
        thin_directives.append({
            "line_id": line_id, "pitch_group": pitch, "syllable_index": offset,
            "syllable": word, "review_status": row.get("review_status"),
            "evidence_kind": row.get("evidence_kind"),
            "annotator_action": row.get("annotator_action"),
            "tuner_action": row.get("tuner_action"),
            "source_oto_references": row.get("source_oto_references", []),
            "candidate_realizations": candidates,
        })
    if linked_candidates != directives.get("source_linked_internal_candidates"):
        raise ValueError("source linked candidate count mismatch")
    tasks = action.get("tasks", [])
    if (not isinstance(tasks, list) or len(tasks) != action.get("consolidated_tasks")
            or len({t.get("task_id") for t in tasks}) != len(tasks)):
        raise ValueError("action plan task count mismatch")
    if not isinstance(units, list) or not units:
        raise ValueError("empty synthesis inventory")
    return {
        "schema": WORKSPACE_SCHEMA,
        "source_pack_sha256": sha256(manifest_raw).hexdigest(),
        "speaker_id": action.get("speaker_id", ""),
        "pitches": pitches,
        "summary": {
            "reference_lines": reference["all_pitch_reference_sequences"],
            "selected_lines": len(lines),
            "source_cells": len(thin_directives),
            "linked_candidates": linked_candidates,
            "review_tasks": len(tasks),
            "naming_groups": len(naming),
            "production_ready": False,
        },
        "lines": overview_lines,
        "directives": thin_directives,
        "units": units,
        "tasks": tasks,
        "naming_groups": [
            {**group, "pitches": sorted(group["pitches"])}
            for _key, group in sorted(naming.items())
        ],
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PhonoWeave · Recording Workspace</title>
<style>
:root{color-scheme:dark;--bg:#101521;--panel:#1b2433;--panel2:#202b3d;--line:#334156;--text:#edf4f9;--muted:#a7b5c8;--cyan:#70e3d4;--purple:#bda9f6;--amber:#f4cc85;--rose:#f2a9b7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Noto Sans SC",system-ui,sans-serif}button,input,select,textarea{font:inherit}button{cursor:pointer}button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:2px solid var(--cyan);outline-offset:2px}
.shell{display:grid;grid-template-columns:240px minmax(0,1fr);min-height:100vh}.side{position:sticky;top:0;height:100vh;display:flex;flex-direction:column;gap:32px;background:#141c2a;border-right:1px solid #2b394c;padding:30px 18px}.logo{display:flex;gap:12px;align-items:center;padding:0 8px}.mark{width:42px;height:42px;border-radius:15px 5px 16px 6px;display:grid;place-items:center;font-size:23px;font-weight:800;letter-spacing:-3px;background:linear-gradient(145deg,#80e9d9,#9e9aff);color:#142335}.brand strong{display:block;font-size:18px;letter-spacing:-.6px}.brand small{color:var(--muted);font-size:11px;letter-spacing:1.1px}.nav{display:grid;gap:8px}.nav button{display:flex;align-items:center;gap:12px;text-align:left;width:100%;background:none;color:var(--muted);border:1px solid transparent;border-radius:12px;padding:13px 16px}.nav button.active{background:#263649;color:#f8ffff;border-color:#365268}.nav button:hover{background:#222e40}.nav .sym{font-size:16px;width:21px;text-align:center}.side-foot{margin-top:auto;padding:16px 11px;border-top:1px solid #2e3949;color:var(--muted);font-size:12px;line-height:1.6}.online{width:8px;height:8px;background:var(--cyan);display:inline-block;border-radius:50%;margin-right:7px;box-shadow:0 0 12px #70e3d460}
.main{padding:37px clamp(22px,4vw,70px) 60px;max-width:1700px;width:100%;margin:auto}.top{display:flex;gap:20px;justify-content:space-between;align-items:start;margin-bottom:32px}.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:2.4px;color:var(--cyan);font-weight:750}.top h1{font-size:clamp(25px,2.5vw,36px);letter-spacing:-1px;font-weight:760;margin:8px 0}.top p,.subtle{color:var(--muted);line-height:1.65;font-size:13px}.top p{margin:0}.toolbar{display:flex;gap:8px;flex-wrap:wrap}.btn{border:1px solid #405369;border-radius:12px;padding:10px 14px;background:#222e3e;color:#e5f1f9;white-space:nowrap}.btn:hover{border-color:#7ccdc6}.btn.primary{color:#12232a;background:var(--cyan);border-color:var(--cyan);font-weight:750}.btn:disabled{opacity:.45;cursor:not-allowed}.tag{display:inline-flex;align-items:center;gap:5px;background:#223749;border:1px solid #345064;color:#a4e8dd;border-radius:999px;padding:5px 10px;font-size:11px;font-weight:700;white-space:nowrap}.tag.warn{background:#3e342b;border-color:#715d41;color:#f1ce97}.tag.rose{background:#392d38;border-color:#714b5a;color:#f1adba}.tag.neutral{background:#293242;border-color:#485366;color:#c2ceda}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px;margin-bottom:26px}.stat{background:linear-gradient(150deg,#212c3e,#1b2636);border:1px solid #344156;padding:20px 22px;border-radius:18px;position:relative;overflow:hidden}.stat:after{content:"";position:absolute;top:-30px;right:-30px;width:95px;height:95px;border:1px solid #5acfb71a;border-radius:50%}.stat .cap{color:var(--muted);font-size:12px}.stat b{display:block;font-size:28px;letter-spacing:-1px;margin:8px 0 4px}.stat small{color:#7fd7cc;font-size:11px}.section-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:0 0 18px}.section-head h2{font-size:19px;margin:0;letter-spacing:-.4px}.panel{border:1px solid var(--line);border-radius:18px;background:var(--panel);padding:22px}.grid2{display:grid;grid-template-columns:minmax(250px,0.85fr) minmax(0,1.5fr);gap:18px}.grid2 > *{min-width:0}.itemlist{max-height:580px;overflow:auto;display:grid;gap:7px;padding-right:3px}.item{width:100%;padding:12px 14px;text-align:left;border:1px solid #334355;border-radius:11px;background:#222d3e;color:var(--text)}.item.active{border-color:#82dcd1;background:#293d4e}.item .item-title{display:block;font-weight:680;font-size:13px;overflow-wrap:anywhere}.item .item-desc{display:block;color:var(--muted);font-size:11px;margin-top:5px;line-height:1.5}.search{background:#111b29;border:1px solid #3b4c62;border-radius:11px;color:var(--text);padding:11px 13px;width:100%;margin-bottom:13px}.select{padding:9px 12px;border-radius:11px;background:#202d3e;color:var(--text);border:1px solid #3c5167}.pitchbar{display:flex;gap:8px;flex-wrap:wrap;margin:15px 0}.chip{padding:8px 11px;background:#202e41;border:1px solid #43536a;border-radius:10px;color:#d0deee}.chip.active{border-color:var(--cyan);color:var(--cyan);background:#243e48}.tokens{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.token{border:1px solid #455166;color:#dce9f4;border-radius:10px;background:#223044;min-width:50px;padding:11px 9px;text-align:center}.token small{display:block;font-size:9px;color:var(--muted);margin-top:4px}.token.active{border-color:var(--cyan);background:#2b4c4d}.token.linked{box-shadow:inset 0 -3px 0 var(--purple)}.token.provisional{box-shadow:inset 0 -3px 0 var(--amber)}.detail{padding:17px;background:#141e2c;border-radius:13px;border:1px solid #324356;margin-top:14px}.detail strong{font-size:15px}.detail .line{margin:12px 0;color:#aabace;font-size:12px;line-height:1.7;overflow-wrap:anywhere}.detail code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-wrap:anywhere}.detail code{color:#a4e9df}.splitlist{display:grid;gap:8px}.kv{display:grid;grid-template-columns:110px minmax(0,1fr);gap:8px;margin:9px 0;font-size:12px}.kv .key{color:var(--muted)}.kv .val{overflow-wrap:anywhere}.note{display:block;min-height:100px;width:100%;margin:12px 0;color:var(--text);background:#111b29;border:1px solid #3c5066;padding:12px;border-radius:12px;resize:vertical}.divider{height:1px;background:#334256;margin:17px 0}.notice{display:flex;gap:11px;align-items:start;border:1px solid #4e614e;background:#25352e;border-radius:14px;padding:14px 17px;margin-bottom:20px;color:#b6e8ce;font-size:12px;line-height:1.6}.notice.warn{background:#322d2c;border-color:#645247;color:#f4d2a4}.hidden{display:none!important}.foot{padding-top:40px;color:#77899b;font-size:11px}.capacity{height:9px;background:#121c2a;border-radius:99px;overflow:hidden;margin:14px 0}.capacity span{display:block;background:linear-gradient(90deg,#77d9d0,#a7a4f8);height:100%;border-radius:99px}.tiny{font-size:11px;color:var(--muted)}.source-note{padding:10px 12px;background:#202e3d;border:1px solid #3a5065;border-radius:10px;margin:8px 0;font-size:12px;line-height:1.7;overflow-wrap:anywhere}.empty{text-align:center;padding:35px 10px;color:var(--muted);font-size:13px}.symbol{color:var(--cyan)}.status{margin-left:8px}
@media(max-width:950px){.shell{grid-template-columns:1fr}.side{height:auto;position:static;gap:15px;padding:16px}.side-foot{display:none}.nav{display:flex;gap:5px;overflow:auto}.nav button{width:auto;flex-shrink:0;padding:9px}.main{padding:25px 18px}.grid2{grid-template-columns:1fr}.itemlist{max-height:260px}}@media(max-width:650px){.stats{grid-template-columns:repeat(2,1fr)}.top{flex-direction:column}.panel{padding:15px}.stat{padding:14px}}
</style>
</head>
<body>
<div class="shell"><aside class="side"><div class="logo"><div class="mark">P</div><div class="brand"><strong>PhonoWeave</strong><small>RECORDING STUDIO</small></div></div><nav class="nav" aria-label="工作台导航">
<button data-page="overview" class="active"><span class="sym">◈</span> 项目概览</button><button data-page="recordings"><span class="sym">♫</span> 录音方案</button><button data-page="units"><span class="sym">⌘</span> 拆音单元</button><button data-page="reviews"><span class="sym">▤</span> 标注与调教</button></nav><div class="side-foot"><span class="online"></span>离线工作台 · 仅本地数据<br/>所有生产级 OTO / 路由写入均禁用。</div></aside>
<main class="main"><header class="top"><div><span class="eyebrow">SPEAKER-ADAPTIVE WORKSPACE</span><h1 id="page-title">录音方案工作台</h1><p id="page-subtitle">查看来源、比较拆音候选、整理制作审核事项。</p></div><div class="toolbar"><label class="btn" for="import-notes">导入备注</label><input type="file" id="import-notes" accept=".json,application/json" hidden/><button class="btn primary" id="export-notes">导出审核备注 ↓</button></div></header>
<div class="notice"><span>◇</span><span><strong>当前为审核模式。</strong> 拆分建议和内部 CV 关联属于候选证据，不代表已确认声学边界。工作台不生成或写入新的 OTO、alias 和 phonemizer 路由；备注仅用于追踪检查进度。</span></div>
<section id="overview-page"><div class="stats" id="stats"></div><div class="grid2"><section class="panel"><div class="section-head"><h2>录音预算</h2><span class="tag neutral">PILOT</span></div><div id="budget-progress"></div><div class="divider"></div><h2 style="font-size:16px">当前声库</h2><p class="subtle" id="speaker-summary"></p></section><section class="panel"><div class="section-head"><h2>待解决的制作工作</h2><span class="tag warn">人工审核</span></div><div id="overview-actions"></div></section></div></section>
<section id="recordings-page" class="hidden"><div class="section-head"><h2>录音序列 · 逐音节来源</h2><span class="tag neutral" id="line-count"></span></div><div class="grid2"><div class="panel"><input class="search" id="line-search" placeholder="搜索音节序列、RL 编号…" aria-label="搜索录音行"/><div class="itemlist" id="line-list"></div></div><div class="panel" id="line-detail"></div></div></section>
<section id="units-page" class="hidden"><div class="section-head"><h2>拆音上下文 · 命名组</h2><span class="tag warn" id="group-count"></span></div><div class="grid2"><div class="panel"><input class="search" id="unit-search" placeholder="搜索单元、辅音、上下文…"/><div class="itemlist" id="unit-list"></div></div><div class="panel" id="unit-detail"></div></div></section>
<section id="reviews-page" class="hidden"><div class="section-head"><h2>标注师 & 调教师 · 审核台</h2><span class="tag warn" id="task-count"></span></div><div class="grid2"><div class="panel"><input class="search" id="task-search" placeholder="搜索审核任务或相关音节…"/><div class="itemlist" id="task-list"></div></div><div class="panel" id="task-detail"></div></div></section>
<div class="foot">PhonoWeave · Offline Review Workspace v0.1 · 无外部资源请求 · <span id="pack-id"></span></div></main></div>
<script>const DATA=__PW_WORKSPACE_DATA__;</script>
<script>
"use strict";
const $=id=>document.getElementById(id);
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
const attr=esc;
const lineById=new Map(DATA.lines.map(r=>[r.id,r]));
const cells=new Map(DATA.directives.map(r=>[`${r.line_id}|${r.pitch_group}|${r.syllable_index}`,r]));
const taskById=new Map(DATA.tasks.map(t=>[t.task_id,t]));
let activePage="overview", activeLine=DATA.lines[0]?.id, activePitch=DATA.pitches[0], activeToken=0, activeUnit=DATA.units[0]?.unit_id, activeTask=DATA.tasks[0]?.task_id;
let notes={};
function statusTag(s){if(!s)return '<span class="tag neutral">待检查</span>';if(s.includes('provisional')||s.includes('split'))return '<span class="tag warn">暂定拆分</span>';if(s.includes('ambiguous')||s.includes('unresolved'))return '<span class="tag rose">待判定</span>';if(s.includes('linked')||s.includes('candidate'))return '<span class="tag neutral">有来源候选</span>';return '<span class="tag neutral">需审核</span>';}
function setPage(page){activePage=page;document.querySelectorAll('[data-page]').forEach(b=>b.classList.toggle('active',b.dataset.page===page));for(const p of ['overview','recordings','units','reviews'])$(p+'-page').classList.toggle('hidden',p!==page);const name={overview:'录音方案工作台',recordings:'录音方案 · Recording Lines',units:'上下文拆音 · Split Units',reviews:'标注与调教 · Review Desk'}; $('page-title').textContent=name[page];if(page==='recordings'){drawLineList();drawLineDetail()}if(page==='units'){drawUnitList();drawUnitDetail()}if(page==='reviews'){drawTaskList();drawTaskDetail()}}
document.querySelectorAll('[data-page]').forEach(b=>b.addEventListener('click',()=>setPage(b.dataset.page)));
function drawOverview(){const s=DATA.summary;const values=[['参考录音序列',s.reference_lines,'现有声库结构'],['预算选中序列',s.selected_lines,'每音高的候选录音行'],['逐音节检查',s.source_cells,'跨音高的审核位置'],['关联内部候选',s.linked_candidates,'有原始 OTO 来源']];$('stats').innerHTML=values.map(x=>`<div class="stat"><span class="cap">${esc(x[0])}</span><b>${x[1]}</b><small>${esc(x[2])}</small></div>`).join('');const ratio=s.reference_lines?Math.round(100*s.selected_lines/s.reference_lines):0;$('budget-progress').innerHTML=`<div style="display:flex;justify-content:space-between;align-items:end"><div><strong style="font-size:26px">${s.selected_lines} <span style="font-size:14px;color:var(--muted)">/ ${s.reference_lines}</span></strong><div class="subtle">被选入候选录音表的参考序列</div></div><span class="tag">${ratio}%</span></div><div class="capacity"><span style="width:${ratio}%"></span></div><div class="subtle">${esc(DATA.pitches.join(' · '))}。预算优化尚未通过合成质量验证。</div>`;$('speaker-summary').textContent=`本地项目：${DATA.speaker_id||'未命名'}。受保护上下文：${DATA.naming_groups.length} 个有直接 OTO 来源的命名组；${DATA.tasks.length} 项人工审核任务。`;
const counts={};for(const t of DATA.tasks){counts[t.review_action]=(counts[t.review_action]||0)+1;}const names={'review_internal_candidate_mapping':'内部过渡关联','review_provisional_unit_decision':'拆分／合并决策','review_context_mapping':'直接上下文映射','review_role_scoped_mapping':'未决内部角色','review_cross_pitch_coverage':'跨音高 alias 覆盖'};$('overview-actions').innerHTML=Object.entries(counts).sort((a,b)=>b[1]-a[1]).map(([key,value])=>`<div class="kv"><span class="key">${esc(names[key]||key)}</span><span class="val">${value} 项 · 待人工判断</span></div>`).join('')+'<div class="divider"></div><button class="btn" id="go-reviews">查看审核任务 →</button>';$('go-reviews').onclick=()=>setPage('reviews');$('pack-id').textContent='Pack SHA256: '+DATA.source_pack_sha256.slice(0,12);$('line-count').textContent=DATA.lines.length+' 条录音行';$('group-count').textContent=DATA.units.length+' 个单元 · '+DATA.naming_groups.length+' 个有来源组';$('task-count').textContent=DATA.tasks.length+' 项任务';}
function drawLineList(){const q=$('line-search').value.trim().toLowerCase();const arr=DATA.lines.filter(r=>(r.id+' '+r.sequence).toLowerCase().includes(q));$('line-list').innerHTML=arr.map(r=>`<button class="item ${r.id===activeLine?'active':''}" data-line="${attr(r.id)}"><span class="item-title">${esc(r.id)}　${esc(r.sequence)}</span><span class="item-desc">${r.syllables.length} 音节 · ${DATA.pitches.length} 个音高层</span></button>`).join('')||'<div class="empty">没有找到匹配的录音序列</div>';document.querySelectorAll('[data-line]').forEach(b=>b.onclick=()=>{activeLine=b.dataset.line;activeToken=0;drawLineList();drawLineDetail()});}
function drawLineDetail(){const row=lineById.get(activeLine);if(!row){$('line-detail').innerHTML='<p>无录音行</p>';return}const p=activePitch;const cell=cells.get(`${row.id}|${p}|${activeToken}`);$('line-detail').innerHTML=`<span class="eyebrow">${esc(row.id)} · SELECTED LINE</span><h2 style="overflow-wrap:anywhere">${esc(row.sequence)}</h2><div class="subtle">选择一个音高与音节，查看标注和调教检查指令。</div><div class="pitchbar">${DATA.pitches.map(x=>`<button class="chip ${x===p?'active':''}" data-pitch="${attr(x)}">${esc(x)}</button>`).join('')}</div><div class="tiny mono">SOURCE WAV · ${esc(row.sources[p]||'')}</div><div class="tokens">${row.syllables.map((x,i)=>{const c=cells.get(`${row.id}|${p}|${i}`);const cls=c?.candidate_realizations?.some(y=>y.evidence_kind==='source_linked_internal_VC_candidate')?'linked':c?.review_status?.includes('provisional')?'provisional':'';return `<button class="token ${cls} ${i===activeToken?'active':''}" data-token="${i}">${esc(x)}<small>${i+1}</small></button>`}).join('')}</div><div id="token-inspect"></div>`;document.querySelectorAll('[data-pitch]').forEach(b=>b.onclick=()=>{activePitch=b.dataset.pitch;drawLineDetail()});document.querySelectorAll('[data-token]').forEach(b=>b.onclick=()=>{activeToken=Number(b.dataset.token);drawLineDetail()});drawTokenInspect(cell,row);}
function drawTokenInspect(c,row){if(!c){$('token-inspect').innerHTML='<div class="empty">无此位置的审核记录</div>';return}const refs=(c.candidate_realizations||[]).filter(x=>x.original_alias);$('token-inspect').innerHTML=`<div class="detail"><div style="display:flex;justify-content:space-between;gap:9px;align-items:center"><strong>音节 ${activeToken+1} · ${esc(c.syllable)}</strong>${statusTag(c.review_status)}</div><div class="kv"><span class="key">证据类别</span><span class="val mono">${esc(c.evidence_kind)}</span></div><div class="kv"><span class="key">标注师任务</span><span class="val">${esc(c.annotator_action)}</span></div><div class="kv"><span class="key">调教师任务</span><span class="val">${esc(c.tuner_action)}</span></div><div class="divider"></div><strong>原始来源证据</strong>${refs.length?refs.map(r=>`<div class="source-note"><span class="tag neutral">${esc(r.unit_id)}</span> ${esc(r.context)}<br/>原 alias：<code>${esc(r.original_alias)}</code><br/>OTO：<code>${esc(r.source_oto)}:${esc(r.oto_line_number)}</code><br/>映射状态：<span style="color:var(--amber)">未批准</span></div>`).join(''):'<p class="subtle">此位置暂无直接 OTO 候选关联。这不代表录音缺失。</p>'}</div>`;}
$('line-search').addEventListener('input',drawLineList);
function drawUnitList(){const q=$('unit-search').value.trim().toLowerCase();const rows=DATA.units.filter(u=>(u.unit_id+' '+u.base_unit+' '+u.contexts).toLowerCase().includes(q));$('unit-list').innerHTML=rows.map(u=>{const groups=DATA.naming_groups.filter(g=>g.unit_id===u.unit_id);const count=groups.reduce((sum,g)=>sum+g.count,0);return `<button class="item ${u.unit_id===activeUnit?'active':''}" data-unit="${attr(u.unit_id)}"><span class="item-title">${esc(u.unit_id)}</span><span class="item-desc">${esc(u.contexts||'未指定上下文')} · ${count?count+' 条候选':'无当前内部关联'} · ${esc(u.decision||'待审核')}</span></button>`}).join('')||'<div class="empty">没有匹配的上下文单元</div>';document.querySelectorAll('[data-unit]').forEach(b=>b.onclick=()=>{activeUnit=b.dataset.unit;drawUnitList();drawUnitDetail()})}
function drawUnitDetail(){const unit=DATA.units.find(u=>u.unit_id===activeUnit);if(!unit){$('unit-detail').innerHTML='<div class="empty">没有拆音单元</div>';return}const groups=DATA.naming_groups.filter(g=>g.unit_id===activeUnit);const total=groups.reduce((sum,g)=>sum+g.count,0);const examples=groups.flatMap(g=>g.examples.slice(0,4));$('unit-detail').innerHTML=`<span class="eyebrow">CONTEXT REALIZATION</span><h2>${esc(unit.unit_id)}</h2><div class="pitchbar"><span class="tag warn">${esc(unit.decision||'待审核')}</span><span class="tag neutral">${total} 条原始内部候选</span></div><div class="detail"><div class="kv"><span class="key">辅音</span><span class="val mono">${esc(unit.base_unit||'')}</span></div><div class="kv"><span class="key">上下文</span><span class="val">${esc(unit.contexts||'未指定')}</span></div><div class="kv"><span class="key">置信度</span><span class="val">${esc(unit.confidence||'未提供')}</span></div><div class="kv"><span class="key">关联组</span><span class="val">${esc(groups.map(g=>g.context).join(' · ')||'无直接内部来源')}</span></div></div><h3>代表性原始 OTO 来源</h3>${examples.length?examples.map(x=>`<div class="source-note"><span class="tag neutral">${esc(x.pitch)}</span> <code>${esc(x.original_alias)}</code><br/><span class="subtle">${esc(x.wav)}</span><br/><span class="subtle mono">${esc(x.source_oto)}:${esc(x.oto_line_number)}</span></div>`).join(''):'<p class="subtle">当前预算选中的录音行没有该单元的直接内部 VC 关联；这不代表整个声库缺少对应录音，也不代表允许合并。</p>'}<div class="notice warn" style="margin-top:20px">来源可追溯不等于存在独立可用的 CV 样本；任何新命名都必须再经过声学与标注师审核。</div>`;}
$('unit-search').addEventListener('input',drawUnitList);
function drawTaskList(){const q=$('task-search').value.trim().toLowerCase();const arr=DATA.tasks.filter(t=>[t.task_id,t.unit_id,t.review_action,(t.suggested_syllables||[]).join(' ')].join(' ').toLowerCase().includes(q));$('task-list').innerHTML=arr.map(t=>`<button class="item ${t.task_id===activeTask?'active':''}" data-task="${attr(t.task_id)}"><span class="item-title">${esc(t.task_id)} · ${esc(t.unit_id)}</span><span class="item-desc">${esc(t.review_action)} · ${esc((t.pitch_groups||[]).join(' / '))}${notes[t.task_id]?.status?' · '+esc(notes[t.task_id].status):''}</span></button>`).join('')||'<div class="empty">没有匹配的审核任务</div>';document.querySelectorAll('[data-task]').forEach(b=>b.onclick=()=>{activeTask=b.dataset.task;drawTaskList();drawTaskDetail()})}
function drawTaskDetail(){const t=taskById.get(activeTask);if(!t){$('task-detail').innerHTML='<div class="empty">暂无审核任务</div>';return}const current=notes[t.task_id]||{status:'未检查',note:''};const examples=t.source_examples||[];$('task-detail').innerHTML=`<span class="eyebrow">REVIEW ITEM · ${esc(t.task_id)}</span><h2>${esc(t.unit_id)}</h2><div class="pitchbar"><span class="tag warn">尚未批准生产配置</span><span class="tag neutral">优先级 ${esc(t.priority??'?')}</span></div><div class="detail"><div class="kv"><span class="key">审核类型</span><span class="val mono">${esc(t.review_action)}</span></div><div class="kv"><span class="key">涉及音高</span><span class="val">${esc((t.pitch_groups||[]).join(' · '))}</span></div><div class="kv"><span class="key">相关音节</span><span class="val">${esc((t.suggested_syllables||[]).join(' · ')||'未指定')}</span></div><div class="divider"></div><strong>核验任务</strong><p class="subtle">${esc(t.verification||'核查原始音频及 OTO 关联')}</p>${examples.length?'<strong>原始来源</strong>'+examples.slice(0,8).map(x=>`<div class="source-note">${esc(x.pitch_group)} · <code>${esc(x.original_alias)}</code><br/>${esc(x.source_wav)}<br/>${esc(x.source_oto)}:${esc(x.line_number)}</div>`).join(''):'<p class="subtle">无直接来源示例：必须依照核验步骤确认。</p>'}</div><div class="divider"></div><strong>我的审核备注 <span class="tiny">（不会改变原始审核状态）</span></strong><div class="pitchbar"><select class="select" id="review-status"><option value="未检查">未检查</option><option value="已查看">已查看</option><option value="需进一步听测">需进一步听测</option><option value="存在问题">存在问题</option></select></div><textarea class="note" id="review-note" placeholder="记录听测发现、标注疑问、下一步处理…"></textarea><button class="btn" id="save-note">保存本次备注</button><p class="tiny">备注保存在当前页面内存。点击右上角“导出审核备注”才能保留到文件。</p>`;$('review-status').value=current.status;$('review-note').value=current.note;$('save-note').onclick=()=>{notes[t.task_id]={status:$('review-status').value,note:$('review-note').value};drawTaskList();drawTaskDetail();showMessage('本次备注已记录。记得导出 JSON 保存。')};}
$('task-search').addEventListener('input',drawTaskList);
function showMessage(msg){const b=document.createElement('div');b.textContent=msg;b.style.cssText='position:fixed;bottom:22px;right:22px;z-index:999;background:#294e51;color:#e5fffc;padding:13px 18px;border:1px solid #77d9d0;border-radius:12px;max-width:330px;box-shadow:0 8px 30px #0008';document.body.append(b);setTimeout(()=>b.remove(),3300)}
$('export-notes').onclick=()=>{if(activePage==='reviews'&&activeTask&&$('review-note'))notes[activeTask]={status:$('review-status').value,note:$('review-note').value};const data={schema:'phonoweave.workspace_notes.v0.1',source_pack_sha256:DATA.source_pack_sha256,notes};const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='phonoweave_review_notes_'+DATA.source_pack_sha256.slice(0,8)+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),3000);showMessage('已导出本地审核备注 JSON（不是生产审核批准）')};
$('import-notes').onchange=async e=>{const file=e.target.files?.[0];if(!file)return;try{const obj=JSON.parse(await file.text());if(obj.schema!=='phonoweave.workspace_notes.v0.1'||obj.source_pack_sha256!==DATA.source_pack_sha256||!obj.notes||typeof obj.notes!=='object'||Array.isArray(obj.notes))throw Error('文件与当前审核包不匹配');for(const [id,n] of Object.entries(obj.notes)){if(!taskById.has(id)||typeof n.note!=='string'||!['未检查','已查看','需进一步听测','存在问题'].includes(n.status))throw Error('包含无效任务或备注');}notes=obj.notes;drawTaskList();drawTaskDetail();showMessage('已导入本地审核备注')}catch(err){showMessage('导入失败：'+err.message)}finally{e.target.value=''}};
drawOverview();setPage('overview');
</script>
</body>
</html>'''


def render_workspace(data: dict) -> str:
    if data.get("schema") != WORKSPACE_SCHEMA:
        raise ValueError("unexpected workspace schema")
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).replace("<", "\\u003c").replace(
                             ">", "\\u003e").replace("\u2028", "\\u2028")
    return HTML_TEMPLATE.replace("__PW_WORKSPACE_DATA__", encoded)


def write_workspace(pack: Path, output: Path) -> Path:
    root = pack.expanduser().resolve(strict=True)
    destination = output.expanduser().resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("workspace output must be outside source review pack")
    if destination.exists():
        raise FileExistsError("workspace HTML already exists; choose a new filename")
    result = render_workspace(load_workspace(root))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as f:
        f.write(result)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an offline recording review workspace")
    parser.add_argument("--pack", type=Path, required=True,
                        help="completed review_pack_pipeline output directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="new standalone .html file (never overwritten)")
    parser.add_argument("--open", action="store_true",
                        help="open the local HTML in your default browser")
    args = parser.parse_args(argv)
    try:
        result = write_workspace(args.pack, args.output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print("Recording workspace:", result)
    print("Source voicebank and OTO unchanged; offline review only")
    if args.open:
        webbrowser.open(result.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
