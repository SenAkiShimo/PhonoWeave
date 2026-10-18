"""Traceable annotator and tuner handoff for provisional split decisions.

No new OTO, alias or audio is generated. Both manuals derive from one
review-only machine-readable manifest tied to source plan identifiers.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile

PLAN_SCHEMA = "phonoweave.speaker_context_budget_plan.v0.1"
ACTION_SCHEMA = "phonoweave.recording_scheme_action_plan.v0.1"
SOURCE_SCHEMA = "phonoweave.lossless_alias_record.v0.1"
SCHEMA = "phonoweave.split_handoff.v0.1"


def _safe(value) -> str:
    return str(value).replace("\n", " ").replace("\r", " ").replace("|", "/")


def compile_split_handoff(budget: dict, units: list[dict],
                          actions: dict, sources: list[dict]) -> dict:
    if budget.get("schema") != PLAN_SCHEMA or actions.get("schema") != ACTION_SCHEMA:
        raise ValueError("incompatible source plan schema")
    pitch_groups = budget.get("pitch_groups", [])
    selection = budget.get("selection", {}).get("selected_lines", [])
    if (not pitch_groups or len(set(pitch_groups)) != len(pitch_groups)
            or not isinstance(selection, list) or len(set(selection)) != len(selection)
            or len(selection) != budget["selection"]["selected_line_count"]):
        raise ValueError("invalid budget selection")
    unit_map = {}
    for row in units:
        uid = row.get("unit_id")
        if not uid or uid in unit_map or not row.get("base_unit"):
            raise ValueError("invalid or duplicate inventory unit")
        unit_map[uid] = row
    protected = set(budget["protected_units"])
    abstained = set(budget["unresolved_or_not_proxy_supported_units_not_optimized"])
    if not protected <= unit_map.keys() or not abstained <= unit_map.keys() or protected & abstained:
        raise ValueError("budget references unknown or conflicting units")
    if any(unit_map[u].get("decision") != "split_recommended" for u in protected):
        raise ValueError("split recommendation differs from inventory")
    by_task = defaultdict(list)
    task_ids = set()
    for task in actions.get("tasks", []):
        uid, tid = task["unit_id"], task["task_id"]
        if uid not in unit_map or tid in task_ids:
            raise ValueError("unknown unit or duplicate review task")
        task_ids.add(tid)
        by_task[uid].append(task)
    by_source = defaultdict(list)
    identifiers = set()
    for record in sources:
        if record.get("schema") != SOURCE_SCHEMA:
            raise ValueError("unexpected source alias record schema")
        src, role, compilation = (record["source"], record["role"],
                                  record["compilation_preview"])
        fields = ("source_oto", "line_number", "original_alias")
        if any(src.get(k) != item.get(k) for item in (role, compilation) for k in fields):
            raise ValueError("source alias provenance mismatch")
        ident = tuple(src[k] for k in fields)
        if ident in identifiers:
            raise ValueError("duplicate OTO source identity")
        identifiers.add(ident)
        uid, pitch = compilation.get("candidate_unit_id"), role.get("pitch_suffix")
        if (uid in unit_map and pitch in pitch_groups and
                compilation.get("compile_status") == "roundtrip_review_only" and
                role.get("role") in {"initial_cv", "standalone_cv"} and
                role.get("pitch_suffix_status") == "matched"):
            by_source[(uid, pitch)].append({
                "original_alias": src["original_alias"],
                "source_oto": src["source_oto"],
                "line_number": src["line_number"],
                "source_wav": src["source_wav"],
                "role": role["role"],
                "selected_recording_line": any(
                    src["source_wav"].endswith("/_" + line + pitch + ".wav")
                    for line in selection),
                "original_oto_timing_reference": {
                    key: record.get("original_oto_timing", {}).get(key)
                    for key in ("offset_ms", "consonant_ms", "cutoff_ms",
                                "preutterance_ms", "overlap_ms")
                },
            })
    rows = []
    for uid, unit in sorted(unit_map.items(), key=lambda x: (x[1]["base_unit"], x[0])):
        status = ("provisional_split_review" if uid in protected else
                  "unresolved_do_not_merge" if uid in abstained else
                  "outside_optimized_scope")
        pitches = []
        for pitch in pitch_groups:
            direct = sorted(by_source[(uid, pitch)], key=lambda x: (
                not x["selected_recording_line"], x["source_oto"], x["line_number"]))[:3]
            transitions = [{
                key: sample.get(key) for key in (
                    "source_oto", "line_number", "source_wav", "original_alias",
                    "syllable", "review_status")
            } for task in by_task.get(uid, []) for sample in task.get("source_examples", [])
                if sample.get("pitch_group") == pitch][:3]
            pitches.append({
                "pitch_group": pitch, "original_cv_examples": direct,
                "internal_transition_candidates": transitions,
                "new_oto_timing": None, "new_alias_mapping": None,
                "annotation_status": "pending_manual_review",
            })
        rows.append({
            "unit_id": uid, "base_unit": unit["base_unit"],
            "contexts": unit.get("contexts", "").split("|"),
            "profile_decision": unit.get("decision", ""),
            "confidence": unit.get("confidence", ""),
            "acoustic_evidence": unit.get("acoustic_evidence", ""),
            "synthesis_evidence": unit.get("synthesis_evidence", ""),
            "status": status,
            "review_task_ids": [task["task_id"] for task in by_task.get(uid, [])],
            "pitch_evidence": pitches,
            "annotator_approval": "pending", "tuner_approval": "pending",
        })
    return {
        "schema": SCHEMA,
        "speaker_id": actions.get("speaker_id", ""),
        "pitch_groups": pitch_groups,
        "selected_recording_lines": selection,
        "unit_count": len(rows),
        "provisional_split_units": len(protected),
        "units_excluded_from_optimization": len(abstained),
        "units": rows,
        "production_ready": False,
        "new_oto_timing_authorized": False,
        "new_alias_mapping_authorized": False,
    }


def annotator_guide(data: dict) -> str:
    out = [
        "# PhonoWeave — 拆音与 OTO 标注方案（标注师）", "",
        "状态：候选审核稿。任何拆分均未获生产级批准，源 WAV/OTO 不得覆盖。", "",
        "## 标注原则", "",
        "1. 先核对音高层、WAV、实际发声音节及录音顺序，再划分声学边界。",
        "2. 区分起始 CV、内部 CV、VC 过渡、VV 过渡和独立元音；不能仅凭拼写互相代替。",
        "3. 使用实际波形、频谱和听感核实声母起点、元音起点与过渡段。",
        "4. offset、consonant、cutoff、preutterance、overlap 按实际录音标注；",
        "   原 OTO 数值仅可作为原始来源参考，不得照抄至新录音。",
        "5. 新 alias 必须先进行上下文语义审核、同名冲突检查，再进入确认流程。",
        "6. 证据不足的单元保持待定，不自动合并，也不把缺少 alias 当成缺少 WAV。", "",
        "## 单元分配及原始证据", ""
    ]
    for row in data["units"]:
        out.extend([
            "### " + _safe(row["unit_id"]), "",
            "上下文：" + _safe(", ".join(row["contexts"])) +
            "；状态：" + row["status"] +
            "；已有决策：" + _safe(row["profile_decision"]) +
            "；置信度：" + _safe(row["confidence"]) + "。",
            "关联审核任务：" + _safe(", ".join(row["review_task_ids"]) or "无") + "。"
        ])
        for layer in row["pitch_evidence"]:
            out.append("- " + layer["pitch_group"] + "：")
            for s in layer["original_cv_examples"]:
                out.append("  - CV " + _safe(s["original_alias"]) + " | WAV " +
                           _safe(s["source_wav"]) + " | OTO " +
                           _safe(s["source_oto"]) + ":" + str(s["line_number"]) +
                           " | 选中录音序列=" + str(s["selected_recording_line"]))
            for s in layer["internal_transition_candidates"]:
                out.append("  - 内部过渡候选 " + _safe(s["original_alias"]) +
                           " | WAV " + _safe(s["source_wav"]) +
                           " | OTO " + _safe(s["source_oto"]) +
                           ":" + str(s["line_number"]))
            if not (layer["original_cv_examples"] or layer["internal_transition_candidates"]):
                out.append("  - 本预览暂无可直接关联来源；不能据此认定录音缺失")
        out.extend(["人工标注决定：待定；新 OTO 参数：未生成。", ""])
    return "\n".join(out)


def tuner_guide(data: dict) -> str:
    out = [
        "# PhonoWeave — 拆音与调用方案（调教师）", "",
        "状态：候选审核稿；不是可导入 OpenUtau 的 phonemizer/alias 映射。", "",
        "## 调用与听测规则", "",
        "1. 仅使用已经确认的原始 alias 及对应音高层，不自行构造上下文后缀。",
        "2. 起始 CV、内部 CV、VC、VV 等角色不能互换。暂定拆音不代表强制调用。",
        "3. 未解决上下文、无对应 alias 或跨音高覆盖异常时应记录，而非静默替换。",
        "4. 对短音、长音、连唱、不同元音衔接及跨音高调用分别听测。",
        "5. 报告问题时记录单元、音高、alias、源 WAV、测试短句和听感问题。",
        "6. 标注师确认边界与映射、听测通过后才可能形成生产使用说明。", "",
        "## 分单元审阅", ""
    ]
    for row in data["units"]:
        out.extend([
            "### " + _safe(row["unit_id"]), "",
            "上下文：" + _safe(", ".join(row["contexts"])) +
            "；状态：" + row["status"] +
            "；审核任务：" + _safe(", ".join(row["review_task_ids"]) or "无") + "。"
        ])
        for layer in row["pitch_evidence"]:
            examples = ", ".join(s["original_alias"] for s in layer["original_cv_examples"])
            out.append("- " + layer["pitch_group"] + " 原始 CV 别名示例：" +
                       _safe(examples or "无直接关联") + "；调用映射仍待审核")
        out.extend(["听测结论：待定；自动回退规则：未批准。", ""])
    return "\n".join(out)


def write_split_handoff(data: dict, output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("destination already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phonoweave-handoff-", dir=output.parent) as temp:
        stage = Path(temp)
        (stage / "split_handoff_manifest.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        (stage / "ANNOTATOR_SPLIT_AND_OTO_GUIDE.md").write_text(
            annotator_guide(data), encoding="utf-8")
        (stage / "TUNER_ALIAS_AND_CONTEXT_GUIDE.md").write_text(
            tuner_guide(data), encoding="utf-8")
        (stage / "README.txt").write_text(
            "REVIEW ONLY. No new alias or OTO parameter has been generated.\n"
            "Do not modify the original voicebank from this handoff.\n",
            encoding="utf-8")
        if output.exists():
            raise FileExistsError("destination appeared during export")
        os.rename(stage, output)
    return output
