"""Compile a candidate recording handoff into explicit split review directives.

The output is a production-blocked work specification, not an acoustic split
approval, OpenUtau alias table, or source OTO edit. Existing line-level OTO
links and Mandarin inventory contexts are kept distinct from verified routing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import tempfile

from .mandarin import _split_syllable
from .speaker_context_budget import _direct_context

HANDOFF_SCHEMA = "phonoweave.recording_line_handoff.v0.1"
BUDGET_SCHEMA = "phonoweave.speaker_context_budget_plan.v0.1"
OUTPUT_SCHEMA = "phonoweave.split_decision_directives.v0.1"

COLUMNS = (
    "line_id", "sequence", "pitch_group", "source_wav", "syllable_index",
    "syllable", "position", "base_unit", "final", "review_status",
    "candidate_unit_ids", "candidate_contexts", "evidence_kind",
    "annotator_action", "tuner_action", "source_oto_references",
    "production_routing", "new_oto_timing",
)


def _inventory(rows: list[dict]) -> tuple[dict, dict]:
    by_id = {}
    by_context = defaultdict(list)
    for row in rows:
        uid, base = row.get("unit_id"), row.get("base_unit")
        contexts = row.get("contexts", "").split("|")
        if not uid or uid in by_id or not base or not contexts or not all(contexts):
            raise ValueError("missing, duplicate or incomplete inventory unit")
        by_id[uid] = row
        for context in contexts:
            by_context[(base, context)].append(uid)
    return by_id, by_context


def _status(uid: str, protected: set[str], deferred: set[str]) -> str:
    if uid in protected:
        return "provisional_split_requires_review"
    if uid in deferred:
        return "unresolved_keep_separate_pending_review"
    return "not_optimized_requires_review"


def compile_split_decisions(handoff: dict, budget: dict,
                            unit_rows: list[dict]) -> dict:
    """Create one actionable check per syllable/pitch, with guarded abstentions."""
    if handoff.get("schema") != HANDOFF_SCHEMA or budget.get("schema") != BUDGET_SCHEMA:
        raise ValueError("incompatible recording handoff or budget schema")
    pitches = budget.get("pitch_groups")
    lines = budget.get("selection", {}).get("selected_lines")
    if (not isinstance(pitches, list) or not pitches or len(set(pitches)) != len(pitches)
            or not isinstance(lines, list) or len(set(lines)) != len(lines)
            or handoff.get("pitch_groups") != pitches
            or handoff.get("selected_line_count") != len(lines)
            or len(lines) != budget.get("selection", {}).get("selected_line_count")):
        raise ValueError("budget and handoff recording selection differ")
    by_id, by_context = _inventory(unit_rows)
    protected = set(budget.get("protected_units", []))
    deferred = set(budget.get("unresolved_or_not_proxy_supported_units_not_optimized", []))
    if protected & deferred or not (protected | deferred) <= by_id.keys():
        raise ValueError("conflicting or unknown protected/deferred units")
    if any(by_id[u].get("decision") != "split_recommended" for u in protected):
        raise ValueError("protected units must have a provisional split recommendation")

    directives, evidence_count, seen_lines = [], 0, set()
    for record, name in zip(handoff.get("recording_lines", []), lines):
        if record.get("sequence") != name or record.get("line_id") in seen_lines:
            raise ValueError("handoff sequence ordering or line identity mismatch")
        seen_lines.add(record["line_id"])
        sequence = name.split("_")
        layers = record.get("pitch_layers", [])
        if (len(layers) != len(pitches) or
                {layer.get("pitch_group") for layer in layers} != set(pitches)):
            raise ValueError("incomplete or duplicated pitch layers")
        for layer in layers:
            pitch = layer["pitch_group"]
            wav = layer.get("source_wav")
            cells = layer.get("syllables", [])
            if not wav or len(cells) != len(sequence):
                raise ValueError("source WAV or syllable count mismatch")
            for index, (word, cell) in enumerate(zip(sequence, cells)):
                position = "sequence_initial" if index == 0 else "sequence_internal"
                if (cell.get("syllable_index") != index or cell.get("syllable") != word
                        or cell.get("position") != position):
                    raise ValueError("handoff token, position or index mismatch")
                parsed = _split_syllable(word)
                base, final = parsed if parsed else ("", "")
                links = cell.get("internal_context_candidates", [])
                if not isinstance(links, list) or (links and index == 0):
                    raise ValueError("internal candidates must be attached to internal positions")
                candidates, refs = [], []
                if links:
                    evidence_kind = "source_linked_internal_VC_candidate"
                    for link in links:
                        uid, context = link.get("unit_id"), link.get("context")
                        unit = by_id.get(uid)
                        if (unit is None or unit["base_unit"] != base or
                                context not in unit["contexts"].split("|") or
                                not context.startswith("internal:") or
                                not link.get("source_oto") or
                                not link.get("oto_line_number") or
                                not link.get("original_alias") or
                                link.get("review_status") != (
                                    "provisional_split" if uid in protected
                                    else "unresolved_no_automatic_merge")):
                            raise ValueError("internal candidate inventory or provenance mismatch")
                        candidates.append({
                            "unit_id": uid, "context": context,
                            "decision_state": _status(uid, protected, deferred),
                            "evidence_kind": evidence_kind,
                            "source_oto": link["source_oto"],
                            "oto_line_number": link["oto_line_number"],
                            "original_alias": link["original_alias"],
                            "alias_mapping_approved": False,
                        })
                        refs.append(f"{link['source_oto']}:{link['oto_line_number']}")
                    evidence_count += len(candidates)
                elif index == 0 and parsed is not None:
                    # A lexical context is an annotation cue, NOT an existing
                    # alias assignment or proof of this phoneme's OTO segment.
                    context = _direct_context(base, final)
                    matches = by_context.get((base, context), []) if context else []
                    evidence_kind = "syllable_context_only" if matches else "no_direct_context_link"
                    for uid in sorted(set(matches)):
                        candidates.append({
                            "unit_id": uid, "context": context,
                            "decision_state": _status(uid, protected, deferred),
                            "evidence_kind": "syllable_context_only",
                            "source_oto": None, "oto_line_number": None,
                            "original_alias": None, "alias_mapping_approved": False,
                        })
                else:
                    evidence_kind = "no_verified_role_link"
                # Multiple distinct targets are not resolved by arbitrary priority.
                distinct = {c["unit_id"] for c in candidates}
                if len(distinct) > 1:
                    review = "ambiguous_context_hold"
                    annotator = "Inspect source and resolve competing realization IDs; keep OTO unchanged"
                    tuner = "Do not select a context alias; resolve ambiguous role assignment"
                elif candidates:
                    review = candidates[0]["decision_state"]
                    annotator = (
                        "Review previous vowel, consonant onset and CV boundary against the referenced OTO/WAV"
                        if links else
                        "Inspect initial consonant onset and vowel onset; locate matching original OTO entry"
                    )
                    tuner = (
                        "Listen to context-specific VC-to-CV joins at this pitch; routing pending review"
                        if links else
                        "Test isolated attack and pitch-layer behavior using confirmed existing aliases only"
                    )
                else:
                    review = ("no_internal_role_link_do_not_infer_absence" if index else
                              "no_supported_direct_unit_link")
                    annotator = (
                        "Inspect WAV and OTO for the internal transition; do not infer missing audio"
                        if index else
                        "Inspect initial CV or vowel onset; no candidate realization assigned"
                    )
                    tuner = "No generated alias/fallback; test only confirmed existing source aliases"
                directives.append({
                    "line_id": record["line_id"], "sequence": name,
                    "pitch_group": pitch, "source_wav": wav,
                    "syllable_index": index, "syllable": word,
                    "position": position, "base_unit": base, "final": final,
                    "review_status": review, "evidence_kind": evidence_kind,
                    "candidate_realizations": candidates,
                    "source_oto_references": sorted(set(refs)),
                    "annotator_action": annotator, "tuner_action": tuner,
                    "production_routing": None, "new_oto_timing": None,
                })
    if len(seen_lines) != len(lines):
        raise ValueError("handoff has missing or extra recording lines")
    if len(directives) != handoff.get("source_line_syllable_cells"):
        raise ValueError("handoff syllable cell total mismatch")
    if evidence_count != handoff.get("traceable_internal_candidates"):
        raise ValueError("internal candidate total mismatch")
    return {
        "schema": OUTPUT_SCHEMA,
        "source_handoff_schema": HANDOFF_SCHEMA,
        "selected_line_count": len(lines), "pitch_groups": pitches,
        "syllable_pitch_checks": len(directives),
        "source_linked_internal_candidates": evidence_count,
        "review_status_counts": dict(sorted(Counter(x["review_status"] for x in directives).items())),
        "directives": directives,
        "production_ready": False,
        "automatic_split_or_merge_approvals": 0,
        "alias_and_oto_changes": 0,
        "limitations": [
            "Only existing internal sequence links have OTO provenance in this output.",
            "An initial syllable lexical context is not a verified source alias.",
            "A split recommendation is not a verified synthesis-quality advantage.",
            "No unresolved context is automatically merged or assigned a fallback.",
        ],
    }


def _csv_safe(value) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def _write_csv(report: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for item in report["directives"]:
            values = {key: item.get(key, "") for key in COLUMNS}
            values["candidate_unit_ids"] = "|".join(c["unit_id"] for c in item["candidate_realizations"])
            values["candidate_contexts"] = "|".join(c["context"] for c in item["candidate_realizations"])
            values["source_oto_references"] = "|".join(item["source_oto_references"])
            writer.writerow({key: _csv_safe(val if val is not None else "")
                             for key, val in values.items()})


def _guide(report: dict, audience: str) -> str:
    annotator = audience == "annotator"
    title = "拆音标注指令" if annotator else "音源调用与调教检查"
    lines = [
        "# PhonoWeave — " + title, "",
        "候选工作稿；不包含已批准的 alias、合并决策或新 OTO 时间值。", "",
        "## 审核原则", "",
        "- 使用真实 WAV 与对应 OTO 原始条目核实声学证据；不从文件名推断时间戳。",
        "- 标记候选拆分、未决以及无来源关联的差异；任何缺口不直接等于缺少录音。",
        "- 生产级映射、自动 fallback 与 OTO 写入全部关闭。", "",
        "## 逐音节任务", "",
    ]
    for item in report["directives"]:
        action = item["annotator_action"] if annotator else item["tuner_action"]
        targets = ", ".join(x["unit_id"] for x in item["candidate_realizations"]) or "无安全候选"
        refs = ", ".join(item["source_oto_references"]) or "无已确认 OTO 关联"
        lines.append(
            f"- {item['line_id']} / {item['pitch_group']} / "
            f"{item['syllable_index'] + 1}:{item['syllable']} — "
            f"{item['review_status']}；候选：{targets}；"
            f"{refs}；操作：{action}"
        )
    return "\n".join(lines) + "\n"


def write_split_decisions(report: dict, output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phonoweave-directives-", dir=output.parent) as td:
        root = Path(td)
        (root / "split_decision_directives.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_csv(report, root / "ANNOTATOR_SPLIT_DIRECTIVES.csv")
        (root / "ANNOTATOR_SPLIT_GUIDE.md").write_text(
            _guide(report, "annotator"), encoding="utf-8")
        (root / "TUNER_CONTEXT_CHECKS.md").write_text(
            _guide(report, "tuner"), encoding="utf-8")
        (root / "README.txt").write_text(
            "REVIEW-ONLY split/merge directives; do not install as an OpenUtau configuration.\n"
            "CSV is spreadsheet-safe; machine consumers use lossless JSON.\n"
            "No original WAV/OTO is modified and no new alias is authorized.\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError("output directory appeared during compilation")
        os.rename(root, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile evidence-gated per-syllable split directives")
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.handoff.expanduser().open(encoding="utf-8") as f:
        handoff = json.load(f)
    with args.budget.expanduser().open(encoding="utf-8") as f:
        budget = json.load(f)
    with args.units.expanduser().open(encoding="utf-8", newline="") as f:
        units = list(csv.DictReader(f))
    report = compile_split_decisions(handoff, budget, units)
    path = write_split_decisions(report, args.output)
    print("Split decision directives:", path)
    print("Syllable/pitch reviews:", report["syllable_pitch_checks"])
    print("Source-linked internal candidates:", report["source_linked_internal_candidates"])
    print("Production alias/OTO generation: disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
