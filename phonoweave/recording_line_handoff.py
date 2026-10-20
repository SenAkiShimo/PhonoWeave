"""Compile one selected reclist into traceable annotation and tuning assignments.

Existing WAV and OTO references provide *candidate* links only. Neither new
alias routing nor OTO timing is inferred. No source voicebank is changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

BLUEPRINT_SCHEMA = "phonoweave.reference_reclist_blueprint.v0.1"
BUDGET_SCHEMA = "phonoweave.speaker_context_budget_plan.v0.1"
OUTPUT_SCHEMA = "phonoweave.recording_line_handoff.v0.1"
LINK_EVIDENCE = "ordered_reclist_filename_and_oto_boundary"
LINK_STATUS = "candidate_link_not_synthesis_validated"

COLUMNS = (
    "line_id", "pitch_group", "source_wav", "syllable_index", "syllable",
    "position", "candidate_unit_ids", "candidate_contexts", "original_vc_aliases",
    "oto_sources", "annotation_status", "tuning_status",
)


def compile_line_handoff(blueprint: dict, budget: dict,
                         unit_rows: list[dict], link_rows: list[dict]) -> dict:
    """Assign recording lines to OTO-linked internal context review targets.

    The filename sequence provides the token order; only source-linked VC
    records provide internal context candidates. Token spelling alone never
    authorizes a role-specific source alias.
    """
    if blueprint.get("schema") != BLUEPRINT_SCHEMA or budget.get("schema") != BUDGET_SCHEMA:
        raise ValueError("incompatible blueprint or budget schema")
    pitches = budget.get("pitch_groups")
    if not isinstance(pitches, list) or not pitches or len(pitches) != len(set(pitches)):
        raise ValueError("invalid pitch groups")
    if pitches != blueprint.get("pitch_groups"):
        raise ValueError("budget and blueprint pitch groups differ")
    selection = budget.get("selection", {}).get("selected_lines")
    if not isinstance(selection, list) or len(selection) != len(set(selection)):
        raise ValueError("invalid or duplicate selected recording lines")
    if (len(selection) != budget["selection"].get("selected_line_count")
            or len(selection) > budget.get("per_pitch_line_budget_max", -1)):
        raise ValueError("selected recording lines exceed or mismatch budget")

    inventory = {}
    for row in unit_rows:
        uid = row.get("unit_id")
        if not uid or uid in inventory or not row.get("base_unit"):
            raise ValueError("duplicate or incomplete synthesis unit")
        inventory[uid] = row
    protected = set(budget.get("protected_units", []))
    deferred = set(budget.get("unresolved_or_not_proxy_supported_units_not_optimized", []))
    if not protected <= inventory.keys() or not deferred <= inventory.keys() or protected & deferred:
        raise ValueError("budget contains invalid protected/deferred unit IDs")

    reference = {}
    for row in blueprint.get("sequences", []):
        name = row.get("sequence")
        if not isinstance(name, str) or not name or name in reference:
            raise ValueError("duplicate or invalid reference sequence")
        reference[name] = row

    selected_wavs = {}
    assignments = []
    for line_number, sequence in enumerate(selection, 1):
        source = reference.get(sequence)
        if source is None or source.get("status") != "all_pitch_groups_observed":
            raise ValueError("selected recording line has no verified source sequence")
        tokens = sequence.split("_")
        if not tokens or any(not t or not t.isalpha() for t in tokens):
            raise ValueError("invalid recording sequence tokens")
        wavs = source.get("source_wavs", [])
        if len(wavs) != len(pitches) or {r.get("pitch_group") for r in wavs} != set(pitches):
            raise ValueError("incomplete or duplicate pitch source for selected line")
        layers = []
        for pitch in pitches:
            src = next(s for s in wavs if s["pitch_group"] == pitch)
            wav = src.get("source_wav", "")
            filename = PurePosixPath(wav)
            if (filename.is_absolute() or ".." in filename.parts or
                    not filename.parts or filename.parts[0] != pitch or
                    filename.name != f"_{sequence}{pitch}.wav"):
                raise ValueError("source WAV name does not match the selected sequence")
            oto_files = src.get("oto_files", [])
            if not isinstance(oto_files, list) or not oto_files:
                raise ValueError("missing source OTO provenance")
            if (pitch, wav) in selected_wavs:
                raise ValueError("same WAV assigned to multiple recording lines")
            selected_wavs[(pitch, wav)] = {
                "line_number": line_number, "tokens": tokens,
                "oto_files": set(oto_files), "links": {},
            }
            layers.append({
                "pitch_group": pitch, "source_wav": wav,
                "source_oto_files": oto_files, "syllables": [],
            })
        assignments.append({
            "line_id": f"RL-{line_number:03d}", "sequence": sequence,
            "syllable_count": len(tokens), "pitch_layers": layers,
        })

    seen_links = set()
    selected_links = 0
    for link in link_rows:
        pitch, wav = link.get("pitch_group"), link.get("source_wav")
        source = selected_wavs.get((pitch, wav))
        if source is None:
            continue  # This reference line was not selected at this budget.
        uid = link.get("unit_id")
        unit = inventory.get(uid)
        if unit is None:
            raise ValueError("selected internal link targets an unknown unit")
        try:
            position = int(link["boundary_order"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("non-numeric internal boundary order") from exc
        tokens = source["tokens"]
        if not (0 < position < len(tokens) and
                link.get("previous_syllable") == tokens[position - 1] and
                link.get("next_syllable") == tokens[position] and
                link.get("syllable") == tokens[position] and
                link.get("source_oto") in source["oto_files"] and
                link.get("base_unit") == unit["base_unit"] and
                link.get("context") in unit.get("contexts", "").split("|") and
                link.get("context", "").startswith("internal:") and
                link.get("evidence") == LINK_EVIDENCE and
                link.get("review_status") == LINK_STATUS):
            raise ValueError("internal candidate provenance or sequence mismatch")
        identity = (pitch, wav, link.get("source_oto"), link.get("line_number"),
                    link.get("original_alias"), uid, position)
        if identity in seen_links:
            raise ValueError("duplicate internal link")
        seen_links.add(identity)
        source["links"].setdefault(position, []).append({
            "unit_id": uid, "context": link["context"],
            "source_oto": link["source_oto"],
            "oto_line_number": link["line_number"],
            "original_alias": link["original_alias"],
            "review_status": ("provisional_split" if uid in protected
                              else "unresolved_no_automatic_merge"),
        })
        selected_links += 1

    covered_units = set()
    protected_internal = {
        uid for uid in protected
        if any(c.startswith("internal:") for c in inventory[uid].get("contexts", "").split("|"))
    }
    for line in assignments:
        tokens = line["sequence"].split("_")
        for layer in line["pitch_layers"]:
            source = selected_wavs[(layer["pitch_group"], layer["source_wav"])]
            for position, syllable in enumerate(tokens):
                links = sorted(source["links"].get(position, []),
                               key=lambda x: (x["unit_id"], x["source_oto"], str(x["oto_line_number"])))
                covered_units.update(x["unit_id"] for x in links)
                layer["syllables"].append({
                    "syllable_index": position, "syllable": syllable,
                    "position": "sequence_initial" if position == 0 else "sequence_internal",
                    "internal_context_candidates": links,
                    "annotation_status": "source_boundary_review_required",
                    "tuning_status": "not_routed_no_audio_validation",
                })
    return {
        "schema": OUTPUT_SCHEMA,
        "pitch_groups": pitches,
        "selected_line_count": len(assignments),
        "source_line_syllable_cells": sum(
            len(layer["syllables"]) for row in assignments for layer in row["pitch_layers"]),
        "traceable_internal_candidates": selected_links,
        "protected_internal_units_with_candidate": sorted(covered_units & protected_internal),
        "protected_internal_units_without_candidate": sorted(protected_internal - covered_units),
        "protected_noninternal_units_outside_this_linker": sorted(protected - protected_internal),
        "unresolved_units_with_internal_candidate": sorted(covered_units & deferred),
        "recording_lines": assignments,
        "production_ready": False,
        "alias_mapping_authorized": False,
        "oto_timing_authorized": False,
        "recordings_authorized": 0,
    }


def _write_csv(path: Path, report: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for line in report["recording_lines"]:
            for layer in line["pitch_layers"]:
                for cell in layer["syllables"]:
                    links = cell["internal_context_candidates"]
                    values = {
                        "line_id": line["line_id"], "pitch_group": layer["pitch_group"],
                        "source_wav": layer["source_wav"],
                        "syllable_index": cell["syllable_index"],
                        "syllable": cell["syllable"], "position": cell["position"],
                        "candidate_unit_ids": "|".join(x["unit_id"] for x in links),
                        "candidate_contexts": "|".join(x["context"] for x in links),
                        "original_vc_aliases": "|".join(x["original_alias"] for x in links),
                        "oto_sources": "|".join(
                            f"{x['source_oto']}:{x['oto_line_number']}" for x in links),
                        "annotation_status": cell["annotation_status"],
                        "tuning_status": cell["tuning_status"],
                    }
                    # Presentation CSV must not become canonical alias input.
                    safe = {key: ("'" + str(val) if str(val).lstrip().startswith(
                        ("=", "+", "-", "@")) else val) for key, val in values.items()}
                    writer.writerow(safe)


def _guide(report: dict, audience: str) -> str:
    if audience == "annotator":
        steps = [
            "核对每行音高子库、原始 WAV 和完整音节顺序。",
            "根据实际波形和频谱核实声母起点、元音起点及内部过渡；不能按文件名猜测时间参数。",
            "internal_context_candidates 仅为原 OTO 顺序提供的候选；确认后另行审批拆分。",
            "逐个审阅 offset、consonant、cutoff、preutterance、overlap，不得照搬旧 OTO。",
        ]
    else:
        steps = [
            "先对照原始 WAV/OTO 与现有 alias；本文件不提供可导入的 phonemizer 路由。",
            "逐项测试起始 CV、内部 CV、VC/VV 过渡、长音、短音及跨音高衔接。",
            "未审核的独立上下文不可强行调用，也不能静默合并或假造 fallback。",
            "反馈时记录行号、音高、音节序号、原 alias、WAV、听感问题。",
        ]
    lines = [
        "# PhonoWeave — " + ("逐行拆音与 OTO 检查" if audience == "annotator" else "逐行音源与调教检查"),
        "", "状态：候选审核方案，不是正式 OTO/alias 或可立即使用的声库。", "",
        "## 工作规则", "",
    ] + [f"{i}. {step}" for i, step in enumerate(steps, 1)] + ["", "## 逐行来源与候选", ""]
    for row in report["recording_lines"]:
        lines += [f"### {row['line_id']} — `{row['sequence']}`", ""]
        for layer in row["pitch_layers"]:
            lines.append(f"**{layer['pitch_group']}** — `{layer['source_wav']}`")
            for cell in layer["syllables"]:
                links = cell["internal_context_candidates"]
                if links:
                    desc = "; ".join(
                        f"{l['unit_id']} / {l['context']} / {l['original_alias']}"
                        f" ({l['source_oto']}:{l['oto_line_number']}, {l['review_status']})"
                        for l in links)
                else:
                    desc = "无已验证的内部上下文来源关联（不代表缺失录音）"
                lines.append(f"- {cell['syllable_index'] + 1}. {cell['syllable']}: {desc}")
            lines.append("")
    return "\n".join(lines) + "\n"


def write_line_handoff(report: dict, output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".phonoweave-line-handoff-") as td:
        root = Path(td)
        (root / "recording_line_handoff.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _write_csv(root / "ANNOTATOR_LINE_TARGETS.csv", report)
        (root / "ANNOTATOR_LINE_GUIDE.md").write_text(_guide(report, "annotator"), encoding="utf-8")
        (root / "TUNER_LINE_GUIDE.md").write_text(_guide(report, "tuner"), encoding="utf-8")
        (root / "README.txt").write_text(
            "CANDIDATE LINE-LEVEL SPLIT HANDOFF ONLY.\n"
            "Read recording_line_handoff.json for lossless alias strings.\n"
            "CSV is a human review sheet, never a production alias source.\n"
            "No recordings, OTO values, aliases or phonemizer routes are authorized.\n",
            encoding="utf-8")
        if output.exists():
            raise FileExistsError("destination appeared during compilation")
        os.rename(root, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create line-specific annotator and tuner handoff")
    parser.add_argument("--blueprint", required=True, type=Path)
    parser.add_argument("--budget", required=True, type=Path)
    parser.add_argument("--units", required=True, type=Path)
    parser.add_argument("--internal-links", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    with args.blueprint.expanduser().open(encoding="utf-8") as f:
        blueprint = json.load(f)
    with args.budget.expanduser().open(encoding="utf-8") as f:
        budget = json.load(f)
    with args.units.expanduser().open(encoding="utf-8", newline="") as f:
        units = list(csv.DictReader(f))
    with args.internal_links.expanduser().open(encoding="utf-8", newline="") as f:
        links = list(csv.DictReader(f))
    report = compile_line_handoff(blueprint, budget, units, links)
    where = write_line_handoff(report, args.output)
    print("Recording line handoff:", where)
    print("Selected lines:", report["selected_line_count"])
    print("Traceable internal candidates:", report["traceable_internal_candidates"])
    print("Original voicebank unchanged; production mapping disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
