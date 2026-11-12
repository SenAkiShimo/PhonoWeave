"""One-command preview/export of an existing Mandarin voicebank's recording gaps.

Composition only: PhonoWeave's existing analyzers and source-aware budgeted
reclist remain responsible for scientific decisions and legal syllables. This
module never generates units, aliases, OTO entries, or audio.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .coverage import analyze_coverage
from .evidence_gap import build_evidence_completion_plan
from .inspect import inspect_voicebank
from .inventory import analyze_voicebank_inventory
from .recording_budget_v16 import _read_existing_reclist
from .source_aware_recording_v17 import inspect_recording_sources, source_aware_plan
from .supplement_plan import build_supplement_plan

SCHEMA = "phonoweave.recording_workflow.v1.8"


def _check_voicebank(path: Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("voicebank must be an existing directory (not a ZIP or file)")
    return root


def _decision_row(decision: Any) -> dict[str, Any]:
    return {
        "base_unit": decision.base_unit,
        "class_name": decision.class_name,
        "decision": decision.decision,
        "confidence": decision.confidence,
        "acoustic_evidence": decision.acoustic_evidence,
        "synthesis_evidence": decision.synthesis_evidence,
    }


def _gap_row(gap: Any) -> dict[str, Any]:
    return {
        "base_unit": gap.base_unit,
        "class_name": gap.class_name,
        "gap_type": gap.gap_type,
        "priority": gap.priority,
        "next_action": gap.recommended_action,
        "role_scope": gap.role_scope,
        "context_families": list(gap.context_families),
        "rationale": list(gap.rationale),
    }


def build_recording_workflow(
    voicebank: Path,
    *,
    max_lines: int = 8,
    existing_reclist: Path | None = None,
) -> dict[str, Any]:
    """Inspect once, analyze once and compose established coverage recommendations.

    A source fingerprint is not proof of independent performances. A missing
    inventory decision or unrecognized language is not proof of full coverage.
    """
    if type(max_lines) is not int or not 0 <= max_lines <= 10000:
        raise ValueError("max_lines must be an integer from 0 through 10000")
    root = _check_voicebank(voicebank)
    inspection = inspect_voicebank(root)
    if inspection.oto_files == 0 or inspection.entries == 0:
        raise ValueError("no usable oto.ini entries found; cannot assess this voicebank")
    if inspection.valid_entries == 0:
        raise ValueError("all OTO-referenced WAV paths are missing; cannot assess recordings")
    coverage = analyze_coverage(root)
    if coverage.observations == 0:
        raise ValueError(
            "no recognized Mandarin syllables in available WAV/OTO records; "
            "this workflow cannot safely infer coverage for the user's notation"
        )

    analysis = analyze_voicebank_inventory(root)
    completion = build_evidence_completion_plan(analysis)
    supplement = build_supplement_plan(completion)
    physical = inspect_recording_sources(root)
    existing = _read_existing_reclist(existing_reclist)
    plan = source_aware_plan(completion, supplement, physical,
                             max_lines=max_lines, existing_lines=existing)

    categories = ("analyzed", "experimental", "unsupported")
    counts = {status: sum(x.status == status for x in coverage.items) for status in categories}
    warnings: list[str] = []
    if inspection.missing_wavs:
        warnings.append("some OTO rows reference missing WAV files; those rows do not prove audio coverage")
    if inspection.parse_warnings:
        warnings.append("some oto.ini rows could not be parsed")
    if physical["unreadable_or_missing_wav_count"]:
        warnings.append("some OTO-referenced WAV files cannot be decoded as uncompressed PCM")
    if counts["unsupported"]:
        warnings.append("some recognized onsets lack an implemented analysis method")
    if plan["unfilled_by_original_selector"]:
        warnings.append("some requested contexts have no supported legal syllable candidate")
    if not plan["recording_lines"]:
        warnings.append("no new diagnostic lines recommended; this does not prove complete acoustic coverage")

    return {
        "schema": SCHEMA,
        "scope": "existing_Mandarin_OTO_based_voicebank_only",
        "voicebank": {
            "root": str(root),
            "oto_files": inspection.oto_files,
            "oto_entries": inspection.entries,
            "oto_entries_with_existing_wav": inspection.valid_entries,
            "oto_entries_missing_wav": inspection.missing_wavs,
            "oto_parse_warnings": inspection.parse_warnings,
            "pcm_wav_paths_fingerprinted": physical["fingerprinted_source_wav_paths"],
            "unreadable_or_missing_pcm_files": physical["unreadable_or_missing_wav_count"],
            "subbank_count": len(inspection.subbanks),
        },
        "coverage": {
            "recognized_syllable_observations": coverage.observations,
            "onset_observations": coverage.onset_observations,
            "zero_onset_observations": coverage.zero_onset_observations,
            "onset_type_counts": counts,
            "onset_types": [{
                "base_unit": row.base_unit,
                "observations": row.observations,
                "status": row.status,
            } for row in coverage.items],
        },
        "evidence": {
            "decisions": [_decision_row(row) for row in analysis.decisions],
            "gaps": [_gap_row(gap) for gap in completion.gaps],
            "diagnostic_recording_gaps": len(completion.supplemental_recording),
            "needs_other_validation": len(completion.perceptual_validation),
        },
        "recording_plan": plan,
        "warnings": warnings,
        "restrictions": {
            "no_new_phoneme_or_alias_generated": True,
            "no_automatic_split_merge": True,
            "voicebank_unchanged": True,
            "reclist_lines_are_not_total_pitch_takes": True,
            "pcm_fingerprints_are_not_independent_performances": True,
        },
    }


def export_recording_workflow(report: dict[str, Any], target: Path, *, voicebank: Path) -> Path:
    """Export to a NEW external directory, leaving source audio/OTO untouched."""
    root = _check_voicebank(voicebank)
    destination = Path(target).expanduser().resolve()
    if destination == root or root in destination.parents:
        raise ValueError("export directory must be outside the source voicebank")
    if destination.exists():
        raise FileExistsError("refusing to overwrite an existing export directory")
    if not destination.parent.is_dir():
        raise ValueError("export parent directory must already exist")
    destination.mkdir()  # no overwrite, including racing processes
    try:
        text = "".join(line + "\n" for line in report["recording_plan"]["recording_lines"])
        (destination / "supplement_reclist.txt").write_text(text, encoding="utf-8")
        (destination / "recording_plan.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if "review_tasks" in report.get("recording_plan", {}):
            from .recording_review_v23 import format_review_tasks
            (destination / "recording_review.txt").write_text(
                format_review_tasks(report), encoding="utf-8",
            )
    except Exception:
        # Do not recursively delete unrelated files that might have appeared.
        for filename in ("supplement_reclist.txt", "recording_plan.json", "recording_review.txt"):
            (destination / filename).unlink(missing_ok=True)
        try:
            destination.rmdir()
        except OSError:
            pass
        raise
    return destination


def _print_preview(report: dict[str, Any]) -> None:
    info = report["voicebank"]
    coverage = report["coverage"]
    evidence = report["evidence"]
    plan = report["recording_plan"]
    print("PhonoWeave | Recording Plan Preview (Mandarin only)")
    print(f"Voicebank: {info['root']}")
    print(f"OTO entries: {info['oto_entries']}  |  WAV-backed entries: {info['oto_entries_with_existing_wav']}")
    print(f"Recognized observations: {coverage['recognized_syllable_observations']}")
    print(f"Evidence gaps: {len(evidence['gaps'])}; recording: {evidence['diagnostic_recording_gaps']}; other validation: {evidence['needs_other_validation']}")
    print(f"New recording lines: {plan['selected_count']}/{plan['max_lines']}  |  already scheduled: {plan.get('already_scheduled_count', 0)}")
    for row in plan["selected"]:
        print(f"  {row['line']}  # {row['base_unit']} / {row['context_family']} / {row['gap_type']}; distinct PCM={row.get('existing_distinct_pcm_sources', '?')}")
    for message in report["warnings"]:
        print("Warning: " + message)
    print("Preview only; no OTO/alias/voicebank changes. A line is not one total pitch take.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phonoweave plan-recording",
        description="Inspect a Mandarin voicebank and preview/export bounded source-aware supplemental recordings",
    )
    parser.add_argument("voicebank", type=Path, help="existing unpacked voicebank directory")
    parser.add_argument("--max-lines", type=int, default=8)
    parser.add_argument("--existing-reclist", type=Path, help="scheduled lines to exclude from new work")
    parser.add_argument("--native-reclist", type=Path, help="existing user-native recording templates (no Mandarin parsing)")
    parser.add_argument("--native-bank", action="append", help="target subbank in native mode; repeatable")
    parser.add_argument("--unit-evidence", action="append", type=Path,
                        help="optional existing user-native acoustic evidence.json; repeat 2+ times for sensitivity audit")
    parser.add_argument("--export", type=Path, help="create a NEW directory containing reclist and JSON report")
    parser.add_argument("--json", action="store_true", help="print full JSON report to stdout")
    args = parser.parse_args(argv)
    if args.export is not None:
        voicebank = _check_voicebank(args.voicebank)
        target = args.export.expanduser().resolve()
        if target == voicebank or voicebank in target.parents:
            parser.error("export directory must be outside the source voicebank")
        if target.exists():
            parser.error("export directory already exists")
    try:
        if args.native_reclist is not None:
            from .native_recording_plan_v20 import build_native_recording_workflow
            result = build_native_recording_workflow(
                args.voicebank, expected_reclist=args.native_reclist,
                max_lines=args.max_lines,
                bank_names=tuple(args.native_bank) if args.native_bank else None,
                existing_reclist=args.existing_reclist,
            )
        else:
            if args.native_bank:
                parser.error("--native-bank requires --native-reclist")
            result = build_recording_workflow(
                args.voicebank, max_lines=args.max_lines, existing_reclist=args.existing_reclist,
            )
        if args.unit_evidence:
            from .native_unit_review_v24 import attach_existing_unit_review
            result = attach_existing_unit_review(result, args.unit_evidence)
        if args.export is not None:
            export_recording_workflow(result, args.export, voicebank=args.voicebank)
    except (ValueError, FileNotFoundError, FileExistsError, PermissionError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        if args.native_reclist is not None:
            plan = result["recording_plan"]
            print("PhonoWeave | User-native template coverage (no IPA or phoneme inference)")
            print(f"Checked {result['coverage']['expected_template_count']} existing templates; proposed {plan['selected_count']} new lines")
            for item in plan["selected"]:
                print(f"  {item['line']}: missing in {', '.join(item['missing_audio_banks'])}")
            for warning in result["warnings"]:
                print("Warning:", warning)
        else:
            _print_preview(result)
        if args.unit_evidence:
            review = result["existing_sound_unit_review"]
            print("Existing user-native realization audit (not new phonemes):")
            for row in review["realizations"]:
                print(f"  {row['original_user_label']}: {row['review_status']}")
            print("No new recordings or aliases authorized by these acoustic reports.")
        if args.export is not None:
            print(f"Exported: {args.export.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())