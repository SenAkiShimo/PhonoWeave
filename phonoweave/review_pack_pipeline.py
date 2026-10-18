"""Run the existing recording review compilers as one non-destructive pipeline.

This emits a coherent *review* package, not a production reclist, phonemizer
configuration, or new OTO. Intermediate files remain available within the
finished package for staff review and downstream reproducibility.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

SCHEMA = "phonoweave.recording_review_pack.v0.1"
STAGES = ("scheme", "reference", "budget", "line", "directives")
ARTIFACTS = (
    "scheme/recording_units_review.csv",
    "scheme/internal_context_links.csv",
    "scheme/alias_records.jsonl",
    "scheme/recording_action_plan.json",
    "reference/reference_reclist_blueprint.json",
    "reference/reference_reclist.txt",
    "budget/speaker_context_budget_plan.json",
    "budget/speaker_context_pilot_reclist.txt",
    "line/recording_line_handoff.json",
    "line/ANNOTATOR_LINE_GUIDE.md",
    "line/TUNER_LINE_GUIDE.md",
    "directives/split_decision_directives.json",
    "directives/ANNOTATOR_SPLIT_GUIDE.md",
    "directives/TUNER_CONTEXT_CHECKS.md",
    "directives/ANNOTATOR_SPLIT_DIRECTIVES.csv",
)


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _build_stages(voicebank: Path, stage: Path, budget: int,
                  repeat_target: int) -> dict:
    """Reuse validated existing APIs; no duplicate parsing heuristics."""
    from .recording_scheme_review import build_from_voicebank, write_review_package
    from .reference_reclist_blueprint import (
        compile_reference_reclist, write_reference_reclist,
    )
    from .speaker_context_budget import (
        compile_speaker_context_budget, write_speaker_context_budget,
    )
    from .recording_line_handoff import compile_line_handoff, write_line_handoff
    from .split_decision_compiler import compile_split_decisions, write_split_decisions

    scheme = build_from_voicebank(voicebank)
    write_review_package(scheme, stage / "scheme")
    alias_records = [json.loads(line) for line in
                     (stage / "scheme/alias_records.jsonl").read_text(
                         encoding="utf-8").splitlines() if line.strip()]
    units = _rows(stage / "scheme/recording_units_review.csv")
    links = _rows(stage / "scheme/internal_context_links.csv")

    reference = compile_reference_reclist(alias_records)
    write_reference_reclist(reference, stage / "reference")
    proposal = compile_speaker_context_budget(
        reference, units, links, budget, repeat_target=repeat_target
    )
    write_speaker_context_budget(proposal, stage / "budget")
    line = compile_line_handoff(reference, proposal, units, links)
    write_line_handoff(line, stage / "line")
    directives = compile_split_decisions(line, proposal, units)
    write_split_decisions(directives, stage / "directives")

    if (line["selected_line_count"] != proposal["selection"]["selected_line_count"]
            or directives["syllable_pitch_checks"] != line["source_line_syllable_cells"]
            or directives["source_linked_internal_candidates"] !=
            line["traceable_internal_candidates"]):
        raise ValueError("inconsistent counts between recording review stages")
    if (proposal.get("production_ready") is not False
            or line.get("production_ready") is not False
            or directives.get("production_ready") is not False):
        raise ValueError("unexpected production-ready state in review-only pipeline")
    return {
        "reference_sequences": reference["all_pitch_reference_sequences"],
        "selected_lines_per_pitch": line["selected_line_count"],
        "pitch_groups": line["pitch_groups"],
        "syllable_pitch_checks": directives["syllable_pitch_checks"],
        "source_linked_internal_candidates":
            directives["source_linked_internal_candidates"],
    }


def _manifest(stage: Path, stats: dict, budget: int, repeat_target: int) -> dict:
    digest = {}
    for name in ARTIFACTS:
        path = stage / name
        if not path.is_file():
            raise ValueError(f"missing required review artifact: {name}")
        digest[name] = sha256(path.read_bytes()).hexdigest()
    return {
        "schema": SCHEMA,
        "stage_directories": list(STAGES),
        "budget_lines_per_pitch_max": budget,
        "repeat_target_distinct_source_sequences": repeat_target,
        "counts": stats,
        "artifact_sha256": digest,
        "export_status": "human_review_required",
        "production_alias_or_oto_export": "disabled",
        "new_recordings_authorized": 0,
        "source_voicebank_mutation": "none",
    }


def compile_review_pack(voicebank: Path, output: Path, *, budget: int = 60,
                        repeat_target: int = 3) -> dict:
    """Build in a temporary sibling and publish only on complete success.

    An existing destination is never overwritten. Outputs inside the source
    voicebank are prohibited, including when reached through symlinks.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if type(repeat_target) is not int or not 1 <= repeat_target <= 10:
        raise ValueError("repeat_target must be in 1..10")
    source = voicebank.expanduser().resolve(strict=True)
    if not source.is_dir():
        raise ValueError("source voicebank is not a directory")
    output = output.expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("output must not be inside the source voicebank")
    if output.exists():
        raise FileExistsError(f"destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phonoweave-review-pack-",
                                     dir=output.parent) as temp:
        stage = Path(temp) / "pack"
        stage.mkdir()
        stats = _build_stages(source, stage, budget, repeat_target)
        manifest = _manifest(stage, stats, budget, repeat_target)
        (stage / "review_pack_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n", encoding="utf-8"
        )
        (stage / "README.txt").write_text(
            "REVIEW-ONLY RECORDING PACK; NOT A PRODUCTION OPENUTAU BANK.\n"
            "The five stage directories and checksums are in review_pack_manifest.json.\n"
            "Use budget/ for the candidate reclist, directives/ for staff instructions.\n"
            "CSV aliases are spreadsheet-safe; use scheme/alias_records.jsonl\n"
            "for lossless original strings. No new OTO, alias or WAV is authorized.\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(f"destination appeared during compilation: {output}")
        os.rename(stage, output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile a review-only recording, annotation and tuning package"
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--voicebank", type=Path,
                         help="local source voicebank directory")
    sources.add_argument("--manifest", type=Path,
                         help="local registered voicebank manifest JSON")
    parser.add_argument("--bank-id", help="bank identifier (with --manifest)")
    parser.add_argument("--budget", type=int, default=60,
                        help="maximum recording sequence lines per pitch (default: 60)")
    parser.add_argument("--repeat-target", type=int, default=3,
                        help="candidate context source depth goal (default: 3)")
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory (will not overwrite)")
    args = parser.parse_args(argv)
    if bool(args.manifest) != bool(args.bank_id):
        parser.error("--manifest and --bank-id must be used together")
    try:
        if args.manifest:
            from .recording_scheme_review_cli import voicebank_from_manifest
            source = voicebank_from_manifest(args.manifest, args.bank_id)
        else:
            source = args.voicebank
        report = compile_review_pack(
            source, args.output, budget=args.budget,
            repeat_target=args.repeat_target,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print("Recording review pack:", args.output.expanduser().resolve())
    print("Selected lines per pitch:", report["counts"]["selected_lines_per_pitch"])
    print("Syllable/pitch checks:", report["counts"]["syllable_pitch_checks"])
    print("Production export: disabled; source bank unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
