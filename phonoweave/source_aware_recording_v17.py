"""Physical-source-aware, minimal supplemental recording suggestions.

Reuses the pre-existing Mandarin diagnostic syllable inventory and v16 budget
scheduler. Counts byte-identical PCM waveforms only once per onset/final: many
OTO aliases do not mean independent recording coverage. No new phones, IPA,
or synthesis-unit decisions are made. Existing reclist items are *scheduled*
work, not acoustic evidence; they consume a diagnostic request slot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import wave
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from .diagnostic_selector import DiagnosticSelection, _candidates
from .evidence_gap import build_evidence_completion_plan
from .inventory import analyze_voicebank_inventory
from .mandarin import collect_observations
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .recording_budget_v16 import budget_recording_selection, _line, _read_existing_reclist
from .supplement_plan import build_supplement_plan

SCHEMA = 'phonoweave.source_aware_recording.v1.7'


def pcm_identity(path: Path) -> str:
    """Exact decoded PCM sample-byte fingerprint, including audio format.

    This detects identical digital copies, NOT resampled copies or whether two
    different WAVs came from independent *performances*.
    """
    h = hashlib.sha256()
    with wave.open(str(path), 'rb') as audio:
        if audio.getcomptype() != 'NONE':
            raise ValueError('compressed WAV is not usable for PCM identity')
        h.update(f'{audio.getnchannels()}|{audio.getsampwidth()}|{audio.getframerate()}|'.encode('ascii'))
        while True:
            frames = audio.readframes(65536)
            if not frames:
                break
            h.update(frames)
    return h.hexdigest()


def inspect_recording_sources(root: Path) -> dict[str, Any]:
    """Observe OTO-listed files but distinguish aliases from distinct PCM.

    A single source WAV can contain several aliases. Count a decoded source
    once for each parsed (base, final), not once per OTO row. Missing/corrupt
    WAVs contribute no evidence of recorded audio.
    """
    root = Path(root).expanduser().resolve()
    entries, warnings = load_voicebank(root)
    observations = collect_observations(entries, affix_pairs(load_prefix_maps(root)))
    fingerprints: dict[Path, str | None] = {}
    oto_counts: Counter[tuple[str, str]] = Counter()
    pcm_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    file_sets: dict[tuple[str, str], set[Path]] = defaultdict(set)
    bank_sets: dict[tuple[str, str], set[Path]] = defaultdict(set)
    bad_files: set[Path] = set()
    for ob in observations:
        key = (ob.base_unit, ob.final)
        oto_counts[key] += 1
        path = Path(ob.entry.wav_path).resolve()
        if path not in fingerprints:
            try:
                fingerprints[path] = pcm_identity(path)
            except (OSError, EOFError, ValueError, wave.Error):
                fingerprints[path] = None
                bad_files.add(path)
        digest = fingerprints[path]
        if digest is not None:
            pcm_sets[key].add(digest)
            file_sets[key].add(path)
            bank_sets[key].add(Path(ob.entry.oto_path).parent.resolve())
    return {
        'oto_rows': oto_counts,
        'distinct_pcm': {k: len(v) for k, v in pcm_sets.items()},
        'distinct_wav_paths': {k: len(v) for k, v in file_sets.items()},
        'oto_directories_with_audio': {k: len(v) for k, v in bank_sets.items()},
        'unreadable_or_missing_wav_count': len(bad_files),
        'oto_parse_warning_count': len(warnings),
        'fingerprinted_source_wav_paths': sum(v is not None for v in fingerprints.values()),
    }


def select_source_aware_items(supplement: Any, source_evidence: dict[str, Any],
                              existing_lines: tuple[str, ...] | list[str] = ()) -> tuple[DiagnosticSelection, list[dict]]:
    """Fill unmet target slots, ranking legal syllables by distinct source PCM.

    Existing scheduled lines satisfy *planning* slots only; their presence
    does not increase physically measured voicebank coverage. Never invent
    candidates outside the old Mandarin legal-syllable selector.
    """
    if any(not isinstance(x, str) for x in existing_lines):
        raise ValueError('existing reclist must contain strings')
    already = {s.strip() for s in existing_lines if s.strip()}
    selected = []
    planned = []
    unfilled = []
    observed = Counter(source_evidence['oto_rows'])
    distinct = source_evidence['distinct_pcm']
    paths = source_evidence['distinct_wav_paths']
    banks = source_evidence['oto_directories_with_audio']
    used: set[str] = set()
    for request in supplement.requests:
        for target in request.targets:
            if not isinstance(target.diagnostic_items, int) or target.diagnostic_items < 0:
                raise ValueError('invalid diagnostic target count')
            candidates = _candidates(request, observed, target.context_family)
            candidates.sort(key=lambda item: (
                distinct.get((item.base_unit, item.final), 0),
                banks.get((item.base_unit, item.final), 0),
                paths.get((item.base_unit, item.final), 0),
                item.existing_observations,
                item.syllable,
            ))
            if not candidates and target.diagnostic_items:
                unfilled.append(f'{request.base_unit}:{target.context_family}:requested={target.diagnostic_items}:available=0')
                continue
            # Match only known legal lines; never parse arbitrary author text as
            # phonetic content. More than one carrier may be in an existing list.
            options = []
            for repetition in range(1, target.diagnostic_items + 2):
                for candidate in candidates:
                    item = replace(candidate, replicate=repetition)
                    line = _line(item).strip()
                    if line not in {x[0] for x in options}:
                        options.append((line, item))
            matches = [(line, item) for line, item in options if line in already]
            scheduled_credit = min(target.diagnostic_items, len(matches))
            for line, item in matches[:scheduled_credit]:
                planned.append({
                    'line': line, 'base_unit': item.base_unit,
                    'context_family': target.context_family,
                    'status': 'existing_script_line_not_verified_audio',
                })
            need = max(0, target.diagnostic_items - scheduled_credit)
            for line, item in options:
                if not need:
                    break
                if line in already or line in used:
                    continue
                selected.append(item)
                used.add(line)
                need -= 1
            if need:
                unfilled.append(f'{request.base_unit}:{target.context_family}:requested_new={target.diagnostic_items - scheduled_credit}:unfilled={need}')
    return DiagnosticSelection(items=tuple(selected), unfilled=tuple(unfilled)), planned


def source_aware_plan(completion: Any, supplement: Any, source_evidence: dict[str, Any], *,
                      max_lines: int = 8, existing_lines: tuple[str, ...] | list[str] = ()) -> dict:
    items, preplanned = select_source_aware_items(supplement, source_evidence, existing_lines)
    report = budget_recording_selection(completion, items, max_lines=max_lines,
                                        existing_lines=existing_lines)
    report['schema'] = SCHEMA
    report['coverage_measure'] = 'distinct_exact_pcm_fingerprints_not_independent_performances'
    report['scheduled_lines_not_counted_as_audio_evidence'] = True
    report['already_scheduled'] = preplanned
    report['already_scheduled_count'] = len(preplanned)
    report['diagnostic_slots_requested'] = sum(
        t.diagnostic_items for req in supplement.requests for t in req.targets)
    report['new_lines_proposed'] = report['selected_count']
    report['unreadable_or_missing_wav_count'] = source_evidence['unreadable_or_missing_wav_count']
    report['oto_parse_warning_count'] = source_evidence['oto_parse_warning_count']
    report['fingerprinted_source_wav_paths'] = source_evidence['fingerprinted_source_wav_paths']
    for row in report['selected']:
        key = (row['base_unit'], row['syllable'][len(row['base_unit']):])
        row['existing_distinct_pcm_sources'] = source_evidence['distinct_pcm'].get(key, 0)
        row['existing_distinct_wav_paths'] = source_evidence['distinct_wav_paths'].get(key, 0)
        row['existing_oto_directories_with_audio'] = source_evidence['oto_directories_with_audio'].get(key, 0)
    report['no_new_synthesis_units_authorized'] = True
    return report


def plan_voicebank(root: Path, *, max_lines: int = 8, existing_reclist: Path | None = None) -> dict:
    root = Path(root)
    analysis = analyze_voicebank_inventory(root)
    completion = build_evidence_completion_plan(analysis)
    supplement = build_supplement_plan(completion)
    source_evidence = inspect_recording_sources(root)
    return source_aware_plan(completion, supplement, source_evidence, max_lines=max_lines,
                             existing_lines=_read_existing_reclist(existing_reclist))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Source-aware, minimal Mandarin diagnostic reclist')
    parser.add_argument('voicebank', type=Path)
    parser.add_argument('--max-lines', type=int, default=8)
    parser.add_argument('--existing-reclist', type=Path)
    parser.add_argument('--write-reclist', type=Path)
    parser.add_argument('--write-report', type=Path)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    outputs = [p.expanduser().resolve() for p in (args.write_reclist, args.write_report) if p]
    if len(set(outputs)) != len(outputs) or any(p.exists() for p in outputs):
        parser.error('output paths must be distinct and must not exist')
    if type(args.max_lines) is not int or not 0 <= args.max_lines <= 10000:
        parser.error('max-lines must be 0..10000')
    report = plan_voicebank(args.voicebank, max_lines=args.max_lines,
                            existing_reclist=args.existing_reclist)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.write_reclist:
        with args.write_reclist.open('x', encoding='utf-8') as f:
            f.write(''.join(line+'\n' for line in report['recording_lines']))
    if args.write_report:
        with args.write_report.open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write('\n')
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"PhonoWeave: {report['selected_count']} new lines; "
              f"{report['already_scheduled_count']} already scheduled; "
              f"max {report['max_lines']} new lines")
        for item in report['selected']:
            print(f"  {item['line']}  # {item['base_unit']}/{item['context_family']}, "
                  f"distinct digital PCM {item['existing_distinct_pcm_sources']}, "
                  f"OTO rows {item['existing_oto_observations']}")
        print('Preview only; original WAV/OTO and aliases unchanged.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())