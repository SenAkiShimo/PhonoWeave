"""Bounded, explainable supplement reclists for the *existing* Mandarin coverage workflow.

This is NOT a phone discovery system. It consumes the established evidence-gap
planner and selected legal Mandarin syllables, then orders recording *lines* by
coverage priority. It never expands a research motif into a new phoneme, changes
an alias/OTO entry, or assumes one line equals one take per pitch.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .diagnostic_selector import DiagnosticSelection, select_diagnostic_items
from .evidence_gap import EvidenceCompletionPlan, build_evidence_completion_plan
from .inventory import analyze_voicebank_inventory
from .supplement_plan import build_supplement_plan
from .supplement_reclist import build_supplement_reclist

SCHEMA = 'phonoweave.budgeted_supplement_reclist.v1.6'
PRIORITIES = {'high': 0, 'medium': 1, 'low': 2}


def _line(item: Any) -> str:
    # Source of truth is the EXISTING reclist renderer, not a second formatter.
    one = DiagnosticSelection(items=(item,), unfilled=())
    lines = build_supplement_reclist(one).lines
    if len(lines) != 1 or not lines[0].text.strip():
        raise ValueError('existing reclist generator did not return one valid line')
    return lines[0].text


def budget_recording_selection(
    completion: EvidenceCompletionPlan,
    selection: DiagnosticSelection,
    *,
    max_lines: int = 8,
    existing_lines: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Allocate one line per distinct evidence gap before allocating replicates.

    The pre-existing selector supplies legal, least-observed syllables. This
    layer only orders, caps and de-duplicates *suggestions*. Neither OTO
    observation counts nor existing script lines prove phonetic equivalence.
    """
    if type(max_lines) is not int or not 0 <= max_lines <= 10000:
        raise ValueError('max_lines must be an integer from 0 through 10000')
    if any(not isinstance(x, str) for x in existing_lines):
        raise ValueError('existing recording lines must be strings')
    already = {s.strip() for s in existing_lines if s.strip()}

    approved: dict[str, Any] = {}
    for gap in completion.gaps:
        if gap.recommended_action != 'supplemental_recording':
            continue
        if gap.priority not in PRIORITIES:
            raise ValueError('unsupported gap priority')
        if gap.base_unit in approved:
            raise ValueError('ambiguous repeated recording gap for ' + gap.base_unit)
        approved[gap.base_unit] = gap

    # Group by the exact requested base/context, not by phonetically guessed IDs.
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in selection.items:
        gap = approved.get(item.base_unit)
        if gap is None:
            continue  # never turn perceptual-only or resolved claims into recordings
        if gap.context_families and item.context_family not in gap.context_families:
            raise ValueError('selected syllable is outside its requested context family')
        if item.replicate < 1 or item.existing_observations < 0:
            raise ValueError('invalid supplement candidate counts')
        key = (item.base_unit, item.context_family, item.role_scope or '')
        groups[key].append({
            'line': _line(item),
            'base_unit': item.base_unit,
            'syllable': item.syllable,
            'context_family': item.context_family,
            'role_scope': item.role_scope,
            'replicate': item.replicate,
            'existing_oto_observations': item.existing_observations,
            'priority': gap.priority,
            'gap_type': gap.gap_type,
            'why': 'diagnose_' + gap.gap_type + '_in_' + item.context_family,
        })

    # Preserve the existing selector's least-observed-first ordering within a
    # family. Rotate across families before repeating any family.
    ordered_keys = sorted(groups, key=lambda k: (PRIORITIES[groups[k][0]['priority']], k))
    layers = max((len(v) for v in groups.values()), default=0)
    candidates: list[dict[str, Any]] = []
    for depth in range(layers):
        for priority in range(3):
            for key in ordered_keys:
                group = groups[key]
                if PRIORITIES[group[0]['priority']] == priority and depth < len(group):
                    candidates.append(group[depth])

    chosen: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    seen = set(already)
    for candidate in candidates:
        text = candidate['line'].strip()
        if text in already:
            reason = 'line_already_in_supplied_reclist'
        elif text in seen:
            reason = 'duplicate_line_in_suggestions'
        elif len(chosen) >= max_lines:
            reason = 'global_line_budget_reached'
        else:
            chosen.append(candidate)
            seen.add(text)
            continue
        omitted.append({**candidate, 'reason': reason})

    grouped_covered = sorted({(x['base_unit'], x['context_family']) for x in chosen})
    grouped_all = sorted({(x['base_unit'], x['context_family']) for x in candidates})
    return {
        'schema': SCHEMA,
        'mode': 'diagnostic_recording_suggestions_not_new_synthesis_units',
        'language_scope': 'existing_mandarin_syllable_selector_only',
        'max_lines': max_lines,
        'selected_count': len(chosen),
        'recording_lines': [x['line'] for x in chosen],
        'selected': chosen,
        'deferred_or_skipped': omitted,
        'unfilled_by_original_selector': list(selection.unfilled),
        'contexts_suggested': len(grouped_all),
        'contexts_in_selected_lines': len(grouped_covered),
        'pitch_policy': 'existing_planner_policy_still_applies_per_line',
        'line_count_is_not_number_of_total_pitch_takes': True,
        'auto_split_merge': False,
        'aliases_changed': False,
        'existing_voicebank_unchanged': True,
    }


def _read_existing_reclist(path: Path | None) -> list[str]:
    if path is None:
        return []
    content = path.read_text(encoding='utf-8-sig').splitlines()
    return [s.strip() for s in content if s.strip() and not s.lstrip().startswith('#')]


def plan_voicebank(root: Path, *, max_lines: int = 8, existing_reclist: Path | None = None) -> dict:
    """Re-use the existing soundbank analyzer; new phonemes cannot be invented."""
    analysis = analyze_voicebank_inventory(root)
    completion = build_evidence_completion_plan(analysis)
    supplement = build_supplement_plan(completion)
    selection = select_diagnostic_items(root, supplement)
    return budget_recording_selection(
        completion, selection, max_lines=max_lines,
        existing_lines=_read_existing_reclist(existing_reclist),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Small-budget diagnostic reclist (Mandarin module)')
    p.add_argument('voicebank', type=Path)
    p.add_argument('--max-lines', type=int, default=8,
                   help='maximum NEW recording-script lines (not per-pitch takes), default 8')
    p.add_argument('--existing-reclist', type=Path,
                   help='optional already-scheduled lines to avoid proposing twice')
    p.add_argument('--source-aware', action='store_true',
                   help='rank by distinct underlying PCM recordings, not duplicate OTO aliases')
    p.add_argument('--write-reclist', type=Path,
                   help='write new lines (never overwrite); omit to preview')
    p.add_argument('--write-report', type=Path,
                   help='write JSON explanation (never overwrite)')
    p.add_argument('--json', action='store_true', help='print JSON to stdout')
    args = p.parse_args(argv)

    outputs = [x.expanduser().resolve() for x in (args.write_reclist, args.write_report) if x]
    if len(set(outputs)) != len(outputs) or any(x.exists() for x in outputs):
        p.error('output files must not already exist or point to the same path')
    if not 0 <= args.max_lines <= 10000:
        p.error('--max-lines must be between 0 and 10000')
    if args.source_aware:
        from .source_aware_recording_v17 import plan_voicebank as source_plan_voicebank
        result = source_plan_voicebank(args.voicebank, max_lines=args.max_lines,
                                       existing_reclist=args.existing_reclist)
    else:
        result = plan_voicebank(args.voicebank, max_lines=args.max_lines,
                                existing_reclist=args.existing_reclist)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.write_reclist:
        with args.write_reclist.open('x', encoding='utf-8') as f:
            f.write(''.join(x + '\n' for x in result['recording_lines']))
    if args.write_report:
        with args.write_report.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write('\n')
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print('PhonoWeave bounded diagnostic reclist (not a synthesis-unit decision)')
        print(f"Selected {result['selected_count']} / {result['max_lines']} lines; "
              f"covered {result['contexts_in_selected_lines']} / {result['contexts_suggested']} "
              'suggested base/context groups')
        for row in result['selected']:
            print(f"  {row['line']}  # {row['base_unit']}/{row['context_family']}: "
                  f"{row['gap_type']}, seen {row['existing_oto_observations']} OTO observations")
        print(f"Deferred/skipped: {len(result['deferred_or_skipped'])}; "
              f"unfillable: {len(result['unfilled_by_original_selector'])}")
        print('Note: each line may require multiple pitch takes; no aliases or voicebank files changed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())