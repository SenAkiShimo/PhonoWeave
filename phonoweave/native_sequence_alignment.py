"""Constrained, abstaining sequence alignment of user-native labels to unverified v5 regions.

No IPA/G2P, OTO timestamps, or gold boundary annotations are read during inference.
The labels are *independent recording-script order*, not automatically inferred phones.
Candidate regions may be skipped; adjacent regions may realize one unit; a region
may contain two unresolved units, in which case NO interior boundary is invented.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = 'phonoweave.native_sequence_alignment.v0.6'
SEQUENCE_SCHEMA = 'phonoweave.native_sequences.v1'
GOLD_SCHEMA = 'phonoweave.independent_boundaries.v1'
MAX_REGIONS_PER_STEP = 4
MAX_UNITS_PER_STEP = 2
SKIP_COST = 1.6
SHARED_UNIT_PENALTY = 2.0
UNKNOWN_LABEL_COST = 2.8


@dataclass(frozen=True)
class Region:
    start_ms: float
    end_ms: float
    candidate_unit: str
    alternatives: tuple[str, ...]
    status: str

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


def _key(row: dict) -> tuple[str, str]:
    return row['bank'], row['source_wav']


def _sha(value: str) -> str:
    if len(value) != 64 or any(x not in '0123456789abcdef' for x in value.lower()):
        raise ValueError('invalid 64-digit SHA256')
    return value.lower()


def _read_v5_rows(path: Path) -> list[dict]:
    """Read only v5 acoustic region candidates; never use OTO diagnostics."""
    if path.is_dir():
        data = (path / 'candidate_segment_unit_hints.csv').read_bytes()
    else:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            matches = [n for n in names if n == 'results/candidate_segment_unit_hints.csv']
            if len(matches) != 1:
                raise ValueError('v5 archive must contain results/candidate_segment_unit_hints.csv')
            data = z.read(matches[0])
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def _parse_sequences(data: dict) -> list[dict]:
    if data.get('schema') != SEQUENCE_SCHEMA or not isinstance(data.get('entries'), list):
        raise ValueError('wrong sequences schema')
    found: set[tuple[str, str]] = set()
    for entry in data['entries']:
        key = _key(entry)
        if key in found:
            raise ValueError('duplicate sequence for one source WAV')
        found.add(key)
        if not all(isinstance(s, str) and s.strip() for s in entry['units']):
            raise ValueError('all user-native symbols must be nonempty strings')
        if not entry['units']:
            raise ValueError('empty user pronunciation order')
        if entry.get('provenance') != 'independent_recording_script':
            raise ValueError('sequence order must come from an independent script, not OTO')
        _sha(entry['source_pcm_sha256'])
    return data['entries']


def _read_regions(rows: list[dict]) -> tuple[dict[tuple[str, str], list[Region]], dict[tuple[str, str], str]]:
    groups: dict[tuple[str, str], list[Region]] = defaultdict(list)
    digests: dict[tuple[str, str], str] = {}
    for row in rows:
        key = _key(row)
        digest = _sha(row['source_pcm_sha256'])
        if key in digests and digests[key] != digest:
            raise ValueError('conflicting PCM hashes in v5 candidates')
        digests[key] = digest
        a, b = float(row['start_ms']), float(row['end_ms'])
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b):
            raise ValueError('invalid acoustic interval')
        alternatives = tuple(x for x in row.get('alternatives', '').split('|') if x)
        groups[key].append(Region(a, b, row.get('candidate_unit', ''), alternatives, row.get('status', '')))
    for key, arr in groups.items():
        arr.sort(key=lambda x: (x.start_ms, x.end_ms))
        if any(a.end_ms > b.start_ms + 0.02 for a, b in zip(arr, arr[1:])):
            raise ValueError('overlapping acoustic regions for ' + str(key))
    return groups, digests


def label_cost(region: Region, label: str) -> float:
    if region.status != 'candidate_UNVERIFIED_not_a_phone':
        return UNKNOWN_LABEL_COST
    if region.candidate_unit == label:
        return 0.0
    try:
        return 0.65 + 0.55 * region.alternatives.index(label)
    except ValueError:
        return UNKNOWN_LABEL_COST


def _assign_cost(regions: list[Region], labels: list[str]) -> float:
    """Fixed exploratory costs, not learned probabilities or validated confidence."""
    mean_cost = sum(min(label_cost(r, label) for label in labels) for r in regions) / len(regions)
    duration = regions[-1].end_ms - regions[0].start_ms
    # Very long grouping is uncertain. Do not impose a universal phonetic length.
    long_penalty = max(0.0, duration - 650 * len(labels)) / 650
    shared_penalty = SHARED_UNIT_PENALTY if len(labels) > 1 else 0.0
    return mean_cost + long_penalty + shared_penalty + 0.12 * (len(regions) - 1)


def align_regions(regions: list[Region], units: list[str]) -> dict:
    """Monotone global DP with region skips and 1..2 unresolved units per 1..4 regions.

    A transition with two units claims only that their *combined* time interval
    can be associated with both. It does NOT infer an internal phone boundary.
    """
    n, m = len(regions), len(units)
    if not n or not m:
        return {'status': 'abstain_empty_evidence', 'units': [], 'skipped_regions': n}
    # Each state keeps two distinct best paths for an ambiguity check.
    best: dict[tuple[int, int], list[tuple[float, tuple]]] = {(0, 0): [(0., ())]}

    def offer(i: int, j: int, score: float, path: tuple) -> None:
        key = (i, j)
        candidates = best.get(key, [])
        # Paths can differ because a noisy acoustic region is skipped.
        if any(q[1] == path for q in candidates):
            return
        candidates.append((score, path))
        candidates.sort(key=lambda q: (q[0], q[1]))
        best[key] = candidates[:2]

    for i in range(n + 1):
        for j in range(m + 1):
            for cost, path in list(best.get((i, j), [])):
                if i < n:
                    offer(i + 1, j, cost + SKIP_COST, path + (('skip', i, i + 1, j, j),))
                if i >= n or j >= m:
                    continue
                for rcount in range(1, min(MAX_REGIONS_PER_STEP, n - i) + 1):
                    reg_block = regions[i:i + rcount]
                    # Do not bridge silence/gaps with one unit.
                    if any(a.end_ms + 30 < b.start_ms for a, b in zip(reg_block, reg_block[1:])):
                        break
                    for ucount in range(1, min(MAX_UNITS_PER_STEP, m - j) + 1):
                        labels = units[j:j + ucount]
                        stepcost = _assign_cost(reg_block, labels)
                        offer(i + rcount, j + ucount, cost + stepcost,
                              path + (('assign', i, i + rcount, j, j + ucount),))
    finals = best.get((n, m))
    if not finals:
        return {'status': 'abstain_no_path', 'units': [], 'skipped_regions': n}
    cost, path = finals[0]
    runner_up_cost = finals[1][0] if len(finals) > 1 else None
    spans: list[dict] = []
    skipped = 0
    for op, i, end_i, j, end_j in path:
        if op == 'skip':
            skipped += end_i - i
            continue
        span = regions[i:end_i]
        label_slice = units[j:end_j]
        for idx, label in enumerate(label_slice, j):
            evidence = min(label_cost(r, label) for r in span)
            shared = len(label_slice) > 1
            spans.append({
                'unit_index': idx, 'symbol': label,
                'start_ms': round(span[0].start_ms, 3),
                'end_ms': round(span[-1].end_ms, 3),
                'supporting_regions': len(span),
                'min_label_cost': round(evidence, 3),
                'status': ('unresolved_shared_interval_NOT_a_phone_boundary' if shared
                           else 'candidate_UNVERIFIED' if evidence <= 1.75
                           else 'abstain_no_support_for_label')
            })
    if len(spans) != m:
        raise AssertionError('alignment failed to preserve label order')
    # A low-cost path can still be untrustworthy if the only available acoustic
    # evidence has no matching user label; never force that into a phone inventory.
    trustworthy = all(s['status'] == 'candidate_UNVERIFIED' for s in spans)
    # Top-two margin is a heuristic warning, not a posterior probability.
    global_margin = None if runner_up_cost is None else round(runner_up_cost - cost, 4)
    if global_margin is not None and global_margin < 0.25:
        trustworthy = False
        for span in spans:
            if span['status'] == 'candidate_UNVERIFIED':
                span['status'] = 'abstain_ambiguous_alignment'
    status = ('candidate_UNVERIFIED' if trustworthy
              else 'abstain_or_shared_unresolved')
    return {'status': status, 'units': spans, 'skipped_regions': skipped,
            'total_regions': n, 'total_units': m, 'path_cost': round(cost, 4),
            'second_path_cost': round(runner_up_cost, 4) if runner_up_cost is not None else None,
            'path_margin_NOT_probability': global_margin,
            'candidate_interpretation': 'ordered_alignment_not_verified_phone_segmentation'}


def run(v5: Path, sequences: Path, output: Path) -> dict:
    """Inference phase: gold boundaries and source OTO files never opened."""
    if output.exists():
        raise FileExistsError('output directory already exists')
    seq = _parse_sequences(json.loads(sequences.read_text(encoding='utf-8')))
    groups, hashes = _read_regions(_read_v5_rows(v5))
    # Validate all inputs before creating output (no half-written failed runs).
    for item in seq:
        key = _key(item)
        if key not in hashes:
            raise ValueError('sequence source is missing in v5: ' + str(key))
        if _sha(item['source_pcm_sha256']) != hashes[key]:
            raise ValueError('sequence PCM hash differs from v5: ' + str(key))
    output.mkdir(parents=True)
    records = []
    for item in sorted(seq, key=_key):
        key = _key(item)
        aligned = align_regions(groups[key], item['units'])
        records.append({'bank': key[0], 'source_wav': key[1],
                        'source_pcm_sha256': hashes[key], **aligned})
    with (output / 'alignments.json').open('w', encoding='utf8') as f:
        json.dump({'schema': SCHEMA, 'alignments': records}, f, ensure_ascii=False, indent=2)
        f.write('\n')
    counts = {'candidates': sum(x['status'] == 'candidate_UNVERIFIED' for x in records),
              'unresolved': sum(x['status'] != 'candidate_UNVERIFIED' for x in records),
              'sources': len(records), 'distinct_phone_boundaries_validated': False,
              'gold_annotations_consumed_by_inference': False, 'ipa_required': False,
              'new_user_labels_generated': False, 'auto_split_merge_enabled': False}
    (output / 'summary.json').write_text(json.dumps(counts, indent=2) + '\n', encoding='utf8')
    return counts


def evaluate(predictions: Path, gold: Path, output: Path) -> dict:
    """Independent post-hoc audit; no feedback from gold into DP or thresholds."""
    if output.exists():
        raise FileExistsError('evaluation directory already exists')
    pred = json.loads((predictions / 'alignments.json').read_text(encoding='utf8'))
    reference = json.loads(gold.read_text(encoding='utf8'))
    if pred.get('schema') != SCHEMA or reference.get('schema') != GOLD_SCHEMA:
        raise ValueError('wrong prediction or independent-boundary schema')
    mapped = {_key(x): x for x in pred['alignments']}
    errors = []
    abstained = 0
    included = 0
    for annotation in reference['entries']:
        if annotation.get('provenance') != 'independent_manual_annotation':
            raise ValueError('gold annotations must be manual and independent')
        key = _key(annotation)
        if key not in mapped:
            raise ValueError('gold source is missing prediction: ' + str(key))
        x = mapped[key]
        if _sha(annotation['source_pcm_sha256']) != x['source_pcm_sha256']:
            raise ValueError('gold source PCM hash mismatch')
        if len(annotation['units']) != len(x['units']):
            raise ValueError('gold count differs from sequence length')
        for a, b in zip(annotation['units'], x['units']):
            if a['symbol'] != b['symbol'] or int(a['unit_index']) != b['unit_index']:
                raise ValueError('gold labels/order are not the frozen prediction sequence')
            t0, t1 = float(a['start_ms']), float(a['end_ms'])
            if not (math.isfinite(t0) and math.isfinite(t1) and 0 <= t0 < t1):
                raise ValueError('invalid independent gold boundary')
            included += 1
            if b['status'] != 'candidate_UNVERIFIED':
                abstained += 1
            else:
                errors.extend([abs(b['start_ms'] - t0), abs(b['end_ms'] - t1)])
    output.mkdir(parents=True)
    report = {'gold_sources': len(reference['entries']), 'gold_units': included,
              'units_abstained': abstained, 'units_with_two_endpoint_errors': len(errors) // 2,
              'endpoint_mae_ms_on_nonabstentions': round(sum(errors) / len(errors), 3) if errors else None,
              'note': 'Independent, post-hoc endpoint MAE only; coverage and abstention must be reported together. No model selection against these annotations.'}
    (output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog='phonoweave-native-align')
    commands = p.add_subparsers(dest='command', required=True)
    a = commands.add_parser('align')
    a.add_argument('--v5', required=True, type=Path)
    a.add_argument('--sequences', required=True, type=Path)
    a.add_argument('--output', required=True, type=Path)
    e = commands.add_parser('evaluate')
    e.add_argument('--predictions', required=True, type=Path)
    e.add_argument('--gold', required=True, type=Path)
    e.add_argument('--output', required=True, type=Path)
    args = p.parse_args(argv)
    result = (run(args.v5, args.sequences, args.output) if args.command == 'align'
              else evaluate(args.predictions, args.gold, args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
