"""Frozen, IPA-independent validation for unverified user-native sequence alignments.

The freeze phase sees v5 acoustic regions, independent user scripts and v6
predictions, but never gold boundaries. The score phase sees only the frozen
bundle and independently annotated gold. No OTO data or IPA is required.

All reported metrics are research diagnostics, not validated phone discovery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .native_sequence_alignment import (
    GOLD_SCHEMA, SCHEMA as V6_SCHEMA, _key, _parse_sequences, _read_regions,
    _read_v5_rows, _sha, align_regions,
)

SCHEMA = 'phonoweave.user_native_validation.v0.7'
SEED = 20261110


def _json_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _v5_bytes(path: Path) -> bytes:
    if path.is_dir():
        return (path / 'candidate_segment_unit_hints.csv').read_bytes()
    import zipfile
    with zipfile.ZipFile(path) as z:
        return z.read('results/candidate_segment_unit_hints.csv')


def _unique(items: list[dict], name: str) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for item in items:
        key = _key(item)
        if not (all(isinstance(part, str) and part for part in key)):
            raise ValueError(f'{name} has invalid source key')
        if key in out:
            raise ValueError(f'{name} contains duplicate source: {key}')
        out[key] = item
    return out


def _boundaries_for_uniform(regions, count: int) -> list[dict]:
    """No acoustic labels; equal time bins over identical observed v5 extent."""
    start = regions[0].start_ms
    end = regions[-1].end_ms
    if end <= start:
        raise ValueError('invalid acoustic extent')
    return [{'start_ms': start + (end - start) * i / count,
             'end_ms': start + (end - start) * (i + 1) / count}
            for i in range(count)]


def _boundaries_for_event_grid(regions, count: int) -> list[dict | None]:
    """Ignore candidate labels; group an equal count of event regions per unit."""
    n = len(regions)
    if n < count:
        return [None] * count
    edges = [int((i * n + count // 2) // count) for i in range(count + 1)]
    edges[0], edges[-1] = 0, n
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise AssertionError('nonempty event grid partition expected')
    return [{'start_ms': regions[edges[i]].start_ms,
             'end_ms': regions[edges[i + 1] - 1].end_ms}
            for i in range(count)]


def _permutations(units: list[str], bank: str, wav: str, repeats: int) -> list[list[str]]:
    """Deterministic label-order negative controls, generated before gold exists."""
    if len(units) < 2 or len(set(units)) == 1:
        return []
    seed = int.from_bytes(hashlib.sha256((bank + '\0' + wav).encode()).digest()[:8], 'big') ^ SEED
    rng = random.Random(seed)
    altered = []
    for _ in range(repeats):
        copy = list(units)
        for _try in range(20):
            rng.shuffle(copy)
            if copy != units:
                break
        if copy == units:
            copy = next((units[i:] + units[:i] for i in range(1, len(units))
                         if units[i:] + units[:i] != units), copy)
        if copy != units:
            altered.append(copy)
    return altered


def _candidate_fraction(aligned: dict) -> float:
    spans = aligned.get('units', [])
    return sum(s.get('status') == 'candidate_UNVERIFIED' for s in spans) / len(spans) if spans else 0.0


def freeze(v5: Path, sequences: Path, predictions: Path, output: Path,
           shuffle_repeats: int = 5) -> dict:
    """Create pre-annotation baselines and negative controls, refusing untracked sources."""
    if output.exists():
        raise FileExistsError('freeze output already exists')
    if not (0 <= shuffle_repeats <= 40):
        raise ValueError('shuffle_repeats must be in 0..40')
    v5_data = _v5_bytes(v5)
    seq_data = _json_bytes(sequences)
    pred_data = (predictions / 'alignments.json').read_bytes()
    seq_json = json.loads(seq_data)
    pred_json = json.loads(pred_data)
    if pred_json.get('schema') != V6_SCHEMA:
        raise ValueError('incorrect v6 predictions schema')
    seq = _unique(_parse_sequences(seq_json), 'independent script')
    preds = _unique(pred_json['alignments'], 'v6 predictions')
    regions, hashes = _read_regions(_read_v5_rows(v5))
    if not seq or set(seq) != set(preds):
        raise ValueError('v6 predictions and independently sourced scripts must match exactly')

    outputs = []
    for key in sorted(seq):
        script, rec = seq[key], preds[key]
        if key not in regions or not regions[key]:
            raise ValueError('no v5 acoustic regions for source: ' + str(key))
        sha = _sha(script['source_pcm_sha256'])
        if sha != hashes[key] or sha != _sha(rec['source_pcm_sha256']):
            raise ValueError('PCM hash mismatch across v5/v6/script for ' + str(key))
        labels = script['units']
        if len(labels) != len(rec['units']):
            raise ValueError('v6 prediction length differs from independent script')
        for index, (label, unit) in enumerate(zip(labels, rec['units'])):
            if unit['unit_index'] != index or unit['symbol'] != label:
                raise ValueError('v6 output symbol/index differs from independent script')
        uniform = _boundaries_for_uniform(regions[key], len(labels))
        event_grid = _boundaries_for_event_grid(regions[key], len(labels))
        # Negative controls are *frozen before* independent gold becomes available.
        shuffled = []
        for perm in _permutations(labels, *key, shuffle_repeats):
            altered = align_regions(regions[key], perm)
            shuffled.append({'candidate_fraction': _candidate_fraction(altered),
                             'path_cost': altered.get('path_cost')})
        outputs.append({
            'bank': key[0], 'source_wav': key[1], 'source_pcm_sha256': sha,
            'labels': labels, 'v6_units': rec['units'],
            'baseline_uniform': uniform, 'baseline_region_count': event_grid,
            'observed_extent_ms': [regions[key][0].start_ms, regions[key][-1].end_ms],
            'region_count': len(regions[key]),
            'v6_candidate_fraction': _candidate_fraction(rec),
            'shuffled_order_controls': shuffled,
        })

    bundle = {
        'schema': SCHEMA,
        'phase': 'frozen_before_gold',
        'input_sha256': {'v5_region_csv': _fingerprint(v5_data),
                         'v6_alignments_json': _fingerprint(pred_data),
                         'independent_sequences_json': _fingerprint(seq_data)},
        'baseline_policy': 'same_v5_candidate_extent_for_all; no gold or OTO; labels treated as opaque',
        'shuffle_seed': SEED, 'shuffle_repeats': shuffle_repeats,
        'sources': outputs,
    }
    report = {
        'sources_frozen': len(outputs),
        'units_frozen': sum(len(x['labels']) for x in outputs),
        'sources_with_shuffled_order_controls': sum(bool(x['shuffled_order_controls']) for x in outputs),
        'gold_read_during_freeze': False,
        'ipa_required': False,
        'note': 'Frozen research comparator, not independent real-audio validation.',
    }
    output.mkdir(parents=True)
    frozen_bytes = (json.dumps(bundle, ensure_ascii=False, indent=2) + '\n').encode('utf8')
    (output / 'frozen_bundle.json').write_bytes(frozen_bytes)
    (output / 'freeze_receipt.json').write_text(
        json.dumps({'schema': SCHEMA, 'frozen_bundle_sha256': _fingerprint(frozen_bytes),
                    'instruction': 'Preregister this hash externally before inspecting gold annotations.'},
                   indent=2) + '\n', encoding='utf8')
    report['freeze_bundle_sha256'] = _fingerprint(frozen_bytes)
    (output / 'freeze_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return report


def _interval_errors(pred: dict | None, gold: dict) -> list[float] | None:
    if pred is None:
        return None
    if not (math.isfinite(float(pred['start_ms'])) and math.isfinite(float(pred['end_ms']))
            and 0 <= float(pred['start_ms']) < float(pred['end_ms'])):
        raise ValueError('invalid predicted interval')
    return [abs(float(pred['start_ms']) - float(gold['start_ms'])),
            abs(float(pred['end_ms']) - float(gold['end_ms']))]


def _metrics(rows: list[dict], name: str) -> dict:
    available = [r for r in rows if r[name] is not None]
    endpoint_errors = [e for r in available for e in r[name]]
    within50 = sum(e <= 50 for e in endpoint_errors)
    return {'units_covered': len(available), 'units_total': len(rows),
            'indexed_endpoint_precision_50ms': round(within50 / len(endpoint_errors), 5) if endpoint_errors else None,
            'indexed_endpoint_recall_50ms': round(within50 / (2 * len(rows)), 5) if rows else None,
            'units_both_endpoints_within_50ms': sum(max(r[name]) <= 50 for r in available),
            'coverage': round(len(available) / len(rows), 5) if rows else None,
            'endpoint_mae_ms': round(mean(endpoint_errors), 3) if endpoint_errors else None,
            'endpoint_within_25ms': round(sum(e <= 25 for e in endpoint_errors) / len(endpoint_errors), 5) if endpoint_errors else None,
            'endpoint_within_50ms': round(sum(e <= 50 for e in endpoint_errors) / len(endpoint_errors), 5) if endpoint_errors else None,
            'endpoint_within_100ms': round(sum(e <= 100 for e in endpoint_errors) / len(endpoint_errors), 5) if endpoint_errors else None}


def _paired_source_bootstrap(rows: list[dict], left: str, right: str,
                             iterations: int = 1000) -> dict:
    """Bootstrap entire WAV sources, not individual correlated unit endpoints."""
    paired = [r for r in rows if r[left] is not None and r[right] is not None]
    by_source = defaultdict(list)
    for row in paired:
        by_source[(row['bank'], row['source_wav'])].append(row)
    keys = sorted(by_source)
    if len(keys) < 2:
        return {'paired_units': len(paired), 'distinct_sources': len(keys),
                'mae_difference_ms': round(mean([mean(r[left]) - mean(r[right]) for r in paired]), 3) if paired else None,
                'source_bootstrap_95pct_ci_ms': None}
    rng = random.Random(SEED)
    differences = []
    for _ in range(iterations):
        sample_keys = [rng.choice(keys) for _ in keys]
        sample = [r for key in sample_keys for r in by_source[key]]
        differences.append(mean([mean(r[left]) - mean(r[right]) for r in sample]))
    differences.sort()
    return {
        'paired_units': len(paired), 'distinct_sources': len(keys),
        'mae_difference_ms': round(mean([mean(r[left]) - mean(r[right]) for r in paired]), 3),
        'source_bootstrap_95pct_ci_ms': [round(differences[int(.025 * (iterations - 1))], 3),
                                         round(differences[int(.975 * (iterations - 1))], 3)],
    }


def score(frozen: Path, gold: Path, output: Path) -> dict:
    """Gold is opened only here; never used to select boundaries or tune costs."""
    if output.exists():
        raise FileExistsError('score output already exists')
    frozen_path = frozen / 'frozen_bundle.json' if frozen.is_dir() else frozen
    frozen_data = frozen_path.read_bytes()
    receipt_path = frozen_path.parent / 'freeze_receipt.json'
    if not receipt_path.exists():
        raise ValueError('missing freeze receipt')
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    if receipt.get('schema') != SCHEMA or receipt.get('frozen_bundle_sha256') != _fingerprint(frozen_data):
        raise ValueError('frozen bundle fingerprint differs from preregistration receipt')
    bundle = json.loads(frozen_data)
    gold_json = json.loads(gold.read_text(encoding='utf8'))
    if bundle.get('schema') != SCHEMA or bundle.get('phase') != 'frozen_before_gold':
        raise ValueError('invalid frozen validation bundle')
    if gold_json.get('schema') != GOLD_SCHEMA:
        raise ValueError('incorrect independent gold schema')
    refs = _unique(gold_json['entries'], 'independent gold')
    items = _unique(bundle['sources'], 'frozen sources')
    if not items or set(refs) != set(items):
        raise ValueError('gold must cover exactly the frozen source set (prevent cherry-picking)')

    rows = []
    for key in sorted(items):
        x, g = items[key], refs[key]
        if g.get('provenance') != 'independent_manual_annotation':
            raise ValueError('gold must be independently manually annotated')
        if _sha(g['source_pcm_sha256']) != x['source_pcm_sha256']:
            raise ValueError('gold PCM hash mismatch')
        if len(g['units']) != len(x['labels']):
            raise ValueError('gold unit count mismatch')
        if not (len(x['v6_units']) == len(x['baseline_uniform'])
                == len(x['baseline_region_count']) == len(x['labels'])):
            raise ValueError('frozen unit array size mismatch')
        last_start = -1.0
        for i, (gt, native, uniform, grid, label) in enumerate(zip(
                g['units'], x['v6_units'], x['baseline_uniform'],
                x['baseline_region_count'], x['labels'])):
            if gt['unit_index'] != i or gt['symbol'] != label:
                raise ValueError('gold unit order/labels differ from frozen independent script')
            if native['unit_index'] != i or native['symbol'] != label:
                raise ValueError('frozen prediction order/labels differ from script')
            start, end = float(gt['start_ms']), float(gt['end_ms'])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError('invalid manual gold boundary')
            if start < last_start:
                raise ValueError('manual gold unit order is non-monotone')
            last_start = start
            native_error = (_interval_errors(native, gt)
                            if native['status'] == 'candidate_UNVERIFIED' else None)
            rows.append({'bank': key[0], 'source_wav': key[1], 'unit_index': i,
                         'symbol': label, 'v6': native_error,
                         'uniform': _interval_errors(uniform, gt),
                         'region_count': _interval_errors(grid, gt)})

    v6_rows = [r for r in rows if r['v6'] is not None]
    metrics = {
        'schema': 'phonoweave.user_native_validation_metrics.v0.7',
        'gold_sources': len(items), 'gold_units': len(rows),
        'v6': _metrics(rows, 'v6'),
        'uniform_all': _metrics(rows, 'uniform'),
        'region_count_all': _metrics(rows, 'region_count'),
        'uniform_on_v6_accepted': _metrics(v6_rows, 'uniform'),
        'region_count_on_v6_accepted': _metrics(v6_rows, 'region_count'),
        'paired_v6_minus_uniform': _paired_source_bootstrap(rows, 'v6', 'uniform'),
        'paired_v6_minus_region_count': _paired_source_bootstrap(rows, 'v6', 'region_count'),
        'per_bank': {},
        'shuffled_order_negative_control': {},
        'note': 'Endpoint error and abstention, not validated phone recognition; baselines use observed v5 extent. Frozen controls must precede manual gold.',
    }
    for bank in sorted({r['bank'] for r in rows}):
        part = [r for r in rows if r['bank'] == bank]
        metrics['per_bank'][bank] = {'v6': _metrics(part, 'v6'),
                                    'uniform_on_v6_accepted': _metrics([r for r in part if r['v6'] is not None], 'uniform')}

    originals = [x['v6_candidate_fraction'] for x in items.values() if x['shuffled_order_controls']]
    shuffled = [c['candidate_fraction'] for x in items.values() for c in x['shuffled_order_controls']]
    metrics['shuffled_order_negative_control'] = {
        'eligible_sources': len(originals), 'permutations': len(shuffled),
        'mean_original_candidate_fraction': round(mean(originals), 5) if originals else None,
        'mean_shuffled_candidate_fraction': round(mean(shuffled), 5) if shuffled else None,
        'caution': 'The original and shuffled proportions are only a sensitivity diagnostic; do not treat as accuracy or a significance test.',
    }
    output.mkdir(parents=True)
    (output / 'validation_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    return metrics


def prepare_gold(sequences: Path, output: Path) -> dict:
    """Blank, blind manual annotation template: never reads model predictions."""
    if output.exists():
        raise FileExistsError('annotation template already exists')
    items = _parse_sequences(json.loads(sequences.read_text(encoding='utf8')))
    entries = []
    for item in sorted(items, key=_key):
        entries.append({
            'bank': item['bank'], 'source_wav': item['source_wav'],
            'source_pcm_sha256': _sha(item['source_pcm_sha256']),
            'provenance': 'PENDING_MANUAL_ANNOTATION',
            'units': [{'unit_index': i, 'symbol': s, 'start_ms': None, 'end_ms': None}
                      for i, s in enumerate(item['units'])],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'schema': GOLD_SCHEMA, 'entries': entries},
                                 ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return {'sources': len(entries), 'blank_units': sum(len(x['units']) for x in entries),
            'model_predictions_read': False,
            'instruction': 'Fill onset/offset milliseconds manually, then change provenance to independent_manual_annotation.'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='phonoweave-native-validation-v7')
    cmds = parser.add_subparsers(dest='command', required=True)
    f = cmds.add_parser('freeze', help='create baseline/negative-control bundle before gold labels')
    f.add_argument('--v5', type=Path, required=True)
    f.add_argument('--sequences', type=Path, required=True)
    f.add_argument('--predictions', type=Path, required=True)
    f.add_argument('--output', type=Path, required=True)
    f.add_argument('--shuffles', type=int, default=5)
    t = cmds.add_parser('prepare-gold', help='create a BLIND blank manual annotation template')
    t.add_argument('--sequences', type=Path, required=True)
    t.add_argument('--output', type=Path, required=True)
    s = cmds.add_parser('score', help='score frozen bundle against independent gold')
    s.add_argument('--frozen', type=Path, required=True)
    s.add_argument('--gold', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        result = freeze(args.v5, args.sequences, args.predictions, args.output, args.shuffles)
    elif args.command == 'prepare-gold':
        result = prepare_gold(args.sequences, args.output)
    else:
        result = score(args.frozen, args.gold, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())