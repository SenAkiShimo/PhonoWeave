"""Synthetic invariants; not a validated test of phone boundaries."""
import csv
import json
from pathlib import Path

import pytest

from phonoweave.native_sequence_alignment import (
    Region, align_regions, run, evaluate, _read_regions, SEQUENCE_SCHEMA, GOLD_SCHEMA,
)

SHA = '9' * 64


def reg(a, b, label='', alts=(), status='candidate_UNVERIFIED_not_a_phone'):
    return Region(float(a), float(b), label, tuple(alts), status)


def test_one_unit_can_cover_multiple_acoustic_regions():
    x = align_regions([reg(0, 80, 'i'), reg(80, 160, 'i'), reg(160, 230, 'a')], ['i', 'a'])
    assert [u['symbol'] for u in x['units']] == ['i', 'a']
    assert x['units'][0]['end_ms'] == 160
    assert x['status'] == 'candidate_UNVERIFIED'


def test_ignores_nonphone_acoustic_change():
    x = align_regions([reg(0, 80, 'i'), reg(80, 100, 'ghost'), reg(100, 190, 'a')], ['i', 'a'])
    assert [v['symbol'] for v in x['units']] == ['i', 'a']
    assert x['units'][0]['start_ms'] == 0
    assert x['units'][1]['end_ms'] == 190
    assert x['status'] != 'candidate_UNVERIFIED'  # tied grouping/skip paths


def test_shared_acoustic_region_never_invents_middle_boundary():
    x = align_regions([reg(10, 110, 'i')], ['i', 'i2'])
    assert x['status'] == 'abstain_or_shared_unresolved'
    assert x['units'][0]['start_ms'] == x['units'][1]['start_ms']
    assert x['units'][0]['end_ms'] == x['units'][1]['end_ms']
    assert 'shared' in x['units'][0]['status']


def test_symbol_opaque_and_order_not_reordered():
    x = align_regions([reg(0, 80, 'i0'), reg(80, 160, '自定义'), reg(160, 240, 'i2')],
                      ['i0', '自定义', 'i2'])
    assert [r['symbol'] for r in x['units']] == ['i0', '自定义', 'i2']


def test_absent_label_abstains_instead_of_forcing_phone():
    x = align_regions([reg(0, 100, 'a')], ['无证据'])
    assert x['status'] != 'candidate_UNVERIFIED'
    assert x['units'][0]['status'] == 'abstain_no_support_for_label'


def test_missing_input_abstains():
    assert align_regions([], ['i'])['status'].startswith('abstain')


def rows(tmp_path: Path, entries):
    f = tmp_path / 'v5'
    f.mkdir()
    path = f / 'candidate_segment_unit_hints.csv'
    fields = ['bank', 'source_wav', 'source_pcm_sha256', 'segment_num',
              'start_ms', 'end_ms', 'candidate_unit', 'alternatives', 'status']
    with path.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for idx, (a, b, label) in enumerate(entries, 1):
            writer.writerow(dict(bank='s_C4', source_wav='test.wav', source_pcm_sha256=SHA,
                                 segment_num=idx, start_ms=a, end_ms=b,
                                 candidate_unit=label, alternatives=label,
                                 status='candidate_UNVERIFIED_not_a_phone'))
    return f


def seq(tmp_path, sha=SHA, labels=('i', 'a')):
    p = tmp_path / 'sequences.json'
    p.write_text(json.dumps({'schema': SEQUENCE_SCHEMA, 'entries': [
        {'bank': 's_C4', 'source_wav': 'test.wav', 'source_pcm_sha256': sha,
         'provenance': 'independent_recording_script', 'units': list(labels)}]}), encoding='utf8')
    return p


def test_full_inference_and_separate_evaluation(tmp_path):
    v5 = rows(tmp_path, [(0, 100, 'i'), (100, 200, 'a')])
    sequences = seq(tmp_path)
    pred = tmp_path / 'pred'
    result = run(v5, sequences, pred)
    assert result['candidates'] == 1
    assert result['gold_annotations_consumed_by_inference'] is False
    gold = tmp_path / 'gold.json'
    gold.write_text(json.dumps({'schema': GOLD_SCHEMA, 'entries': [
        {'bank': 's_C4', 'source_wav': 'test.wav', 'source_pcm_sha256': SHA,
         'provenance': 'independent_manual_annotation', 'units': [
             {'unit_index': 0, 'symbol': 'i', 'start_ms': 4, 'end_ms': 104},
             {'unit_index': 1, 'symbol': 'a', 'start_ms': 98, 'end_ms': 203}]}]}), encoding='utf8')
    report = evaluate(pred, gold, tmp_path / 'eval')
    assert report['gold_units'] == 2
    assert report['endpoint_mae_ms_on_nonabstentions'] is not None
    # Inference output was fixed before gold was opened.
    assert json.loads((pred / 'alignments.json').read_text())['alignments'][0]['units'][0]['start_ms'] == 0


def test_rejects_hash_mismatch_and_creates_no_output(tmp_path):
    v5 = rows(tmp_path, [(0, 100, 'i'), (100, 200, 'a')])
    with pytest.raises(ValueError, match='PCM hash'):
        run(v5, seq(tmp_path, sha='a'*64), tmp_path / 'pred')
    assert not (tmp_path / 'pred').exists()


def test_rejects_conflicting_hashes_and_overlapping_regions():
    base = {'bank': 's_C4', 'source_wav': 'a.wav', 'source_pcm_sha256': SHA,
            'start_ms': 0, 'end_ms': 90, 'candidate_unit': 'i', 'status': 'candidate_UNVERIFIED_not_a_phone'}
    with pytest.raises(ValueError, match='conflicting'):
        _read_regions([base, dict(base, source_pcm_sha256='2'*64)])
    with pytest.raises(ValueError, match='overlapping'):
        _read_regions([base, dict(base, start_ms=50, end_ms=110)])


def test_duplicate_sequence_and_wrong_provenance_rejected(tmp_path):
    v5 = rows(tmp_path, [(0, 100, 'i')])
    sequence = seq(tmp_path, labels=('i',))
    data = json.loads(sequence.read_text())
    data['entries'][0]['provenance'] = 'oto_ini'
    sequence.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='independent script'):
        run(v5, sequence, tmp_path / 'out')


def test_reject_gold_mismatch(tmp_path):
    v5 = rows(tmp_path, [(0, 100, 'i')])
    pred = tmp_path / 'pred'
    run(v5, seq(tmp_path, labels=('i',)), pred)
    gold = tmp_path / 'gold.json'
    gold.write_text(json.dumps({'schema': GOLD_SCHEMA, 'entries': [
        {'bank': 's_C4', 'source_wav': 'test.wav', 'source_pcm_sha256': SHA,
         'provenance': 'independent_manual_annotation', 'units': [
             {'unit_index': 0, 'symbol': 'changed', 'start_ms': 0, 'end_ms': 100}]}]}))
    with pytest.raises(ValueError, match='gold labels/order'):
        evaluate(pred, gold, tmp_path / 'eval')
