"""Leak guards, fair comparisons, negative controls and blank annotation protocol."""
import csv
import json
from pathlib import Path

import pytest

from phonoweave.native_validation_v7 import (
    SCHEMA, _boundaries_for_event_grid, _permutations,
    freeze, score, prepare_gold,
)
from phonoweave.native_sequence_alignment import Region, SCHEMA as V6_SCHEMA, GOLD_SCHEMA


def fixture_dataset(tmp_path):
    v5 = tmp_path / 'v5'
    v5.mkdir()
    sequence_path = tmp_path / 'sequences.json'
    prediction_dir = tmp_path / 'v6'
    prediction_dir.mkdir()
    entries = []
    records = []
    with (v5/'candidate_segment_unit_hints.csv').open('w', encoding='utf8', newline='') as f:
        wr=csv.DictWriter(f, fieldnames=['bank','source_wav','source_pcm_sha256','start_ms','end_ms','candidate_unit','alternatives','status'])
        wr.writeheader()
        for bank, name, sha, offset in [('s_C4','a.wav','a'*64,0),('d_C4','b.wav','b'*64,10)]:
            units=['i','a']
            for start,end,lab in [(0,100,'i'),(100,150,'a'),(150,300,'a')]:
                wr.writerow({'bank':bank,'source_wav':name,'source_pcm_sha256':sha,
                             'start_ms':start+offset,'end_ms':end+offset,
                             'candidate_unit':lab,'alternatives':lab,'status':'candidate_UNVERIFIED_not_a_phone'})
            entries.append({'bank':bank, 'source_wav':name,'source_pcm_sha256':sha,
                            'provenance':'independent_recording_script','units':units})
            records.append({'bank':bank,'source_wav':name,'source_pcm_sha256':sha,
                            'units':[{'unit_index':0,'symbol':'i',
                                      'start_ms':offset,'end_ms':100+offset,'status':'candidate_UNVERIFIED'},
                                     {'unit_index':1,'symbol':'a',
                                      'start_ms':100+offset,'end_ms':300+offset,
                                      'status':'candidate_UNVERIFIED' if name=='a.wav' else 'abstain_ambiguous_alignment'}]})
    sequence_path.write_text(json.dumps({'schema':'phonoweave.native_sequences.v1','entries':entries},ensure_ascii=False))
    (prediction_dir/'alignments.json').write_text(json.dumps({'schema':V6_SCHEMA,'alignments':records},ensure_ascii=False))
    return v5, sequence_path, prediction_dir


def gold_from_script(sequences, gold_path, include=None):
    data=json.loads(sequences.read_text())
    items=[]
    for e in data['entries']:
        if include is not None and e['source_wav'] not in include:
            continue
        offset=10 if e['source_wav']=='b.wav' else 0
        items.append({'bank':e['bank'], 'source_wav':e['source_wav'],
                      'source_pcm_sha256':e['source_pcm_sha256'],
                      'provenance':'independent_manual_annotation',
                      'units':[{'unit_index':0,'symbol':'i','start_ms':offset+5,'end_ms':offset+95},
                               {'unit_index':1,'symbol':'a','start_ms':offset+102,'end_ms':offset+295}]})
    gold_path.write_text(json.dumps({'schema':GOLD_SCHEMA,'entries':items},ensure_ascii=False))
    return gold_path


def test_blank_gold_has_no_generated_boundaries(tmp_path):
    _,seq,_=fixture_dataset(tmp_path)
    out=tmp_path/'blank.json'
    summary=prepare_gold(seq,out)
    assert summary['model_predictions_read'] is False
    x=json.loads(out.read_text())
    assert all(u['start_ms'] is None and u['end_ms'] is None for e in x['entries'] for u in e['units'])
    assert all(e['provenance']=='PENDING_MANUAL_ANNOTATION' for e in x['entries'])
    with pytest.raises(FileExistsError):
        prepare_gold(seq,out)


def test_frozen_baselines_and_independent_scoring(tmp_path):
    v5,seq,pred=fixture_dataset(tmp_path)
    freeze_info=freeze(v5,seq,pred,tmp_path/'frozen',shuffle_repeats=5)
    assert freeze_info['sources_frozen']==2
    assert freeze_info['gold_read_during_freeze'] is False
    frozen=json.loads((tmp_path/'frozen'/'frozen_bundle.json').read_text())
    assert frozen['schema']==SCHEMA
    src = next(s for s in frozen['sources'] if s['bank'] == 's_C4')
    assert src['baseline_uniform'][0]['end_ms']==150
    assert src['baseline_region_count'][0]['end_ms'] == 150
    assert src['baseline_region_count'][1]['start_ms'] == 150
    assert all(e.get('start_ms') is not None for s in frozen['sources'] for e in s['baseline_uniform'])
    assert len(frozen['sources'][0]['shuffled_order_controls']) == 5
    gold=gold_from_script(seq,tmp_path/'gold.json')
    report=score(tmp_path/'frozen',gold,tmp_path/'score')
    assert report['gold_sources']==2 and report['gold_units']==4
    assert report['v6']['units_covered']==3
    assert report['v6']['units_total']==4
    assert report['v6']['coverage']==.75
    assert report['uniform_on_v6_accepted']['units_covered']==3
    assert report['paired_v6_minus_uniform']['distinct_sources']==2
    assert len(report['paired_v6_minus_uniform']['source_bootstrap_95pct_ci_ms'])==2
    assert len(report['per_bank'])==2
    assert report['v6']['endpoint_mae_ms']==4.5


def test_event_grid_abstains_when_too_few_regions():
    regions=[Region(0,100,'a',(),''),Region(100,200,'b',(), '')]
    assert _boundaries_for_event_grid(regions,3)==[None,None,None]


def test_deterministic_shuffled_controls():
    x=_permutations(['i0','a','i2'],'s_C4','x.wav',5)
    assert x == _permutations(['i0','a','i2'],'s_C4','x.wav',5)
    assert all(p!=['i0','a','i2'] for p in x)
    assert _permutations(['i','i'],'s_C4','x.wav',5)==[]


def test_reject_changed_gold_subselection(tmp_path):
    v5,seq,pred=fixture_dataset(tmp_path)
    freeze(v5,seq,pred,tmp_path/'frozen')
    gold=gold_from_script(seq,tmp_path/'gold.json',include=['a.wav'])
    with pytest.raises(ValueError, match='exactly'):
        score(tmp_path/'frozen',gold,tmp_path/'score')
    assert not (tmp_path/'score').exists()


def test_reject_hash_mismatch_and_no_partial_freeze(tmp_path):
    v5,seq,pred=fixture_dataset(tmp_path)
    doc=json.loads(seq.read_text())
    doc['entries'][0]['source_pcm_sha256']='f'*64
    seq.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='PCM hash'):
        freeze(v5,seq,pred,tmp_path/'frozen')
    assert not (tmp_path/'frozen').exists()


def test_reject_changed_symbol_at_freeze(tmp_path):
    v5,seq,pred=fixture_dataset(tmp_path)
    doc=json.loads(seq.read_text())
    doc['entries'][0]['units'][1]='i2'
    seq.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='symbol/index'):
        freeze(v5,seq,pred,tmp_path/'frozen')


def test_reject_receipt_tampering(tmp_path):
    v5,seq,pred=fixture_dataset(tmp_path)
    freeze(v5,seq,pred,tmp_path/'frozen')
    path=tmp_path/'frozen'/'frozen_bundle.json'
    data=json.loads(path.read_text())
    data['sources'][0]['v6_units'][0]['start_ms']=500
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='fingerprint'):
        score(tmp_path/'frozen',gold_from_script(seq,tmp_path/'gold.json'),tmp_path/'scored')


def test_gold_must_be_completed_by_manual_annotator(tmp_path):
    v5,seq,pred=fixture_dataset(tmp_path)
    freeze(v5,seq,pred,tmp_path/'frozen')
    prepare_gold(seq,tmp_path/'gold.json')
    with pytest.raises(ValueError, match='manually'):
        score(tmp_path/'frozen',tmp_path/'gold.json',tmp_path/'scored')