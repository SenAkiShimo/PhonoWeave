"""v12 synthetic invariants: acoustics, not phones or recording scheme approvals."""
import json
from pathlib import Path

import numpy as np
import pytest

from phonoweave.native_nuisance_v12 import (
    fit_offsets, transform, retrieval, relative_gain, phase_audit, validate_source,
    within_bank_geometry_preserved, run, SCHEMA,
)


def make_rows(seed=3, group_count=32, bank_effect=True, phases=(0,)):
    rng=np.random.default_rng(seed)
    banks=['lu_B4','lu_C4','lu_F4','mang_D4','mang_D5','mang_G3','mang_G4']
    rows=[];x=[]
    offsets={b:(rng.normal(scale=1.6,size=48) if bank_effect else np.zeros(48)) for b in banks}
    for g in range(group_count):
        carrier=rng.normal(scale=1.2,size=48)
        for bank in banks:
            for phase in phases:
                # style/bank effect and the carrier variation are independent synthetic causes
                rows.append({'bank':bank,'clip_group':f'carrier{g:03d}',
                             'wav_archive_entry':f'{bank}/carrier{g:03d}.wav',
                             'source_pcm_sha256':f'{g*7+banks.index(bank)+1:064x}',
                             'region_index':phase,'start_ms':float(phase*100),
                             'end_ms':float(phase*100+80)})
                x.append(carrier+offsets[bank]+rng.normal(scale=0.07,size=48)+phase*0.05)
    return rows,np.asarray(x,dtype=np.float64)


def test_paired_fit_only_training_and_reproducible():
    rows,x=make_rows()
    train=np.array([int(r['clip_group'][-3:])<22 for r in rows])
    a,s=fit_offsets(rows,x,train)
    b,t=fit_offsets(rows,x.copy(),train)
    assert set(a)==set(b)
    assert s['paired_training_carriers']==22
    assert all(np.allclose(a[k],b[k]) for k in a)
    xa=x.copy();xa[~train]+=100000
    c,_=fit_offsets(rows,xa,train)
    assert all(np.allclose(a[k],c[k]) for k in a)


def test_bank_translation_keeps_within_bank_geometry_exact():
    rows,x=make_rows()
    train=np.ones(len(rows),bool)
    offsets,_=fit_offsets(rows,x,train)
    y=transform(x,rows,offsets,0.75)
    assert within_bank_geometry_preserved(rows,x,y,np.ones(len(rows),bool))<1e-8
    assert not np.allclose(x,y)


def test_actual_signal_can_improve_cross_style_carrier_retrieval():
    rows,x=make_rows()
    tr=np.array([int(r['clip_group'][-3:])<20 for r in rows])
    va=np.array([20<=int(r['clip_group'][-3:])<27 for r in rows])
    offsets,_=fit_offsets(rows,x,tr)
    base=retrieval(rows,x,va);corrected=retrieval(rows,transform(x,rows,offsets,1.),va)
    assert corrected['queries']>30
    assert corrected['top1_fraction']>=base['top1_fraction']
    assert corrected['matched_to_distractor_distance_ratio']<base['matched_to_distractor_distance_ratio']


def test_no_bank_signal_does_not_fabricate_large_paired_effect():
    rows,x=make_rows(bank_effect=False)
    tr=np.array([int(r['clip_group'][-3:])<20 for r in rows])
    va=np.array([20<=int(r['clip_group'][-3:])<27 for r in rows])
    offsets,_=fit_offsets(rows,x,tr)
    base=retrieval(rows,x,va)
    corrected=retrieval(rows,transform(x,rows,offsets,1.),va)
    gain=relative_gain(base,corrected)
    assert gain['top1_gain']<.03
    assert gain['ratio_gain']<.03


def test_permuted_bank_identity_cannot_reliably_recover_real_offsets():
    rows,x=make_rows()
    tr=np.array([int(r['clip_group'][-3:])<20 for r in rows])
    va=np.array([20<=int(r['clip_group'][-3:])<27 for r in rows])
    exact,_=fit_offsets(rows,x,tr)
    shuffled,_=fit_offsets(rows,x,tr,shuffle_seed=99)
    good=retrieval(rows,transform(x,rows,exact,1.),va)
    wrong=retrieval(rows,transform(x,rows,shuffled,1.),va)
    assert good['matched_to_distractor_distance_ratio'] < wrong['matched_to_distractor_distance_ratio']


def test_phase_audit_does_not_use_audit_to_select_strength_or_threshold():
    rows,x=make_rows(group_count=30)
    original=phase_audit(rows,x,0,rows,x)[0]
    # We compare explicitly retained development properties only; audit description may change.
    from phonoweave.native_motif_link_v10 import group_split
    _,_,aud=group_split(rows)
    altered=x.copy();altered[aud]+=1000
    changed=phase_audit(rows,altered,0,rows,altered)[0]
    assert original['chosen_strength_diagnostic']==changed['chosen_strength_diagnostic']
    assert original['permutation_p_exploratory_not_phone_significance']==changed['permutation_p_exploratory_not_phone_significance']
    assert original['validation_strength_search']==changed['validation_strength_search']


def test_no_audio_or_phone_symbols_used_in_nuisance_fit():
    rows,x=make_rows()
    tr=np.array([int(r['clip_group'][-3:])<20 for r in rows])
    a,_=fit_offsets(rows,x,tr)
    alias=[dict(r,oto_alias='sh',ipa='ʂ',phoneme='test') for r in rows]
    b,_=fit_offsets(alias,x,tr)
    assert all(np.array_equal(a[k],b[k]) for k in a)


def test_provenance_rejects_duplicate_identical_pcm_in_multiple_families():
    rows,x=make_rows(group_count=12)
    altered=[dict(r) for r in rows]
    altered[-1]['source_pcm_sha256']=rows[0]['source_pcm_sha256']
    with pytest.raises(ValueError,match='multiple source families'):
        validate_source(altered,x)


def test_invalid_source_and_malformed_audio_rejected():
    rows,x=make_rows(group_count=12)
    with pytest.raises(ValueError,match='aligned 48D'):
        validate_source(rows,x[:10])
    with pytest.raises(ValueError,match='SHA256'):
        validate_source([dict(rows[0],source_pcm_sha256='malformed')],x[:1])
    with pytest.raises(ValueError,match='interval'):
        validate_source([dict(rows[0],end_ms=-10.)],x[:1])


def test_abstains_for_one_style_or_insufficient_training_support():
    rows,x=make_rows(group_count=12)
    idx=[i for i,r in enumerate(rows) if r['bank'].startswith('lu_')]
    with pytest.raises(ValueError,match='multi-style'):
        fit_offsets([rows[i] for i in idx],x[idx],np.ones(len(idx),bool))
    with pytest.raises(ValueError,match='coverage'):
        fit_offsets(rows,x,np.array([int(r['clip_group'][-3:])<3 for r in rows]))


def test_no_clobbering_or_input_source_confusion(tmp_path):
    dest=tmp_path/'output';dest.mkdir()
    with pytest.raises(FileExistsError): run(None,dest)
    with pytest.raises(ValueError,match='exactly one'):
        run(None,tmp_path/'a')
    with pytest.raises(ValueError,match='exactly one'):
        run(tmp_path/'a',tmp_path/'b',full_wav_zip=tmp_path/'c')
    with pytest.raises(ValueError,match='historical'):
        run(None,tmp_path/'b',v10_results=tmp_path/'v10',full_wav_zip=tmp_path/'full.wav.zip')


def test_small_v9_artifact_writes_research_only_without_phone_claim(tmp_path):
    rows,x=make_rows(group_count=13)
    src=tmp_path/'v9';src.mkdir()
    (src/'motif_regions.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':rows}))
    (src/'summary.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':len(rows),'archive_sha256':'f'*64}))
    np.savez_compressed(src/'raw_spectral_embeddings.npz',vectors=x.astype('float32'))
    out=tmp_path/'v12';result=run(src,out)
    assert result['schema']==SCHEMA
    assert result['confirmed_phone_identity_or_boundary'] is False
    assert result['recording_inventory_split_merge_or_auto_labels'] is False
    assert result['context_residualization_deployed'] is False
    assert (out/'research_bank_offsets.json').exists()


def test_full_wav_zip_can_run_in_no_script_audio_only_mode(tmp_path):
    import io, wave, zipfile
    archive=tmp_path/'test_full.wavs.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for bank,freq in [('lu_B4',220),('mang_D4',410),('lu_C4',330)]:
            # Separate source families, no filenames interpreted as user phones.
            name=f'{bank}/new_carrier_{bank}.wav';rate=16000
            y=np.sin(np.arange(rate*2)*2*np.pi*freq/rate)*.2
            with io.BytesIO() as buf:
                with wave.open(buf,'wb') as wf:
                    wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(rate)
                    wf.writeframes(np.asarray(y*32000,dtype='<i2').tobytes())
                z.writestr(name,buf.getvalue())
    report=run(None,tmp_path/'res',full_wav_zip=archive)
    assert report['source_provenance']=='unanchored_full_wav_user_supplied_archive'
    assert report['regions_total']>=3
    assert report['candidate_bank_corrections_passing_development_gate']==0
    assert report['recording_inventory_split_merge_or_auto_labels'] is False


def test_frozen_v10_pcm_identity_guard_rejects_changed_region(tmp_path):
    rows,x=make_rows(group_count=14)
    src=tmp_path/'v9';src.mkdir()
    (src/'motif_regions.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':rows}))
    (src/'summary.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':len(rows),'archive_sha256':'f'*64}))
    np.savez_compressed(src/'raw_spectral_embeddings.npz',vectors=x.astype('float32'))
    v10=tmp_path/'v10';v10.mkdir()
    corrupted=[dict(r) for r in rows]
    corrupted[0]['end_ms']+=10
    (v10/'regions.json').write_text(json.dumps({'schema':'phonoweave.cross_recording_acoustic_motifs.v1.0','regions':corrupted}))
    (v10/'summary.json').write_text(json.dumps({'schema':'phonoweave.cross_recording_acoustic_motifs.v1.0','regions_total':len(rows),'input_source_zip_sha256':'f'*64}))
    with pytest.raises(ValueError,match='identity mismatch'):
        run(src,tmp_path/'out',v10_results=v10)
    assert not (tmp_path/'out').exists()
