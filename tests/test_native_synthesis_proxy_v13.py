"""v13 synthetic invariants. Never confuse proxy spectral MSE with song quality."""
import json
import hashlib
import copy
from pathlib import Path

import numpy as np
import pytest

from phonoweave.native_synthesis_proxy_v13 import (
    build_clips, freeze_partitions, stage_normalize, rank_donors, donor_pool,
    score, permutation_context, select_on_validation, verify_v12, run,
)


def fixtures(n_groups=20,n_banks=3,correlated=True):
    rng=np.random.default_rng(1007)
    banks=['lu_B4','mang_D4','mang_G3'][:n_banks]
    rows=[];vectors=[]
    for gi in range(n_groups):
        group=f'carrier_{gi:02d}'
        latent=float(gi%4)
        for bank in banks:
            pcm=hashlib.sha256(f'{group}.{bank}'.encode()).hexdigest()
            n=rng.normal(0,.08,(3,48));stage=n.copy()
            stage[0,0]=latent + n[0,0]
            stage[1,0]=latent + n[1,0]
            stage[2,0]=latent + n[2,0] if correlated else rng.normal(0,.5)
            stage[:,:,] += (0.5 if bank.startswith('mang') else 0.)
            wav=f'{bank}/{group}.wav'
            for j in range(3):
                rows.append({'bank':bank,'clip_group':group,'source_pcm_sha256':pcm,
                             'wav_archive_entry':wav,'region_index':j,
                             'start_ms':j*100,'end_ms':(j+1)*100})
                vectors.append(stage[j])
    return rows,np.stack(vectors)


def prepared():
    rows,X=fixtures(); clips,_=build_clips(rows,X)
    splits=freeze_partitions(clips)
    Z,_=stage_normalize(clips,splits['train'])
    return clips,splits,Z


def test_clips_and_provenance():
    rows,X=fixtures()
    clips,report=build_clips(rows,X)
    assert report['complete_three_stage_clips']==60
    assert report['incomplete_clips_skipped']==0
    assert clips[0].stages.shape==(3,48)


def test_incomplete_source_is_not_invented():
    rows,X=fixtures()
    del rows[1]; X=np.delete(X,1,axis=0)
    clips,report=build_clips(rows,X)
    assert report['incomplete_clips_skipped']==1 and len(clips)==59


def test_duplicate_stage_rejected():
    rows,X=fixtures();rows.append(dict(rows[0]));X=np.vstack([X,X[0]])
    with pytest.raises(ValueError,match='duplicate region'):
        build_clips(rows,X)


def test_conflicting_source_hash_rejected():
    rows,X=fixtures();rows[1]=dict(rows[1],source_pcm_sha256='f'*64)
    with pytest.raises(ValueError,match='conflicting PCM hashes'):
        build_clips(rows,X)


def test_group_not_in_two_folds():
    clips,splits,Z=prepared()
    groups=[{c.group for c,yes in zip(clips,splits[part]) if yes} for part in ('train','validation','audit')]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])


def test_target_third_region_must_not_affect_donor_choice():
    clips,splits,Z=prepared()
    i=int(np.flatnonzero(splits['audit'])[0])
    first=rank_donors(Z,clips,splits['train'],i,'preceding_both',3)
    tampered=Z.copy();tampered[i,2,:]+=10000
    second=rank_donors(tampered,clips,splits['train'],i,'preceding_both',3)
    assert first.tolist()==second.tolist()
    a=rank_donors(Z,clips,splits['train'],i,'pooled',3)
    b=rank_donors(tampered,clips,splits['train'],i,'pooled',3)
    assert a.tolist()==b.tolist()


def test_exact_same_donor_budget_and_source_disjoint():
    clips,splits,Z=prepared();i=int(np.flatnonzero(splits['audit'])[0])
    for k in (1,2,3,4,6):
        for mode in ('pooled','preceding_0','preceding_1','preceding_both'):
            ids=rank_donors(Z,clips,splits['train'],i,mode,k)
            assert len(ids)==k
            assert all(splits['train'][d] and clips[d].group!=clips[i].group for d in ids)


def test_source_pool_refuses_insufficient_donors():
    clips,splits,Z=prepared()
    i=int(np.flatnonzero(splits['audit'])[0]);train=np.zeros(len(clips),bool)
    with pytest.raises(ValueError,match='insufficient'):
        rank_donors(Z,clips,train,i,'pooled',1)


def test_strict_permutation_is_derangement_within_bank():
    clips,splits,Z=prepared()
    shuffled=permutation_context(clips,Z,splits['validation'],14)
    for i,context in shuffled.items():
        assert not np.array_equal(context,Z[i,:2])
        assert any(c.bank==clips[i].bank and splits['validation'][j] and j!=i and
                   np.array_equal(context,Z[j,:2]) for j,c in enumerate(clips))


def test_model_selection_ignores_audit_and_frozen_test_target():
    clips,splits,Z=prepared()
    a=select_on_validation(clips,Z,splits,null_trials=3)
    moved=Z.copy();moved[splits['audit'],2,:]+=500
    b=select_on_validation(clips,moved,splits,null_trials=3)
    assert a['candidate']==b['candidate']
    assert a['development_screen_pass']==b['development_screen_pass']


def test_metric_uses_independent_groups_not_7_banks_as_seven_speakers():
    clips,splits,Z=prepared()
    s=score(clips,Z,splits['train'],splits['audit'],'pooled',2)
    assert s['source_groups_scored']==len({c.group for c,yes in zip(clips,splits['audit']) if yes})
    assert s['regions_scored']>s['source_groups_scored']


def test_v12_sha_contract(tmp_path):
    source={'archive_sha256':'1'*64};file=tmp_path/'v12.json'
    file.write_text(json.dumps({'schema':'phonoweave.paired_bank_context_decomposition.v1.2','source_archive_sha256':'1'*64}))
    assert verify_v12(source,file)['v12_source_identity_verified']
    file.write_text(json.dumps({'schema':'phonoweave.paired_bank_context_decomposition.v1.2','source_archive_sha256':'2'*64}))
    with pytest.raises(ValueError,match='SHA256 mismatch'):verify_v12(source,file)


def test_real_source_frozen_output_contract(tmp_path):
    rows,X=fixtures()
    v9=tmp_path/'v9';v9.mkdir()
    (v9/'summary.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9',
       'regions':len(rows),'archive_sha256':'a'*64}))
    (v9/'motif_regions.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':rows}))
    np.savez_compressed(v9/'raw_spectral_embeddings.npz',vectors=X)
    output=tmp_path/'out';report=run(v9,output,null_trials=3)
    assert report['automatic_unit_split_merge_authorized'] is False
    assert report['no_actual_rendered_synthesis_or_listener_test'] is True
    assert report['source_groups_by_split']['audit']>=3
    assert (output/'audit_donor_comparison.csv').exists()
    with pytest.raises(FileExistsError):run(v9,output,null_trials=3)


def test_selection_permutation_null_not_probability():
    clips,splits,Z=prepared();report=select_on_validation(clips,Z,splits,null_trials=3)
    assert report['permutation_null_trials']==3
    assert 0<report['max_across_modes_and_budgets_permutation_p_exploratory']<=1
    assert report['selection_used_audit'] is False