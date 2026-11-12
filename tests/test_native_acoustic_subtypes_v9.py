"""Synthetic, non-phonological invariants for v9 spectral-context clustering."""
from __future__ import annotations

import numpy as np
import pytest

from phonoweave.native_acoustic_subtypes_v9 import (
    FROZEN_HELDOUT, adjusted_rand_index, assign, bank_normalization, discover,
    kmeans2, normalized_mutual_information,
)


def make_dataset(two_modes=True):
    rng=np.random.default_rng(14)
    rows=[];xs=[]
    # 6 templates in development, 2 in frozen heldout, 7 banks, 6 repeats
    banks=['mang_D4','mang_G3','mang_G4','mang_D5','lu_C4','lu_B4','lu_F4']
    for bank_idx, bank in enumerate(banks):
        for wav in ['vcsection1.wav','vcsection3.wav','vcsection4.wav','vcsection5.wav','additional.wav','additional2.wav','a.wav','ai.wav']:
            for j in range(6):
                group=j%2
                v=rng.normal(0,.12,36)
                if two_modes: v[0]+=(-1 if group==0 else 1)*1.5
                v[1]+=bank_idx*.07
                rows.append({'bank':bank,'source_wav':wav,'source_pcm_sha256':'a'*64,
                             'symbol':'自定义','context':f'ctx{j//2}', 'oto_line':str(j)})
                xs.append(v)
    return rows,np.array(xs)


def test_adjusted_rand_index_invariant_under_cluster_permutation():
    a=np.array([0,0,1,1,0,1]);b=1-a
    assert adjusted_rand_index(a,b)==pytest.approx(1.)
    assert adjusted_rand_index(a,np.array([0,1,0,1,0,1]))<1


def test_kmeans_finds_clear_multimodal_synthetic_acoustics():
    rows,X=make_dataset()
    Y,dev,_=bank_normalization(rows,X,FROZEN_HELDOUT)
    centers,y,loss=kmeans2(Y[dev],77)
    assert 0.30 < np.mean(y) < 0.70
    assert loss < np.sum((Y[dev]-Y[dev].mean(axis=0))**2)
    summary,memberships,report=discover(rows,X)
    assert report['ipa_used'] is False
    assert summary[0]['style_heldout_ari']>0.8
    assert summary[0]['status']=='exploratory_repeatable_spectral_context_pattern'
    assert summary[0]['phone_boundary_verified'] is False
    assert not any(m['status']=='verified_phone' for m in memberships)


def test_short_or_missing_heldout_avoids_claim():
    rows,X=make_dataset()
    mask=np.array([r['source_wav'] not in FROZEN_HELDOUT for r in rows])
    rows=[r for r, keep in zip(rows,mask) if keep]
    X=X[mask]
    summary,mem,report=discover(rows,X)
    assert summary[0]['status']=='insufficient_data_or_reproducibility'
    assert summary[0]['reason']=='insufficient_dev_or_frozen_heldout'
    assert not mem


def test_normalization_ignores_frozen_heldout_values():
    rows,X=make_dataset()
    Y,dev,meta=bank_normalization(rows,X,FROZEN_HELDOUT)
    Xt=X.copy();Xt[~dev]+=10000
    Yt,devt,_=bank_normalization(rows,Xt,FROZEN_HELDOUT)
    assert np.allclose(Y[dev],Yt[dev])
    assert meta['training_only']


def test_metadata_nmi_detects_known_context_confound():
    group=['a','a','b','b','a','b']
    assert normalized_mutual_information(group,np.array([0,0,1,1,0,1]))==pytest.approx(1.)
    assert normalized_mutual_information(['same']*6,np.array([0,0,1,1,0,1]))==pytest.approx(0.)


def test_full_procedure_label_rename_does_not_map_to_ipa():
    rows,X=make_dataset()
    for r in rows:r['symbol']='任意☁️'
    summary,_,_=discover(rows,X)
    assert summary[0]['symbol']=='任意☁️'


def test_constant_audio_features_abstain():
    with pytest.raises(ValueError,match='constant'):
        kmeans2(np.zeros((30,36)),1)


def test_wrong_length_rejected():
    rows,X=make_dataset()
    with pytest.raises(ValueError,match='count'):
        discover(rows,X[:-2])


def test_kmeans_is_seed_reproducible():
    X=np.random.default_rng(1).normal(size=(50,36))
    a=kmeans2(X,9);b=kmeans2(X,9)
    assert np.array_equal(a[0],b[0]);assert np.array_equal(a[1],b[1])


def test_one_heldout_template_is_not_independent_replication():
    rows, X = make_dataset()
    mask=np.array([r['source_wav'] not in FROZEN_HELDOUT or r['source_wav']=='a.wav' for r in rows])
    subrows=[r for r,keep in zip(rows,mask) if keep]
    summary,_,_=discover(subrows,X[mask])
    assert summary[0]['geometry_screen_passed']
    assert summary[0]['underreplicated_heldout_warning']
    assert summary[0]['status']=='exploratory_confound_or_underreplicated'
