"""Synthetic invariants only, NOT independent real phone-boundary validation."""
import io
import json
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

from phonoweave.native_motif_link_v10 import (
    SCHEMA, group_split, normalize, fit_kmeans, model_audit,
    edges_between_recordings, _approximate_cross_source_neighbors,
    extract_generic_full_wav_zip, load_v9, run,
)


def synthetic(groups=24, kinds=3, seed=8, noise=.15):
    rng=np.random.default_rng(seed)
    rows=[];V=[]
    centers=np.zeros((kinds,48))
    for k in range(kinds):
        centers[k, k*5:(k+1)*5]=4.
    for g in range(groups):
        for b in ('lu_B4','mang_D5'):
            for k in range(kinds):
                rows.append(dict(clip_group=f'g{g:02d}',bank=b,region_index=k,
                                 source_pcm_sha256=('%064x'%(g*100+k+5+(0 if b=='lu_B4' else 40))),
                                 wav_archive_entry=f'{b}/g{g:02d}_{k}.wav'))
                V.append(centers[k]+rng.normal(0,noise,48)+.01*g)
    return rows,np.asarray(V,dtype=np.float64)


def test_k_auto_selection_recovers_repeated_three_motifs():
    rows,X=synthetic()
    train,val,audit=group_split(rows)
    s,lab,known,hidden=model_audit(X,rows,train,val,audit)
    assert s['selected_k']==3
    assert s['audit_regions_unknown']<=s['regions_train_validation_audit'][-1]
    assert s['model_selection_or_threshold_tuning_on_audit'] is False
    assert s['chosen_stability_median_ari']>=.75


def test_single_unstructured_cloud_does_not_force_two_clusters():
    rows,X=synthetic(groups=45,kinds=1,noise=1.5)
    tr,va,au=group_split(rows)
    s,_,_,_=model_audit(X,rows,tr,va,au)
    assert s['selected_k']==1
    assert s['selection_status'].startswith('abstain')


def test_whole_recording_families_disjoint_across_banks():
    rows,_=synthetic()
    a,b,c=group_split(rows)
    assert np.all((a.astype(int)+b.astype(int)+c.astype(int))==1)
    for group in {r['clip_group'] for r in rows}:
        mask=np.array([r['clip_group']==group for r in rows])
        assert sum(bool(np.any(mask & part)) for part in (a,b,c))==1


def test_heldout_outlier_cannot_change_fitted_centers_or_k():
    rows,X=synthetic()
    a,b,c=group_split(rows)
    first,labels,known,h=model_audit(X,rows,a,b,c)
    modified=X.copy();modified[c]+=10000.
    second,labels2,known2,h2=model_audit(modified,rows,a,b,c)
    assert first['selected_k']==second['selected_k']
    assert first['distance_thresholds_fit_without_audit']==second['distance_thresholds_fit_without_audit']
    assert np.array_equal(h['centers'],h2['centers'])
    assert not known2[c].any()  # true unknown class, not forced nearest-class label


def test_feature_scaler_fit_only_train():
    rows,X=synthetic()
    a,b,c=group_split(rows)
    Y,meta=normalize(X,a)
    X2=X.copy();X2[c]*=9999.
    Z,_=normalize(X2,a)
    assert np.array_equal(Y[a|b],Z[a|b])
    assert meta['fit_from_train_only']


def test_names_and_opaque_symbols_not_acoustic_features():
    rows,X=synthetic()
    a,b,c=group_split(rows)
    first,labels,known,_=model_audit(X,rows,a,b,c)
    other=[dict(x,wav_archive_entry='new/path/'+str(i)+'.wav',user_label='自定义') for i,x in enumerate(rows)]
    a2,b2,c2=group_split(other)
    second,labels2,known2,_=model_audit(X,other,a2,b2,c2)
    assert first['selected_k']==second['selected_k']
    assert np.array_equal(labels,labels2)
    assert np.array_equal(known,known2)


def test_bad_partitions_or_leakage_rejected():
    rows,X=synthetic()
    a,b,c=group_split(rows)
    with pytest.raises(ValueError,match='disjoint and exhaustive'):
        model_audit(X,rows,a,a,c)
    rows2=[dict(x) for x in rows]
    rows2[np.flatnonzero(a)[0]]['clip_group']=rows2[np.flatnonzero(c)[0]]['clip_group']
    with pytest.raises(ValueError,match='leaked'):
        model_audit(X,rows2,a,b,c)


def test_too_few_recording_families_refuse_discovery():
    rows,_=synthetic(groups=5)
    with pytest.raises(ValueError,match='at least 10'):
        group_split(rows)


def test_mutual_links_never_same_recording_or_same_family():
    rows,X=synthetic()
    tr,va,au=group_split(rows)
    s,labels,known,hidden=model_audit(X,rows,tr,va,au)
    edges,report=edges_between_recordings(rows,hidden['normalized'],labels,known,tr|va)
    assert edges and report['mutual_nearest_neighbor_required']
    for e in edges:
        i=e['source_region_a'];j=e['source_region_b']
        assert i<j
        assert rows[i]['source_pcm_sha256']!=rows[j]['source_pcm_sha256']
        assert rows[i]['clip_group']!=rows[j]['clip_group']
        assert e['status'].endswith('NOT_same_phone')


def test_link_radius_unchanged_by_heldout_vectors():
    rows,X=synthetic()
    tr,va,au=group_split(rows)
    s,lab,known,h=model_audit(X,rows,tr,va,au)
    _,old=edges_between_recordings(rows,h['normalized'],lab,known,tr|va)
    altered=h['normalized'].copy();altered[au]*=1000
    newknown=known.copy();newknown[au]=False
    _,new=edges_between_recordings(rows,altered,lab,newknown,tr|va)
    assert old['dev_val_calibrated_link_radii']==new['dev_val_calibrated_link_radii']


def test_no_distance_or_input_shape_trickery():
    rows,X=synthetic()
    tr,va,au=group_split(rows)
    with pytest.raises(ValueError,match='48D'):
        model_audit(X[:,:3],rows,tr,va,au)
    X[0,0]=np.nan
    with pytest.raises(ValueError,match='48D'):
        model_audit(X,rows,tr,va,au)


def test_constant_features_do_not_create_multiple_motifs():
    rows,X=synthetic()
    X[:]=2.
    tr,va,au=group_split(rows)
    s,_,_,_=model_audit(X,rows,tr,va,au)
    assert s['selected_k']==1


def wav_bytes(signal,sr=16000):
    f=io.BytesIO()
    with wave.open(f,'wb') as out:
        out.setnchannels(1);out.setsampwidth(2);out.setframerate(sr)
        out.writeframes((np.clip(signal,-1,1)*32767).astype('<i2').tobytes())
    return f.getvalue()


def test_full_wav_parser_never_reads_oto_or_aliases(tmp_path):
    t=np.arange(16000)/16000
    x=.12*np.sin(2*np.pi*440*t)
    z=tmp_path/'full.zip'
    with zipfile.ZipFile(z,'w') as f:
        f.writestr('voice/lu_B4/abc.wav',wav_bytes(x))
        f.writestr('voice/lu_B4/oto.ini','[invalid utf-8 assumed?]\nStart=99999')
    rows,emb=extract_generic_full_wav_zip(z)
    assert rows and emb.shape[1]==48
    assert rows[0]['clip_group']=='abc.wav'
    assert 'unanchored' in rows[0]['status']


def test_source_zip_rejects_path_traversal(tmp_path):
    z=tmp_path/'bad.zip'
    with zipfile.ZipFile(z,'w') as f:f.writestr('../bad.wav',wav_bytes(np.ones(16000)*.04))
    with pytest.raises(ValueError,match='unsafe'):
        extract_generic_full_wav_zip(z)


def test_v9_manifest_rejects_tampered_region_counts(tmp_path):
    folder=tmp_path/'v9';folder.mkdir()
    np.savez_compressed(folder/'raw_spectral_embeddings.npz',vectors=np.zeros((3,48)))
    (folder/'motif_regions.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':[{}]}))
    (folder/'summary.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':1}))
    with pytest.raises(ValueError,match='count mismatch'):
        load_v9(folder)


def test_refuses_overwrite_and_no_raw_wav_export(tmp_path):
    rows,X=synthetic()
    folder=tmp_path/'v9';folder.mkdir()
    np.savez_compressed(folder/'raw_spectral_embeddings.npz',vectors=X)
    (folder/'motif_regions.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':rows}))
    (folder/'summary.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':len(rows),'archive_sha256':'a'*64}))
    out=tmp_path/'out'
    summ=run(out,v9=folder)
    assert summ['schema']==SCHEMA
    assert (out/'cross_recording_links.csv').exists()
    assert not list(out.rglob('*.wav'))
    with pytest.raises(FileExistsError):run(out,v9=folder)


def test_duplicate_pcm_across_families_rejected(tmp_path):
    rows,X=synthetic()
    # Supply otherwise valid frozen v9 manifest but with one copied PCM hash from an unrelated family.
    rows[1]['source_pcm_sha256']=rows[0]['source_pcm_sha256']
    folder=tmp_path/'v9';folder.mkdir()
    np.savez_compressed(folder/'raw_spectral_embeddings.npz',vectors=X)
    (folder/'motif_regions.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':rows}))
    (folder/'summary.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':len(rows),'archive_sha256':'a'*64}))
    # Same family is not a cross-family identity violation. Force two distinct families.
    rows[-1]['source_pcm_sha256']=rows[0]['source_pcm_sha256']
    (folder/'motif_regions.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':rows}))
    with pytest.raises(ValueError,match='leakage'):
        run(tmp_path/'out',v9=folder)