"""v11 synthetic safety + invariants. No human phone boundary validation."""
import json
from pathlib import Path

import numpy as np
import pytest

from phonoweave.native_context_transfer_v11 import (
    _other_carrier_hits, _ridge_predict, ridge_context_control, run,
    style_replicability, validate_frozen_v10, neighboring_acoustics_transfer,
)


def _row(context,bank,ix=1,t=10):
    return {'wav_archive_entry':f'{bank}/{context}.wav','bank':bank,
            'clip_group':context,'source_pcm_sha256':f'{t:064x}',
            'region_index':ix,'start_ms':0.,'end_ms':100.}


def test_verified_v10_requires_identical_audio_region_and_hash(tmp_path):
    r=_row('opaque','lu_C4',1,5)
    z=tmp_path/'v10';z.mkdir()
    (z/'regions.json').write_text(json.dumps({'schema':'phonoweave.cross_recording_acoustic_motifs.v1.0','regions':[r]}))
    (z/'summary.json').write_text(json.dumps({'schema':'phonoweave.cross_recording_acoustic_motifs.v1.0','regions_total':1,'input_source_zip_sha256':'f'*64}))
    assert validate_frozen_v10([r],z)['v10_region_keys_and_pcm_verified']
    with pytest.raises(ValueError,match='identity'):
        validate_frozen_v10([dict(r,start_ms=10.)],z)
    with pytest.raises(ValueError,match='identity'):
        validate_frozen_v10([dict(r,source_pcm_sha256='e'*64)],z)


def test_style_replicability_no_false_claim_for_single_mode():
    rows=[_row(str(i),'lu_C4',1,i+1) for i in range(30)]
    Y=np.ones((30,48))
    m=style_replicability(Y,rows,np.arange(30)<20,
                          (np.arange(30)>=20)&(np.arange(30)<25),np.arange(30)>=25,1)
    assert m['gate_pass'] is False
    assert m['validation_style_ari'] is None


def test_style_replicability_perfect_shared_axis_recovers_acoustic_cluster():
    rng=np.random.default_rng(2)
    rows=[]; data=[];train=[];val=[];audit=[]
    for n in range(30):
        for style in ['lu','mang']:
            for j in range(2):
                r=_row(f'ctx{n}',f'{style}_{j}',1,1+n*4+j+(0 if style=='lu' else 2))
                rows.append(r)
                x=rng.normal(0,.05,48);x[0]+=(-3 if n%2==0 else 3)
                data.append(x);train.append(n<18);val.append(18<=n<24);audit.append(n>=24)
    q=style_replicability(np.array(data),rows,np.array(train),np.array(val),np.array(audit),2)
    assert q['validation_style_ari']>.9
    assert q['gate_pass']


def test_style_replicability_detects_different_style_partition():
    rng=np.random.default_rng(15)
    rows=[]; data=[];train=[];val=[];audit=[]
    for n in range(30):
        for style in ['lu','mang']:
            x=rng.normal(0,.2,48)
            if style=='lu':x[0]+=(-4 if n%2==0 else 4)
            else:x[1]+=(-4 if n%3==0 else 4)
            rows.append(_row(f'ctx{n}',f'{style}_C4',1,n*2+int(style=='mang')+1))
            data.append(x);train.append(n<18);val.append(18<=n<24);audit.append(n>=24)
    q=style_replicability(np.array(data),rows,np.array(train),np.array(val),np.array(audit),2)
    assert q['validation_style_ari'] < .65
    assert not q['gate_pass']


def test_carrier_support_needs_multiple_families_and_styles():
    rows=[_row('a','lu_C4',1,1),_row('b','mang_C4',1,2),_row('c','lu_C4',1,3)]
    X=np.zeros((3,48));labels=np.zeros(3,int);known=np.ones(3,bool)
    got=_other_carrier_hits(X,rows,labels,known,np.array([1,1,0],bool),np.array([0,0,1],bool))
    assert got['heldout_cross_context_echo_candidates']==0  # cannot calibrate on two examples


def test_carrier_support_never_tunes_radius_on_audit():
    rows=[];X=[]
    for i in range(28):
        style=['lu_C4','mang_C4'][i%2]
        rows.append(_row(f'ctx{i}',style,1,i+1))
        vec=np.zeros(48);vec[0]=i*.01;X.append(vec)
    X=np.array(X);labels=np.zeros(28,int);known=np.ones(28,bool)
    dev=np.arange(28)<24;aud=~dev
    a=_other_carrier_hits(X,rows,labels,known,dev,aud)
    X[dev==False,0]+=200
    b=_other_carrier_hits(X,rows,labels,known,dev,aud)
    assert a['development_distance_radii']==b['development_distance_radii']
    assert a['heldout_cross_context_echo_candidates']>=b['heldout_cross_context_echo_candidates']


def test_ridge_predicts_linear_relation():
    rng=np.random.default_rng(5)
    X=rng.normal(size=(100,48));Y=.5*X
    pred=_ridge_predict(X[:70],Y[:70],X[70:],alpha=1.)
    assert np.mean((pred-Y[70:])**2)<.01


def test_ridge_neighborhood_signal_vs_shuffled_controls():
    rng=np.random.default_rng(4)
    rows=[];x=[];y=[];t=[];v=[];a=[]
    for i in range(90):
        for bank in ['lu_C4','mang_D4']:
            xx=rng.normal(size=48)
            yy=xx*.8+rng.normal(scale=.05,size=48)
            rows.append(_row(f'ctx{i}',bank,1,i+1))
            x.append(xx);y.append(yy);t.append(i<55);v.append(55<=i<72);a.append(i>=72)
    r=ridge_context_control(np.array(x),np.array(y),rows,np.array(t),np.array(v),np.array(a))
    assert r['development_validation_improvement_vs_bank_baseline']>.2
    assert r['valid_on_validation_not_causal'] is True
    assert r['audit_improvement_vs_bank_baseline_descriptive']>.2


def test_ridge_context_uncorrelated_data_abstains():
    rng=np.random.default_rng(7)
    rows=[];x=[];y=[];t=[];v=[];a=[]
    for i in range(60):
        for bank in ['lu_C4','mang_D4']:
            rows.append(_row(f'ctx{i}',bank,1,i+1))
            x.append(rng.normal(size=48));y.append(rng.normal(size=48))
            t.append(i<40);v.append(40<=i<50);a.append(i>=50)
    r=ridge_context_control(np.array(x),np.array(y),rows,np.array(t),np.array(v),np.array(a))
    assert r['valid_on_validation_not_causal'] is False


def test_neighbor_context_is_from_another_segment_not_target():
    source=[_row('first','lu_C4',0,1),_row('first','lu_C4',1,1)]
    X=np.vstack([np.ones(48),np.ones(48)*2])
    r=neighboring_acoustics_transfer(source,X,[source[1]],np.array([1],bool),np.array([0],bool),np.array([0],bool),1)
    assert np.array_equal(r['matrix'],np.ones((1,48)))


def test_run_rejects_existing_output_before_loading(tmp_path):
    dest=tmp_path/'dest';dest.mkdir()
    with pytest.raises(FileExistsError):run(tmp_path/'missing',dest)


def test_run_rejects_nonexclusive_sources(tmp_path):
    with pytest.raises(ValueError,match='exactly one'):
        run(None,tmp_path/'new')
    with pytest.raises(ValueError,match='exactly one'):
        run(tmp_path/'v9',tmp_path/'new',full_wav_zip=tmp_path/'wav.zip')


def test_run_on_small_synthetic_v9_reports_no_phone_approval(tmp_path):
    rows=[];features=[]
    for i in range(28):
        bank=['lu_C4','mang_D4'][i%2]
        for phase in (0,1,2):
            rows.append(_row(f'ctx{i}',bank,phase,i+1))
            features.append(np.ones(48)*(phase+1))
    v9=tmp_path/'v9';v9.mkdir()
    (v9/'motif_regions.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':rows}))
    (v9/'summary.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':len(rows),'archive_sha256':'f'*64}))
    np.savez_compressed(v9/'raw_spectral_embeddings.npz',vectors=np.array(features))
    result=run(v9,tmp_path/'output')
    assert result['stages_approving_new_phone_or_alias']==0
    assert result['no_voicebank_or_reclist_modified']
    assert (tmp_path/'output'/'audit_region_recurrence.json').exists()
    assert len(result['stage_results'])==3


def test_small_full_wav_zip_uses_only_unanchored_audio(tmp_path):
    import io, wave, zipfile
    archive=tmp_path/'original_full_wavs.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for name,freq in [('bankA/carrier_a.wav',240),('bankB/carrier_b.wav',420),('bankC/carrier_c.wav',760)]:
            rate=16000; t=np.arange(32000)/rate
            y=np.sin(2*np.pi*freq*t)*0.18
            data=(y*32767).astype('<i2')
            stream=io.BytesIO()
            with wave.open(stream,'wb') as wf:
                wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(rate)
                wf.writeframes(data.tobytes())
            z.writestr(name,stream.getvalue())
    dest=tmp_path/'out'
    report=run(None,dest,full_wav_zip=archive)
    assert 'unanchored_full_recordings' in report['source_provenance']
    assert report['regions_total']>=3
    assert report['stages_approving_new_phone_or_alias']==0
    assert report['no_ipa_g2p_oto_timestamp_or_transcript_in_model']


def test_run_rejects_invalid_pcm_provenance(tmp_path):
    r=_row('opaque','lu_C4',0,5)
    r['source_pcm_sha256']='bad'
    v9=tmp_path/'v9';v9.mkdir()
    (v9/'motif_regions.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':[r]}))
    (v9/'summary.json').write_text(json.dumps({'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':1,'archive_sha256':'f'*64}))
    np.savez_compressed(v9/'raw_spectral_embeddings.npz',vectors=np.zeros((1,48)))
    with pytest.raises(ValueError,match='source PCM SHA256'):
        run(v9,tmp_path/'out')
    assert not (tmp_path/'out').exists()