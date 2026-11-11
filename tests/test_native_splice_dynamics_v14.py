"""Research invariants: these tests do not validate phone identities or listening quality."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import wave
import io
import zipfile

import numpy as np
import pytest

from phonoweave.native_splice_dynamics_v14 import (
    WavWindow, BUDGET, mel_signature, seam_components, seam_score,
    render_splice, periodicity, transient_metrics, continuation_signature,
    select_donor, fit_trajectory_normalizer, trajectory_loss, train_scales,
    load_source_windows, _wav_bytes, run)
from phonoweave.native_synthesis_proxy_v13 import Clip, build_clips

RATE=16000

def tone(freq,ms,amp=.2):
    t=np.arange(int(RATE*ms/1000))/RATE
    return (amp*np.sin(2*np.pi*freq*t)).astype(np.float32)

def make_window(freq):
    return WavWindow(tone(freq,90),tone(freq,150),RATE,320.)


def test_real_splice_has_expected_duration_and_pcm():
    left=tone(150,90);right=tone(300,150)
    out=render_splice(left,right,RATE)
    assert len(out)==len(left)+len(right)-round(RATE*.008)
    assert out.dtype==np.float32 and np.isfinite(out).all()
    with wave.open(io.BytesIO(_wav_bytes(out,RATE))) as f:
        assert f.getframerate()==RATE and f.getnframes()==len(out)


def test_seam_artifact_is_finite_even_with_silence():
    x=np.zeros(int(RATE*.09),np.float32)
    y=tone(150,150)
    out=transient_metrics(render_splice(x,y,RATE),len(x),RATE)
    assert all(np.isfinite(v) and v>=0 for v in out.values())
    assert 'splice_energy_deviation_abs_log' in out


def test_audio_signature_and_trajectory_shapes():
    w=make_window(220)
    assert mel_signature(w.left,RATE,'left').shape==(24,)
    assert continuation_signature(w.right,RATE).shape==(8,24)


def test_periodicity_gate_refuses_silence():
    assert periodicity(np.zeros(int(RATE*.03),np.float32),RATE) is None


def test_same_donor_budget_and_target_right_isolation():
    left=tone(110,90)
    ws=[make_window(200),make_window(300),make_window(110)]
    scales={'mel_mse':1,'log_rms_gap':1}
    ids=np.array([0,1,2])
    a=select_donor(left,ws,ids,scales)
    corrupted_target=WavWindow(left,np.full(2400,np.nan,np.float32),RATE,320)
    # Target right is never passed to select_donor; changing it cannot affect donor ID.
    assert select_donor(corrupted_target.left,ws,ids,scales)==a
    assert a in list(ids)


def test_donor_selection_depends_on_left_audio():
    ws=[make_window(150),make_window(330)]
    scales={'mel_mse':1,'log_rms_gap':1}
    first=select_donor(tone(150,90),ws,np.array([0,1]),scales)
    second=select_donor(tone(330,90),ws,np.array([0,1]),scales)
    assert first==0 and second==1


def test_train_only_scales_ignore_heldout_waveform():
    ws=[make_window(180+i*30) for i in range(15)]
    mask=np.array([True]*12+[False]*3)
    a=train_scales(ws,mask)
    mu,sc=fit_trajectory_normalizer(ws,mask)
    hacked=list(ws)
    hacked[-1]=WavWindow(np.full_like(ws[-1].left,1000),np.full_like(ws[-1].right,1000),RATE,320)
    b=train_scales(hacked,mask)
    m2,s2=fit_trajectory_normalizer(hacked,mask)
    assert a==b and np.array_equal(mu,m2) and np.array_equal(sc,s2)


def test_continuation_uses_heldout_right_only_for_scoring():
    w=make_window(220)
    mu=np.zeros(24);sc=np.ones(24)
    same=trajectory_loss(continuation_signature(w.right,RATE),continuation_signature(w.right,RATE),mu,sc)
    different=trajectory_loss(continuation_signature(w.right,RATE),continuation_signature(make_window(440).right,RATE),mu,sc)
    assert same==0 and different>0


def test_source_archive_hash_mismatch_rejected_before_audio_load(tmp_path):
    reg={'schema':'phonoweave.audio_only_acoustic_motifs.v0.9','regions':[]}
    (tmp_path/'motif_regions.json').write_text(json.dumps(reg))
    np.savez(tmp_path/'raw_spectral_embeddings.npz',vectors=np.empty((0,48)))
    (tmp_path/'summary.json').write_text(json.dumps({**reg,'regions':0,'archive_sha256':'f'*64}))
    archive=tmp_path/'fake.zip';archive.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='source ZIP SHA256'):
        load_source_windows(tmp_path,archive,[])


def test_seam_score_reacts_to_energy_gap():
    x={'mel_mse':1.0,'log_rms_gap':0.,'periodicity_pair_valid':False,'periodicity_gap':None}
    y={**x,'log_rms_gap':2.}
    assert seam_score(x,{'mel_mse':1.,'log_rms_gap':1.})<seam_score(y,{'mel_mse':1.,'log_rms_gap':1.})


def test_original_audio_permitted_only_as_in_memory_window():
    w=make_window(200)
    assert w.left.dtype==np.float32 and w.right.dtype==np.float32
    assert BUDGET==4


def test_real_run_frozen_identity_and_existing_output_refused():
    p=Path('/mnt/data/pw-v9/audio_results')
    z=Path('/mnt/data/pw_v9_sources/PhonoWeave_Actual_Phone_Segmentation_v0.1.zip')
    if not (p.exists() and z.exists()): pytest.skip('private historical archive not available in CI')
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp)/'result'
        x=run(p,z,out)
        assert x['audio_source']['source_pcm_sha256_verified']==167
        assert x['audit']['observations']==28
        assert x['audit']['trajectory_mse']['relative_gain_positive_is_better']<0
        with pytest.raises(FileExistsError):run(p,z,out)