import io,wave
import numpy as np
import pytest
from phonoweave.event_unit_bridge import extract_at_ends,pcm_bytes,model,label_features,TEST_GROUPS
from phonoweave.orthography_native_units import MEL


def test_pcm_hash_is_stable():
    a=(np.sin(2*np.pi*200*np.arange(44100)/44100)*10000).astype('<i2')
    with io.BytesIO() as b:
        with wave.open(b,'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(44100);w.writeframes(a.tobytes())
        signal,rate,h=pcm_bytes(b.getvalue())
    assert len(signal)==44100 and rate==44100 and len(h)==64


def test_audio_proposals_require_no_ipa_or_alias():
    rate=44100;t=np.arange(rate*2)/rate
    signal=(np.sin(2*np.pi*500*t)*.17).astype(np.float32)
    feat,valid=extract_at_ends(signal,rate,np.array([500.,1000.]),MEL)
    assert feat.shape==(2,36) and valid.all()
    assert np.isfinite(feat).all()


def test_low_energy_abstains():
    feat,valid=extract_at_ends(np.zeros(44100*2,dtype=np.float32),44100,np.array([500.]),MEL)
    assert not valid[0]


def test_early_window_abstains():
    signal=(np.random.default_rng(4).normal(0,.1,44100)).astype(np.float32)
    _,valid=extract_at_ends(signal,44100,np.array([20.]),MEL)
    assert not valid[0]


def test_preserves_user_chosen_unicode_symbols():
    rows=[{'symbol':'i0','bank':'s_C4','source_wav':'origin.wav'},
          {'symbol':'i2','bank':'s_C4','source_wav':'origin.wav'},
          {'symbol':'自定义','bank':'s_C4','source_wav':'origin.wav'},
          {'symbol':'i0','bank':'s_C4','source_wav':next(iter(TEST_GROUPS))}]
    X=np.array([[0,1],[1,0],[2,2],[99,99]],dtype=np.float32)
    labels,Y,means,pooled=model(rows,X)
    assert labels==['i0','i2','自定义']
    assert np.allclose(means['s_C4'],X[:3].mean(axis=0))
    assert not np.any(np.abs(Y)>10)  # held-out outlier never affects centroids
    guesses,_,margin=label_features(X[:3],'s_C4',labels,Y,means,pooled)
    assert guesses[0][0]=='i0' and guesses[1][0]=='i2' and np.all(margin>=0)


def test_modified_audio_hash_is_detectable():
    a=np.arange(100,dtype='<i2')
    def make(data):
        f=io.BytesIO()
        with wave.open(f,'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(44100);w.writeframes(data.tobytes())
        return f.getvalue()
    x=pcm_bytes(make(a))[2];a[50]=0;assert pcm_bytes(make(a))[2]!=x
