"""Acoustic-only WAV tests. These are NOT checks against manual phone boundaries."""
import io
import json
import math
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

from phonoweave.native_wav_motifs_v9 import (
    audio_features, pcm16, propose_segments, embed_segment,
    extract_from_audio_zip, discover_audio_motifs, run,
)


def wav_bytes(x, sr=16000):
    sink=io.BytesIO()
    with wave.open(sink,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr)
        w.writeframes((np.clip(x,-1,1)*32767).astype('<i2').tobytes())
    return sink.getvalue()


def test_silence_produces_no_phoneme_or_motif_segments():
    audio=np.zeros(16000,dtype=np.float32)
    mel,rms,t=audio_features(audio,16000)
    assert propose_segments(mel,rms,t)==[]


def test_spectral_distinction_in_real_pcm_sinusoidal_audio():
    t=np.arange(16000)/16000
    a=np.sin(2*np.pi*240*t)*.15
    b=np.sin(2*np.pi*2900*t)*.15
    x,rate,_=pcm16(wav_bytes(a))
    y,rate,_=pcm16(wav_bytes(b))
    mx,rx,tx=audio_features(x,rate)
    my,ry,ty=audio_features(y,rate)
    ex=embed_segment(mx,2,25);ey=embed_segment(my,2,25)
    assert ex.shape==(48,) and ey.shape==(48,)
    assert np.linalg.norm(ex-ey)>3


def test_waveform_activity_and_spectral_regions_no_oto():
    rate=16000
    x=np.concatenate([np.zeros(rate//5),.1*np.sin(2*np.pi*400*np.arange(rate//3)/rate),
                      .1*np.sin(2*np.pi*2500*np.arange(rate//3)/rate),np.zeros(rate//5)])
    mel,rms,t=audio_features(x.astype(np.float32),rate)
    spans=propose_segments(mel,rms,t)
    assert spans
    assert all(a<b and a>=0 and b<=len(t) for a,b in spans)


def test_rejects_stereo():
    with io.BytesIO() as f:
        with wave.open(f,'wb') as w:
            w.setnchannels(2);w.setsampwidth(2);w.setframerate(16000)
            w.writeframes(np.zeros(200,dtype='<i2').tobytes())
        with pytest.raises(ValueError,match='mono'):
            pcm16(f.getvalue())


def test_extract_from_zip_does_not_use_labels_or_oto_times(tmp_path):
    p=tmp_path/'some.zip'
    t=np.arange(16000//2)/16000
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('anything/lu_C4/EXPLICIT_OTO124/source_context_600ms.wav',wav_bytes(.2*np.sin(2*np.pi*500*t)))
    records,X=extract_from_audio_zip(p)
    assert records and X.shape[1]==48
    assert records[0]['bank']=='lu_C4'
    assert 'oto' not in records[0]['status'].lower()


def test_small_cohort_must_abstain(tmp_path):
    records=[{'clip_group':'A','bank':'lu_C4'} for _ in range(40)]
    X=np.random.default_rng(1).normal(size=(40,48))
    _, report=discover_audio_motifs(records,X)
    assert report['status']=='abstain_too_few_audio_regions_or_unique_clips'


def test_inference_unchanged_when_clip_names_changed(tmp_path):
    t=np.arange(16000//2)/16000
    audio=wav_bytes(.2*np.sin(2*np.pi*500*t))
    for ix in (1,2):
        path=tmp_path/f'archive{ix}.zip'
        with zipfile.ZipFile(path,'w') as z:
            label=f'phony{ix}_OTO99999'
            z.writestr(f'root/lu_C4/{label}/source_context_600ms.wav',audio)
    a,A=extract_from_audio_zip(tmp_path/'archive1.zip')
    b,B=extract_from_audio_zip(tmp_path/'archive2.zip')
    assert np.array_equal(A,B)
