import io
import math
import wave
import zipfile
import numpy as np
import pytest
from phonoweave.independent_acoustic_events import find_events,read_pcm,scan_archive,textgrid

SR=16000

def tone(freq, seconds):
    t=np.arange(round(SR*seconds))/SR
    return (.12*np.sin(2*np.pi*freq*t)).astype(np.float32)

def noise(seconds,seed=20):
    rng=np.random.default_rng(seed)
    raw=rng.normal(size=round(SR*seconds))
    return (.06*(raw-np.convolve(raw,np.ones(11)/11,mode='same'))).astype(np.float32)

def to_wav(x):
    buf=io.BytesIO()
    with wave.open(buf,'wb') as f:
        f.setnchannels(1);f.setframerate(SR);f.setsampwidth(2)
        f.writeframes(np.clip(x*32767,-32768,32767).astype('<i2').tobytes())
    return buf.getvalue()

def test_stationary_tone_has_no_large_fabricated_internal_boundary():
    x=np.r_[np.zeros(SR//5),tone(210,1),np.zeros(SR//5)]
    e=find_events(x,SR)['events']
    assert any(q['kind']=='activity_onset' for q in e)
    assert any(q['kind']=='activity_offset' for q in e)
    assert not [q for q in e if q['kind']=='internal_acoustic_change' and 400<q['time_ms']<1000]

def test_voiced_to_fricative_unanchored_event_detected():
    x=np.r_[np.zeros(SR//5),tone(220,.45),noise(.45),np.zeros(SR//5)]
    e=find_events(x,SR)['events']
    assert any(q['kind']=='internal_acoustic_change' and abs(q['time_ms']-650)<=70 for q in e),e

def test_two_strong_vowels_detectable_without_oto():
    t=np.arange(round(SR*.6))/SR
    a=.08*(np.sin(2*np.pi*220*t)+.3*np.sin(2*np.pi*440*t))
    b=.08*(.25*np.sin(2*np.pi*220*t)+np.sin(2*np.pi*660*t))
    e=find_events(np.r_[np.zeros(SR//5),a,b,np.zeros(SR//5)].astype('float32'),SR)['events']
    assert any(q['kind']=='internal_acoustic_change' and abs(q['time_ms']-800)<85 for q in e)

def test_silence_only_does_not_count_as_internal_phone():
    x=np.r_[tone(220,.3),np.zeros(SR//4),tone(330,.3)]
    e=find_events(x,SR)['events']
    assert not [q for q in e if q['kind']=='internal_acoustic_change' and 290<q['time_ms']<560]

def test_nonfinite_and_short_abstain():
    assert find_events(np.zeros(10,dtype='float32'),SR)['status']=='too_short'
    a=np.zeros(SR,dtype='float32');a[40]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):find_events(a,SR)

def test_pcm_sha_roundtrip():
    x=tone(260,.25);y,s,hashcode=read_pcm(to_wav(x))
    assert s==SR and len(y)==len(x) and len(hashcode)==64

def test_textgrid_marks_are_not_phone_claims():
    txt=textgrid([{'time_ms':250,'kind':'internal_acoustic_change'}],1.0)
    assert 'NOT_phonemes' in txt and 'points: size = 1' in txt

def test_scan_archive_no_oto_required(tmp_path):
    zp=tmp_path/'bank.zip'
    with zipfile.ZipFile(zp,'w') as z:z.writestr('bank/recording.wav',to_wav(np.r_[tone(220,.4),noise(.4)]))
    result=scan_archive(zp,'bank/',tmp_path/'out')
    assert result['source_wavs']==1 and result['oto_files_read']==0
    assert (tmp_path/'out'/'textgrids'/'recording.TextGrid').is_file()
    with pytest.raises(FileExistsError):scan_archive(zp,'bank/',tmp_path/'out')

def test_audio_not_changed_by_discovery():
    x=np.r_[tone(220,.3),noise(.3)];old=x.copy()
    find_events(x,SR)
    assert np.array_equal(old,x)

def test_segments_partition_active_without_phone_assumption():
    from phonoweave.independent_acoustic_events import propose_intervals
    events=[{'time_ms':100,'kind':'activity_onset'},
            {'time_ms':300,'kind':'internal_acoustic_change'},
            {'time_ms':510,'kind':'activity_offset'},
            {'time_ms':730,'kind':'activity_onset'},
            {'time_ms':900,'kind':'activity_offset'}]
    pieces=propose_intervals(events,1000)
    assert [(x['start_ms'],x['end_ms']) for x in pieces]==[(100,300),(300,510),(730,900)]
    assert all('NOT_verified_phone' in x['status'] for x in pieces)
    tg=textgrid(events,1.0,pieces)
    assert 'IntervalTier' in tg and 'unverified_acoustic_partitions' in tg

def test_segments_reject_out_of_bounds_and_overlap():
    from phonoweave.independent_acoustic_events import propose_intervals
    with pytest.raises(ValueError):propose_intervals([{'time_ms':200,'kind':'activity_offset'}],100)

def test_nearest_event_matching():
    from phonoweave.acoustic_event_audit import _nearest_matches
    assert _nearest_matches(np.array([10.,110.]),np.array([30.,90.]),25)==2
    assert _nearest_matches(np.array([10.,110.]),np.array([30.,90.]),10)==0
    assert _nearest_matches(np.array([]),np.array([10.]),25)==0
