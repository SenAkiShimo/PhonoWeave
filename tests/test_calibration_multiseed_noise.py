import numpy as np
import pytest
from phonoweave.calibration_multiseed_noise import (
    add_calibrated_noise, signal_reference_rms, _recording_seed, SEEDS
)

def test_calibrated_white_noise_reproducible():
    x=np.zeros(16000)
    x[2000:13000]=0.08
    a=add_calibrated_noise(x,16000,20.,"white",103)
    b=add_calibrated_noise(x,16000,20.,"white",103)
    c=add_calibrated_noise(x,16000,20.,"white",211)
    assert np.array_equal(a,b)
    assert not np.array_equal(a,c)

def test_white_and_pink_match_target_rms():
    x=np.full(16000,0.05)
    for noise_type in ("white","pink"):
        y=add_calibrated_noise(x,16000,20.,noise_type,SEEDS[0])
        realized=np.sqrt(np.mean((y-x)**2))
        assert realized==pytest.approx(signal_reference_rms(x)/10,rel=1e-8)

def test_pink_noise_is_more_low_frequency_weighted():
    sr=16000;x=np.full(sr,0.08)
    a=add_calibrated_noise(x,sr,20.,"white",123)-x
    b=add_calibrated_noise(x,sr,20.,"pink",123)-x
    freqs=np.fft.rfftfreq(sr,1/sr)
    band=(freqs>50)&(freqs<500)
    upper=(freqs>3000)&(freqs<6000)
    def ratio(y):
        spectrum=np.abs(np.fft.rfft(y))**2
        return np.sum(spectrum[band])/np.sum(spectrum[upper])
    assert ratio(b)>ratio(a)*5

def test_recording_seed_stable():
    assert _recording_seed(103,"fixture.wav","white")==_recording_seed(103,"fixture.wav","white")
    assert _recording_seed(103,"fixture.wav","white")!=_recording_seed(211,"fixture.wav","white")

def test_invalid_noise_input_rejected():
    with pytest.raises(ValueError):
        add_calibrated_noise(np.ones(16000),16000,20.,"unknown",103)
