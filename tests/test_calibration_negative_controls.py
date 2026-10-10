import io
import wave
import pytest
from phonoweave.calibration_negative_controls import save_negative_recording, evaluate_negative_corpus

def wav_bytes():
    bio=io.BytesIO()
    with wave.open(bio,"wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
        w.writeframes(bytes(96000))
    return bio.getvalue()

def test_negative_corpus_is_separate_and_preserves_metadata(tmp_path):
    saved=save_negative_recording(tmp_path,"silence",wav_bytes(),device_category="phone",environment="quiet",actual_processing={"autoGainControl":False})
    assert saved["metadata"]["sample_rate"]==16000
    assert saved["metadata"]["device_category"]=="phone"
    assert saved["metadata"]["audio_processing_actual"]["autoGainControl"] is False
    report=evaluate_negative_corpus(tmp_path)
    assert report["total_recordings"]==1
    assert report["summary"]["silence"]["shoulder_triggers"]==0

def test_reject_invalid_categories(tmp_path):
    with pytest.raises(ValueError,match="category"):
        save_negative_recording(tmp_path,"fricative",wav_bytes())
