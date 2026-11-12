"""VC audible splice ablation: same-CV flanks, real WAVs, no OTO mutation."""
from __future__ import annotations
import csv
import json
import wave
import numpy as np
import pytest

from phonoweave.vc_audio_ablation_v43 import run, SCHEMA


def wav(path,f0,rate=48000):
    path.parent.mkdir(parents=True,exist_ok=True)
    t=np.arange(round(rate*.7))/rate
    a=.15*np.sin(2*np.pi*f0*t)
    with wave.open(str(path),"wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate)
        w.writeframes((a*32767).astype("<i2").tobytes())


def voicebank(tmp_path, *, short_vc=False, shared_wav=False):
    root=tmp_path/"voice";bank=root/"D4";bank.mkdir(parents=True)
    for name,f in (("left.wav",320),("transition.wav",800),("right.wav",600)):
        wav(bank/name,f)
    lines=[
        "left.wav=ma,10,120,-640,125,40",
        f"{'left.wav' if shared_wav else 'transition.wav'}=a sh,10,120,-640,{35 if short_vc else 125},40",
        "right.wav=sha,10,120,-640,125,40",
    ]
    original="\n".join(lines)+"\n"
    (bank/"oto.ini").write_text(original)
    return root,original


def test_exact_case_renders_blinded_fixed_length_audio_without_mutating_sources(tmp_path):
    root,oto=voicebank(tmp_path)
    spec={"schema":SCHEMA,"cases":[{
        "id":"example-one","bank":"D4",
        "preceding_vowel":"a","next_consonant":"sh",
        "left_cv_alias":"ma","vc_alias":"a sh","right_cv_alias":"sha",
    }]}
    cases=tmp_path/"cases.json";cases.write_text(json.dumps(spec))
    output=tmp_path/"audition"
    originals={x.relative_to(root):x.read_bytes() for x in root.rglob("*") if x.is_file()}
    result=run(root,output,cases_file=cases)
    assert result["audible_blind_pairs"]==1
    pair=json.loads((output/"private_condition_key.json").read_text())["conditions"][0]
    assert {pair["vc_condition"],pair["cv_only_condition"]}=={"A","B"}
    files=sorted((output/"blind_audio").glob("*.wav"))
    assert len(files)==2
    with wave.open(str(files[0])) as a, wave.open(str(files[1])) as b:
        assert a.getframerate()==b.getframerate()==48000
        assert a.getnframes()==b.getnframes()
        assert a.getnframes()>10000
    rows=list(csv.DictReader((output/"listening_sheet.csv").open()))
    assert len(rows)==1 and rows[0]["preferred"]==""
    assert all(x.read_bytes()==originals[x.relative_to(root)] for x in root.rglob("*") if x.is_file())
    assert (root/"D4"/"oto.ini").read_text()==oto
    with pytest.raises(FileExistsError):
        run(root,output,cases_file=cases)
    with pytest.raises(ValueError,match="outside"):
        run(root,root/"wrong",cases_file=cases)


def test_auto_case_uses_existing_vc_only_as_target_object(tmp_path):
    root,_=voicebank(tmp_path)
    report=run(root,tmp_path/"auto",max_pairs=1)
    assert report["audible_blind_pairs"]==1
    key=json.loads((tmp_path/"auto"/"private_condition_key.json").read_text())
    assert key["selection"]["VC_alias_used_only_for_test_case_selection"] is True
    assert key["controls"]["common_left_vowel_audio"] is True
    assert key["controls"]["common_right_vowel_audio"] is True
    assert key["controls"]["not_full_renderer_or_recording_budget_equivalence"] is True


def test_reject_bad_case_references(tmp_path):
    root,_=voicebank(tmp_path)
    cases=tmp_path/"bad.json"
    cases.write_text(json.dumps({"schema":SCHEMA,"cases":[{
        "id":"one","bank":"D4","preceding_vowel":"a","next_consonant":"sh",
        "left_cv_alias":"not_exist","vc_alias":"a sh","right_cv_alias":"sha",
    }]}))
    with pytest.raises(ValueError,match="alias not unique/present"):
        run(root,tmp_path/"result",cases_file=cases)
    assert not (tmp_path/"result").exists()


def test_reject_unusable_vc_interval_without_fake_wav(tmp_path):
    root,_=voicebank(tmp_path,short_vc=True)
    with pytest.raises(ValueError,match="no eligible"):
        run(root,tmp_path/"result",max_pairs=1)
    assert not (tmp_path/"result").exists()


def test_refuse_overlapping_shared_input_as_independent_source(tmp_path):
    root,_=voicebank(tmp_path,shared_wav=True)
    # left and VC segments in same source file overlap: strict refusal.
    with pytest.raises(ValueError,match="no eligible"):
        run(root,tmp_path/"result",max_pairs=1)
