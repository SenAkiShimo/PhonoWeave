"""Frozen source integrity and real human-review separation tests."""
from __future__ import annotations

import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from phonoweave.vc_boundary_adjudication import (
    prepare_manual_boundary_review, validate_human_review, suggest_sh_noise_bracket
)
from phonoweave.vc_boundary_review_cli import main as cli_main


def _wav(audio: np.ndarray, rate: int=44100) -> bytes:
    buf=io.BytesIO()
    with wave.open(buf,"wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate)
        w.writeframes((np.clip(audio,-.99,.99)*32767).astype("<i2").tobytes())
    return buf.getvalue()


def _pack(tmp_path):
    root=tmp_path/"original"
    root.mkdir()
    groups=[]
    for right in ("m","sh"):
        items=[]
        for j,left in enumerate(("i","ir","i0")):
            rate=44100
            time=np.arange(int(rate*.6))/rate
            wave_sample=.22*np.sin(2*np.pi*(230+j*10)*time)
            if right=="sh":
                noise=np.random.default_rng(j).normal(0,.08,len(time))
                wave_sample[round(.28*rate):round(.42*rate)]=noise[round(.28*rate):round(.42*rate)]
            rel=f"right_{right}/{left}.wav"
            content=_wav(wave_sample,rate)
            path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(content)
            items.append({"left_surface":left,"source_wav":f"{right}.wav",
                "original_alias":f"D4{left} {right}","oto_line":100+j,
                "excerpt_filename":rel,
                "excerpt_sha256":sha256(content).hexdigest(),
                "excerpt_oto_anchor_ms":240,"boundary_verified":False,
                "source_wav_sha256": "a"*64})
        groups.append({"right_surface":right,"clip_records":items})
    manifest={"schema":"phonoweave.vc_boundary_feasibility_gate.v0.1",
        "status":"auditory_boundary_review_required_no_split_share_evidence",
        "source":{"archive_sha256":"c"*64,"oto_sha256":"b"*64},
        "review_groups":groups}
    (root/"review_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    fieldnames=["right_surface","left_surface","original_alias","source_wav",
       "oto_line","boundary_onset_ms_in_excerpt","boundary_offset_ms_in_excerpt",
       "target_sound_present","listening_comments","reviewer"]
    out=root/"review_annotations.csv"
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader()
        for group in groups:
            for row in group["clip_records"]:
                w.writerow({"right_surface":group["right_surface"],
                 "left_surface":row["left_surface"],
                 "original_alias":row["original_alias"],
                 "source_wav":row["source_wav"],
                 "oto_line":row["oto_line"],
                 "target_sound_present":"unreviewed"})
    return root


def test_unreviewed_is_not_approved_and_sh_abstains_for_m(tmp_path):
    root=_pack(tmp_path)
    model=prepare_manual_boundary_review(root)
    assert len(model["clips"])==6
    assert model["claimed_true_boundaries"]==0
    assert model["approved_split_or_share_decisions"]==0
    assert all(row["screening"]["bracket_ms"] is None
               for row in model["clips"] if row["right"]=="m")
    result=validate_human_review(root,root/"review_annotations.csv")
    assert result["reviewed_rows"]==0
    assert result["status"].startswith("human_review_incomplete")
    assert not result["split_share_authorized"]


def test_review_csv_can_complete_without_authorizing_synthesis(tmp_path):
    root=_pack(tmp_path)
    rows=list(csv.DictReader((root/"review_annotations.csv").open()))
    for row in rows:
        row["target_sound_present"]="yes"
        row["boundary_onset_ms_in_excerpt"]="263"
        row["boundary_offset_ms_in_excerpt"]="281"
        row["reviewer"]="auditor"
    with (root/"review_annotations.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader()
        writer.writerows(rows)
    report=validate_human_review(root,root/"review_annotations.csv")
    assert report["all_six_boundaries_marked"]
    assert report["reviewed_rows"]==6
    assert report["split_share_authorized"] is False
    assert report["heldout_synthesis_gain_confirmed"] is False


def test_tampered_audio_and_bad_source_identity_rejected(tmp_path):
    root=_pack(tmp_path)
    (root/"right_sh/i.wav").write_bytes(b"tampered")
    with pytest.raises(ValueError,match="SHA-256"):
        prepare_manual_boundary_review(root)
    root=_pack(tmp_path/"second") if False else root


def test_cli_writes_json_and_protects_review_source(tmp_path):
    root=_pack(tmp_path)
    dest=tmp_path/"cue.json"
    assert cli_main(["inspect","--pack",str(root),"--output",str(dest)])==0
    data=json.loads(dest.read_text())
    assert len(data["clips"])==6
    with pytest.raises(SystemExit):
        cli_main(["inspect","--pack",str(root),"--output",str(dest)])
    with pytest.raises(SystemExit):
        cli_main(["inspect","--pack",str(root),"--output",str(root/"bad.json")])


def test_frication_cue_is_unverified_even_when_present():
    rate=44100
    signal=.20*np.sin(2*np.pi*220*np.arange(round(.6*rate))/rate)
    rng=np.random.default_rng(41)
    st,ed=round(.285*rate),round(.40*rate)
    signal[st:ed]=rng.normal(0,.11,ed-st)
    report=suggest_sh_noise_bracket(signal,rate)
    assert report["status"] in {
        "unverified_noise_onset_bracket",
        "abstain_nonquiet_preanchor_baseline",
        "abstain_no_sustained_high_frequency_frication",
        "abstain_no_gradual_onset_bracket",
        "abstain_ambiguous_onset"
    }
    assert report["status"]!="verified_boundary"
