"""CV-first regression tests; no original OTO required or consulted."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from phonoweave.cv_first_accent_v40 import (
    SCHEMA, RESULT_SCHEMA, analyze, export, read_manifest,
)


def wav(path, f1, f2):
    path.parent.mkdir(parents=True,exist_ok=True)
    sr=48000
    t=np.arange(sr*2,dtype=float)/sr
    x=np.where(t < 1, .2*np.sin(2*np.pi*f1*t),
               .2*np.sin(2*np.pi*f2*t))
    with wave.open(str(path),"wb") as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr)
        f.writeframes((x*32767).astype("<i2").tobytes())


def prepared(tmp_path):
    root=tmp_path/"input"
    root.mkdir()
    labels=[
        ("sha","sh","a",600,350),
        ("shi","sh","i",600,440),
        ("shu","sh","u",600,500),
        ("ma","m","a",300,350),
        ("mi","m","i",300,440),
    ]
    rows=[]
    for rep in (1,2):
        for index, (cv,c,v,c_hz,v_hz) in enumerate(labels):
            name=f"{cv}_{rep}.wav"
            wav(root/name,c_hz + rep*.7 + index*.02,
                v_hz + rep*.7 + index*.02)
            rows.append({"id":f"{cv}-{rep}","cv":cv,"consonant":c,
                         "vowel":v,"wav":name})
    manifest=root/"cv_manifest.json"
    manifest.write_text(json.dumps({"schema":SCHEMA,"recordings":rows}))
    return manifest


def test_discovers_both_axes_from_cv_only(tmp_path):
    manifest=prepared(tmp_path)
    report=analyze(manifest)
    assert report["schema"]==RESULT_SCHEMA
    assert report["recording_count"]==10
    assert all(not r["existing_OTO_used"] for r in report["cross_context_comparisons"])
    c=[r for r in report["cross_context_comparisons"] if r["comparison"]=="same_C_across_V"]
    v=[r for r in report["cross_context_comparisons"] if r["comparison"]=="same_V_across_C"]
    assert any(r["same_author_symbol"]=="sh" and r["varied_author_symbols"]==["a","i"] for r in c)
    assert any(r["same_author_symbol"]=="a" and r["varied_author_symbols"]==["m","sh"] for r in v)
    assert all(r["descriptive_profile_distance"] is not None for r in c+v)
    assert report["new_split_labels"]==[] and report["automatic_reclist"]==[]
    dest=tmp_path/"results"
    export(report,manifest,dest)
    assert (dest/"cv_acoustic_discovery.json").is_file()
    assert (dest/"cv_comparison.csv").is_file()
    assert not (manifest.parent/"oto.ini").exists()
    with pytest.raises(ValueError,match="outside"):
        export(report,manifest,manifest.parent/"inside")
    with pytest.raises(FileExistsError):
        export(report,manifest,dest)


def test_oto_and_old_split_field_forbidden(tmp_path):
    manifest=prepared(tmp_path)
    obj=json.loads(manifest.read_text())
    obj["recordings"][0]["oto_alias"]="D4-sh a"
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match="CV record needs"):
        read_manifest(manifest)


def test_one_off_examples_abstain(tmp_path):
    manifest=prepared(tmp_path)
    obj=json.loads(manifest.read_text())
    obj["recordings"]=[r for r in obj["recordings"] if r["id"].endswith("-1")]
    manifest.write_text(json.dumps(obj))
    report=analyze(manifest)
    assert all(r["descriptive_profile_distance"] is None
               and r["status"]=="insufficient_independent_CV_recordings"
               for r in report["cross_context_comparisons"])


def test_sound_names_are_not_inferred_from_cv_spellings(tmp_path):
    manifest=prepared(tmp_path)
    obj=json.loads(manifest.read_text())
    obj["recordings"][0]["consonant"]="custom-consonant"
    obj["recordings"][0]["vowel"]="custom-vowel"
    manifest.write_text(json.dumps(obj))
    root, rows=read_manifest(manifest)
    assert rows[0]["consonant"]=="custom-consonant"
    assert rows[0]["cv"]=="sha"


def test_no_duplicated_pcm_as_independent_recordings(tmp_path):
    manifest=prepared(tmp_path)
    obj=json.loads(manifest.read_text())
    obj["recordings"][1]["wav"]=obj["recordings"][0]["wav"]
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match="duplicated decoded WAV"):
        analyze(manifest)
