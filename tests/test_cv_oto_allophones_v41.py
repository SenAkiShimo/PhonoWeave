"""v4.1 synthetic OTO/CV regression: sound localization is not VC split supervision."""
from __future__ import annotations

import json
from pathlib import Path
import wave

import numpy as np
import pytest

from phonoweave.cv_oto_allophones_v41 import (
    SCHEMA, LEXICON_SCHEMA, _cv_from_entry, _context_partition,
    _fold_scores, read_lexicon, extract_cv, analyze_candidates, export_report
)


def _tone(path:Path, c_freq:float, v_freq:float, rep:int):
    path.parent.mkdir(parents=True,exist_ok=True)
    sr=48000
    t=np.arange(int(sr*.52),dtype=np.float64)/sr
    # The effective C region is 40..150ms; V starts around 150ms.
    signal=np.where(t<.150,
                    .14*np.sin(2*np.pi*(c_freq+rep*.4)*t),
                    .14*np.sin(2*np.pi*(v_freq+rep*.3)*t))
    with wave.open(str(path),"wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr)
        w.writeframes((signal*32767).astype("<i2").tobytes())


def _bank(tmp_path, include_vc=True):
    root=tmp_path/"voice"
    for bank in ("D4", "G4"):
        folder=root/bank
        folder.mkdir(parents=True)
        lines=[]
        for cv,c,v,c_freq,v_freq in (
            ("sha","sh","a",550,320),
            ("shu","sh","u",1800,450),
            ("ma","m","a",250,520),
        ):
            for n in range(4):
                wav=f"{cv}_{n}.wav" # same template across pitch/style banks
                _tone(folder/wav,c_freq,v_freq,100*n+(0 if bank=="D4" else 11))
                lines.append(f"{wav}={bank}- {cv},40,100,-360,110,40")
        if include_vc:
            _tone(folder/"existing_vc.wav",900,300,99)
            lines.append(f"existing_vc.wav={bank}- a sh,40,100,-360,110,40")
        (folder/"oto.ini").write_text("\n".join(lines)+"\n")
    return root


def test_cv_oto_timing_and_do_not_learn_existing_vc(tmp_path):
    root=_bank(tmp_path)
    rows,meta=extract_cv(root,adapter="mandarin")
    assert len(rows)==24
    assert meta["selected_full_CV_oto_rows"]==24
    assert meta["excluded"]["not_a_declared_full_CV"]==2
    assert all(row["cv"] in ("sha","shu","ma") for row in rows)
    assert all(row["C"] in ("sh","m") for row in rows)
    assert all(row["C_interval_ms"][0] > 40 and row["C_interval_ms"][1]<150 for row in rows)
    assert all(row["V_interval_ms"][0]>150 for row in rows)
    result=analyze_candidates(rows)
    sh=next(r for r in result["results"]
            if r["phoneme_role"]=="C" and r["author_phoneme"]=="sh")
    assert sh["contexts"]==["a","u"]
    assert len(sh["candidate_groupings"])==2
    assert sh["distinct_source_templates"]==8
    assert sh["source_template_holdout"]["status"]=="exploratory_heldout_context_prediction_only"
    assert len(sh["source_template_holdout"]["k_results"])==2
    assert sh["source_template_holdout"]["k_results"][0]["k"]==1
    assert sh["source_template_holdout"]["k_results"][1]["k"]==2
    assert sh["allophone_split_status"]=="CANDIDATE_ONLY_NOT_VALIDATED"
    vowel_a=next(r for r in result["results"]
                 if r["phoneme_role"]=="V" and r["author_phoneme"]=="a")
    assert vowel_a["contexts"]==["m","sh"]
    assert not result["new_aliases"]
    assert result["VC_marginal_value"].startswith("not_yet_evaluated")
    out=tmp_path/"results"
    export_report(root,out,result,meta)
    assert (out/"allophone_candidates.json").exists()
    assert (out/"allophone_review.csv").exists()
    assert (root/"D4"/"oto.ini").is_file()
    with pytest.raises(FileExistsError):
        export_report(root,out,result,meta)
    with pytest.raises(ValueError,match="outside"):
        export_report(root,root/"results",result,meta)


def test_author_adapter_supports_opaque_multilingual_cv_symbols(tmp_path):
    root=_bank(tmp_path)
    config=tmp_path/"lexicon.json"
    config.write_text(json.dumps({
        "schema":LEXICON_SCHEMA,"alias_prefixes":["D4- ","G4- "],
        "cv":[{"cv":"sha","consonant":"ʂ","vowel":"A"},
              {"cv":"shu","consonant":"ʂ","vowel":"U"},
              {"cv":"ma","consonant":"m","vowel":"A"}]
    },ensure_ascii=False))
    rows,meta=extract_cv(root,adapter="author",lexicon_path=config)
    assert len(rows)==24
    assert {r["C"] for r in rows}=={"ʂ","m"}
    assert {r["V"] for r in rows}=={"A","U"}
    assert meta["selection_source"]=="author_CV_lexicon"
    assert all("existing_vc" not in r["wav"] for r in rows)


def test_oto_alias_not_ground_truth_for_allophones():
    lex={"sha":("sh","a"),"shu":("sh","u")}
    assert _cv_from_entry("D4- sha",lexicon=lex,prefixes=["D4- "],adapter="author")==("sha","sh","a")
    assert _cv_from_entry("D4- a sh",lexicon=lex,prefixes=["D4- "],adapter="author") is None
    assert _cv_from_entry("D4- shu",lexicon=lex,prefixes=["D4- "],adapter="author")==("shu","sh","u")


def test_duplicate_source_aliases_collapse_not_independent():
    rows=[]
    for v in ("a","u"):
        for n in range(16):
            rows.append({"C":"sh","V":v,"bank":"D4","pcm_sha256":"same-pcm",
                         "source_template":"same.wav","wav":"same.wav",
                         "cv":"sh"+v,"oto_alias_for_provenance":f"repeat-{n}",
                         "oto_line":n+1,"C_feature":[.1 if v=="a" else .8]*36,
                         "V_feature":None})
    res=analyze_candidates(rows)["results"]
    assert len(res)==1
    item=res[0]
    assert item["distinct_source_templates"]==1
    assert item["source_group_observations"]==2
    assert item["source_template_holdout"]["status"]=="insufficient_source_template_replication"


def test_no_oracle_test_audio_in_context_grouping():
    means={"a":np.zeros(3),"i":np.ones(3)*.1,"u":np.ones(3)*3.}
    p=_context_partition(means,2)
    assert p["a"]==p["i"] and p["u"]!=p["a"]
    # One target context with no other family to predict from is abstained
    items=[
        {"context":"a","bank":"D4","source_family":"a.wav","vector":np.zeros(3)},
        {"context":"u","bank":"D4","source_family":"u.wav","vector":np.ones(3)}
    ]
    assert _fold_scores(items,max_k=2)["k_results"]==[]


def test_duplicate_lexicon_cv_is_rejected(tmp_path):
    cfg=tmp_path/"lex.json"
    data={"schema":LEXICON_SCHEMA,"cv":[
        {"cv":"a","consonant":"C","vowel":"V"},
        {"cv":"a","consonant":"C","vowel":"V"}],"alias_prefixes":[]}
    cfg.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="duplicate"):
        read_lexicon(cfg)


def test_small_or_silent_CV_region_abstains(tmp_path):
    root=_bank(tmp_path,include_vc=False)
    oto=root/"D4"/"oto.ini"
    lines=oto.read_text().splitlines()
    lines[0]=lines[0].replace(",40,100,-360,110,40",",40,100,-360,30,40")
    oto.write_text("\n".join(lines))
    rows,meta=extract_cv(root,adapter="mandarin")
    assert meta["excluded"]["consonant_region_unusable"]>=1
    assert len(rows)==23


def test_language_adapter_is_opt_in(tmp_path):
    root=_bank(tmp_path,include_vc=False)
    with pytest.raises(ValueError,match="needs --cv-lexicon"):
        extract_cv(root,adapter="author")
