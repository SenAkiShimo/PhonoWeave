"""Accent-context v3.3: provisional comparison, strict source independence."""
from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from phonoweave.accent_context_comparison_v33 import (
    MAP_SCHEMA, SCHEMA, compare_contexts, export, load_context_map, scan_voicebank,
)


def _row(bank, base, context, source, feature):
    return {"bank": bank, "base_unit": base, "context": context,
            "source_pcm_sha256": source, "original_alias": base + "-" + context,
            "original_wav": source + ".wav", "stimulus": base + context,
            "feature": list(feature)}


def _rows(shift=1.8, n=4):
    rows=[]
    for context, bias in (("plain", 0), ("rounded", shift)):
        for i in range(n):
            x=np.zeros(36, dtype=float)
            x[0] = bias + i * .05
            x[12] = bias * .8 + i * .05
            rows.append(_row("D4", "sh", context, f"{context}-{i}", x))
    return rows


def test_generalizes_across_unique_pcm_sources():
    result=compare_contexts(_rows())
    assert result["schema"] == SCHEMA
    assert result["candidate_count"] == 1
    c=result["candidate_pairs"][0]
    assert c["contexts"]==["plain","rounded"]
    assert c["eligible_bank_count"]==1
    assert c["mean_shared_minus_separate_mse"] > 0
    assert c["production_split_or_merge_decision"]=="UNRESOLVED"
    assert not c["new_alias_authorized"]
    assert result["new_phonemes"]==[] and result["new_recording_lines"]==[]


def test_one_long_wav_with_many_aliases_is_one_source_group():
    rows=[]
    for context in ("plain","rounded"):
        for _ in range(40):
            rows.append(_row("D4","sh",context,"the_same_pcm",
                             [0 if context=="plain" else 2]+[0]*35))
    result=compare_contexts(rows)
    assert result["candidate_count"]==1
    r=result["candidate_pairs"][0]["bank_results"]["D4"]
    assert r["source_groups"]=={"plain":1, "rounded":1}
    assert r["status"]=="insufficient_independent_PCM_groups"
    assert r["mean_shared_minus_separate_mse"] is None


def test_feature_reconstruction_without_source_replication_abstains():
    result=compare_contexts(_rows(n=2))
    c=result["candidate_pairs"][0]
    assert c["eligible_bank_count"]==0
    assert c["mean_shared_minus_separate_mse"] is None


def test_different_base_labels_are_not_force_merged():
    rows=_rows()
    for r in rows:
        if r["context"] == "rounded":
            r["base_unit"]="shw"
    result=compare_contexts(rows)
    assert result["candidate_count"]==0


def test_source_group_cannot_train_on_itself_across_contexts():
    rows = []
    for i in range(3):
        pcm=f"shared-source-{i}"
        rows.append(_row("D4","sh","plain",pcm,[i*.01]*36))
        rows.append(_row("D4","sh","rounded",pcm,[1+i*.01]*36))
    c=compare_contexts(rows)["candidate_pairs"][0]
    assert c["eligible_bank_count"]==1
    assert c["bank_results"]["D4"]["heldout_cases"]==6


def test_csv_safety_and_no_modification_to_bank(tmp_path):
    root=tmp_path/"voice";root.mkdir()
    report=compare_contexts([
        _row("D4", "=bad", "plain", "a", [0]*36),
        _row("D4", "=bad", "rounded", "b", [1]*36),
    ])
    dest=tmp_path/"results"
    export(report,dest,root)
    text=(dest/"context_review_sheet.csv").read_text()
    assert "'=bad" in text
    assert (dest/"context_candidates.json").is_file()
    assert not list(root.iterdir())
    with pytest.raises(FileExistsError):
        export(report,dest,root)
    with pytest.raises(ValueError,match="outside"):
        export(report,root/"inside",root)


def test_author_map_exact_aliases_and_duplicate_rejection(tmp_path):
    path=tmp_path/"cases.json"
    cfg={"schema":MAP_SCHEMA,"cases":[
        {"bank":"D4","original_alias":"sh-u","base_unit":"sh",
         "context":"rounded","stimulus":"shu"}]}
    path.write_text(json.dumps(cfg))
    assert load_context_map(path)[("D4","sh-u")]["context"]=="rounded"
    cfg["cases"].append(cfg["cases"][0])
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match="duplicate"):
        load_context_map(path)


def _wav(path,freq):
    path.parent.mkdir(parents=True,exist_ok=True)
    sr=48000
    t=np.arange(sr,dtype=np.float64)/sr
    x=(0.20*np.sin(2*np.pi*freq*t)*32767).astype("<i2")
    with wave.open(str(path),"wb") as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr)
        f.writeframes(x.tobytes())


def test_real_pcm_read_with_author_context_map(tmp_path):
    root=tmp_path/"voice"; bank=root/"D4";bank.mkdir(parents=True)
    lines=[];cases=[]
    for context,freq in (("plain",700),("rounded",1500)):
        for i in range(4):
            alias=f"sh_{context}_{i}"
            wav=f"{context}_{i}.wav"
            _wav(bank/wav,freq + 2*i)
            lines.append(f"{wav}={alias},100,100,100,100,50")
            cases.append({"bank":"D4","original_alias":alias,
                          "base_unit":"sh","context":context,
                          "stimulus":f"test-{context}"})
    (bank/"oto.ini").write_text("\n".join(lines)+"\n")
    cfg=tmp_path/"cases.json"
    cfg.write_text(json.dumps({"schema":MAP_SCHEMA,"cases":cases}))
    rows,metadata=scan_voicebank(root,context_map=cfg)
    assert metadata["usable_candidate_regions"]==8
    assert metadata["source_mode"]=="author_exact_context_map"
    result=compare_contexts(rows)
    assert result["candidate_count"]==1
    assert result["candidate_pairs"][0]["eligible_bank_count"]==1
    assert result["candidate_pairs"][0]["production_split_or_merge_decision"]=="UNRESOLVED"
