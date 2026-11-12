"""v4.2 weak cross-bank context reproducibility, not independent singer tests."""
from __future__ import annotations

import json
import numpy as np
import pytest

from phonoweave.cv_oto_bank_stability_v42 import (
    SCHEMA, _evaluate_one, analyze_bank_stability, export_report
)
from phonoweave.cv_oto_allophones_v41 import analyze_candidates


def _rows(n_banks=4, *, shared_template=True):
    out=[]
    for bank_i in range(n_banks):
        bank=f"B{bank_i}"
        for ctx,shape in (("a",0.0),("i",0.1),("u",1.8)):
            vec=np.zeros(36,dtype=float)
            vec[0]=shape+bank_i*.2
            vec[10]=shape*.7+bank_i*.2
            path=f"{ctx}.wav" if shared_template else f"{bank}_{ctx}.wav"
            out.append({
                "C":"sh","V":ctx,"bank":bank,
                "source_template":path,
                "pcm_sha256":f"unique_pcm_{bank}_{ctx}",
                "cv":"sh"+ctx,"wav":f"{bank}/{path}",
                "oto_alias_for_provenance":f"{bank}sh{ctx}","oto_line":1,
                "C_feature":vec.tolist(),"V_feature":None,
            })
    return out


def test_reuses_correlated_banks_only_without_pretending_source_holdout():
    rows=_rows()
    previous=analyze_candidates(rows)
    strict=next(x for x in previous["results"]
                if x["phoneme_role"]=="C" and x["author_phoneme"]=="sh")
    assert strict["source_template_holdout"]["status"]=="insufficient_source_template_replication"
    result=analyze_bank_stability(rows)
    assert result["schema"]==SCHEMA
    assert result["phonemes_screened"]==1
    assert result["phonemes_with_proxy_scores"]==1
    sh=result["items"][0]
    assert sh["status"]=="CORRELATED_CROSS_BANK_PROXY_ONLY"
    assert sh["common_contexts"]==["a","i","u"]
    assert sh["distinct_source_templates"]==3
    assert len(sh["results"])==3
    assert sh["results"][1]["k"]==2
    assert sh["results"][1]["k1_minus_k_mse"]>0
    assert all(x["fold_count"]==4 for x in sh["results"])
    assert sh["independent_human_singer_sessions"]==0
    assert not sh["split_or_merge_approved"]
    assert result["new_aliases"]==[] and result["new_recording_lines"]==[]


def test_insufficient_banks_abstain():
    result=analyze_bank_stability(_rows(2))
    assert result["phonemes_with_proxy_scores"]==0
    assert result["phonemes_insufficient"]==1
    assert not result["items"][0]["results"]


def test_no_common_context_abstains():
    rows=_rows()
    rows=[r for r in rows if not (r["bank"]=="B3" and r["V"]!="u")]
    result=analyze_bank_stability(rows)
    assert result["phonemes_insufficient"]==1
    assert result["items"][0]["common_context_count"]==1


def test_invalid_vector_fails():
    rows=_rows()
    rows[0]["C_feature"]=[float("nan")]*36
    with pytest.raises(ValueError,match="acoustic vector"):
        analyze_bank_stability(rows)


def test_report_output_guard_and_no_voicebank_edits(tmp_path):
    root=tmp_path/"original"
    root.mkdir()
    report=analyze_bank_stability(_rows())
    target=tmp_path/"report"
    export_report(root,target,report,{"mock":True})
    assert (target/"cross_bank_review.csv").exists()
    saved=json.loads((target/"cross_bank_proxy.json").read_text())
    assert saved["existing_VC_used_to_train"] is False
    assert not list(root.iterdir())
    with pytest.raises(FileExistsError):
        export_report(root,target,report,{})
    with pytest.raises(ValueError,match="outside"):
        export_report(root,root/"inside",report,{})
