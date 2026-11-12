"""v2.5 equal-budget test: prevent same-WAV OTO rows from faking replication."""
import csv
import io
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from phonoweave.native_unit_budget_v25 import (
    SCHEMA, compare_unit_pairs, run_archive, attach_unit_budget_feasibility,
)


def row(label, pcm, value, bank="D4", ctx="sh"):
    return {"bank":bank, "symbol":label, "context":ctx,
            "source_pcm_sha256":pcm,
            "features":tuple([float(value)] * 12)}


def fixture(count=1, duplicate_oto=0, labels=("i", "ir")):
    rows = []
    for n in range(count):
        source = chr(ord("a")+n)*64
        for symbol, base in zip(labels, (0.0, 2.0)):
            rows.append(row(symbol,source,base+0.1*n))
    for _ in range(duplicate_oto):
        rows.extend(deepcopy(rows[:2]))
    return rows


def test_one_shared_wav_per_bank_context_is_not_an_independent_donor():
    result=compare_unit_pairs(fixture(1, duplicate_oto=30),[("i","ir")])
    pair=result["compared_existing_label_pairs"][0]
    assert pair["same_context_cohorts_with_both_labels"]==1
    assert pair["eligible_independent_target_sources"]==0
    assert pair["blocked_cohort_count"]==1
    assert pair["status"]=="insufficient_independent_source_repetitions"
    assert result["automatic_unit_changes"]==[]
    assert result["automatic_recording_lines"]==[]


def test_three_distinct_wavs_allow_equal_budget_proxy():
    result=compare_unit_pairs(fixture(3),[("i","ir")])
    pair=result["compared_existing_label_pairs"][0]
    assert pair["eligible_independent_target_sources"]==3
    assert pair["status"]=="exploratory_equal_donor_proxy_only"
    assert pair["mean_pooled_minus_separate_mse"]>0
    assert pair["unit_merge_or_split_authorized"] is False
    assert result["matched_material"].startswith("exact_same_training")


def test_two_wavs_and_budget_two_abstain():
    pair=compare_unit_pairs(fixture(2),[("i","ir")],budget_per_label=2)["compared_existing_label_pairs"][0]
    assert pair["eligible_independent_target_sources"]==0
    assert pair["blocked_cohort_examples"][0]["reason"]=="insufficient_independent_same_context_source_donors"


def test_same_wav_used_in_different_oto_aliases_does_not_increase_sources():
    rows=fixture(1)
    for index in range(10):
        rows.append(row("i","a"*64,index*0.01))
        rows.append(row("ir","a"*64,2+index*0.01))
    pair=compare_unit_pairs(rows,[("i","ir")])["compared_existing_label_pairs"][0]
    assert pair["eligible_independent_target_sources"]==0
    assert pair["blocked_cohort_examples"][0]["distinct_pcm_sources_for_first_label"]==1


def test_no_equal_context_cannot_be_compared():
    rows=[row("i","a"*64,0,ctx="sh"),row("ir","b"*64,2,ctx="m")]
    pair=compare_unit_pairs(rows,[("i","ir")])["compared_existing_label_pairs"][0]
    assert pair["same_context_cohorts_with_both_labels"]==0
    assert pair["mean_pooled_minus_separate_mse"] is None


@pytest.mark.parametrize("budget",[0,True,9,1.0,-1])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError,match="donor budget"):
        compare_unit_pairs(fixture(2),[("i","ir")],budget_per_label=budget)


def test_repeated_or_unknown_label_pair_rejected():
    rows=fixture(2)
    with pytest.raises(ValueError,match="duplicate label pair"):
        compare_unit_pairs(rows,[("i","ir"),("ir","i")])
    with pytest.raises(ValueError,match="existing"):
        compare_unit_pairs(rows,[("i","invented")])


def test_adding_review_does_not_change_existing_reclist_or_reviews(tmp_path):
    original={"recording_plan":{"recording_lines":["录音A"],"review_tasks":[{"line":"修复OTO"}]},
              "existing_sound_unit_review":{"original_user_labels":["i","ir"]}}
    archive=make_archive(tmp_path, fixture(2))
    result=attach_unit_budget_feasibility(original,archive,[("i","ir")])
    assert original["recording_plan"]==result["recording_plan"]
    assert result["existing_unit_budget_feasibility"]["automatic_recording_lines"]==[]
    assert result["existing_unit_budget_feasibility"]["compared_existing_label_pairs"][0]["labels"]==["i","ir"]


def test_pair_not_in_existing_unit_review_refused(tmp_path):
    archive=make_archive(tmp_path, fixture(2))
    with pytest.raises(ValueError,match="pair not"):
        attach_unit_budget_feasibility(
            {"recording_plan":{},"existing_sound_unit_review":{"original_user_labels":["i"]}},
            archive,[("i","ir")],
        )


def make_archive(tmp_path:Path,rows,break_alignment=False,conflicting_hash=False):
    file=tmp_path/"frozen.zip"
    index=[];vectors=[]
    for n,r in enumerate(rows):
        name=f"take_{n}.wav"
        idx={k:r[k] for k in ("bank","symbol","context","source_pcm_sha256")}
        idx.update({"source_wav":name,"oto_line":str(n)})
        index.append(idx)
        vectors.append({"bank":r["bank"],"source_wav":name,
                        "oto_line":n,"features":list(r["features"])})
    if break_alignment: vectors.pop()
    if conflicting_hash and len(index)>1:
        index[1]["source_wav"]=index[0]["source_wav"]
    fields=["bank","symbol","context","source_pcm_sha256","source_wav","oto_line"]
    buffer=io.StringIO()
    w=csv.DictWriter(buffer,fieldnames=fields);w.writeheader();w.writerows(index)
    with zipfile.ZipFile(file,"w") as z:
        z.writestr("results/source_index.csv",buffer.getvalue())
        z.writestr("results/feature_vectors.json",json.dumps(vectors))
    return file


def test_source_archive_alignment_checked(tmp_path):
    archive=make_archive(tmp_path,fixture(2),break_alignment=True)
    with pytest.raises(ValueError,match="row count"):
        run_archive(archive,[("i","ir")])


def test_conflicting_bank_wav_hash_rejected(tmp_path):
    archive=make_archive(tmp_path,fixture(2),conflicting_hash=True)
    with pytest.raises(ValueError,match="alignment mismatch"):
        run_archive(archive,[("i","ir")])


def test_archive_provides_source_digest_and_keeps_mandarin_independent(tmp_path):
    archive=make_archive(tmp_path,fixture(2))
    report=run_archive(archive,[("i","ir")])
    assert report["schema"]==SCHEMA
    assert len(report["source_archive_sha256"])==64
    assert report["automatic_recording_lines"]==[]
