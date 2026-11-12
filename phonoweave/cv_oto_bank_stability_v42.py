"""CV OTO v4.2 — transductive cross-pitch-bank stability, NOT split validity.

A single original recording template may recur in 4 rendered pitch banks.
v4.1 source-TEMPLATE-disjoint prediction must therefore abstain. This is a
separate, weaker diagnostic: hold out a whole bank and test whether grouping
the same author-C/V contexts generalizes in spectral shape. Do NOT interpret
different banks as independent singers, recording takes or new phone labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .cv_oto_allophones_v41 import extract_cv, _aggregate, _context_partition

SCHEMA="phonoweave.cv_oto_bank_stability.v4.2"


def _evaluate_one(role: str, sound: str, items: list[dict], *, max_k: int = 4) -> dict:
    """Bank-held-out, per-bank centered, fixed-context acoustic approximation.

    Test-bank mean is estimated from the SAME fixed group of known contexts,
    i.e. test-input normalization is TRANSDUCTIVE. No source-family disjoint
    generalization is claimed. Grouping is fitted on training banks only.
    """
    if role not in ("C","V"):
        raise ValueError("invalid phoneme role")
    grouped=defaultdict(list)
    for item in items:
        vec=np.asarray(item["vector"],dtype=np.float64)
        if vec.shape!=(36,) or not np.isfinite(vec).all():
            raise ValueError("expected finite 36-dimensional source features")
        grouped[(item["bank"],item["context"])].append(vec)
    banks=sorted({k[0] for k in grouped})
    counts={bank:sorted(context for b,context in grouped if b==bank) for bank in banks}
    common=set(counts[banks[0]]) if banks else set()
    for bank in banks[1:]:
        common &= set(counts[bank])
    contexts=sorted(common)
    summary={
        "phoneme_role":role,"author_phoneme":sound,
        "banks":banks,
        "bank_count":len(banks),
        "common_context_count":len(contexts),
        "common_contexts":contexts,
        "all_bank_context_counts":{b:len(counts[b]) for b in banks},
        "distinct_source_templates":len({o["source_family"] for o in items}),
        "results":[],
        "status":"insufficient_cross_bank_context_coverage",
        "automatic_unit_decision":"UNRESOLVED",
    }
    if len(banks)<3 or len(contexts)<2:
        return summary

    # Exactly equal set of contexts for every bank. Equal bank/context weight,
    # not repeated OTO aliases or physical WAV frequency.
    raw={
        bank:{c:np.mean(grouped[(bank,c)],axis=0) for c in contexts}
        for bank in banks
    }
    centered={}
    for bank in banks:
        bank_mean=np.mean([raw[bank][c] for c in contexts],axis=0)
        centered[bank]={c:raw[bank][c]-bank_mean for c in contexts}

    kmax=min(max_k,len(contexts))
    # All Ks are evaluated on exactly the same held-out banks and contexts.
    errors={k:[] for k in range(1,kmax+1)}
    fold_detail=[]
    for test_bank in banks:
        donor_banks=[bank for bank in banks if bank != test_bank]
        prototypes={
            c:np.mean([centered[bank][c] for bank in donor_banks],axis=0)
            for c in contexts
        }
        fold={"heldout_bank":test_bank,"k_results":[]}
        for k in range(1,kmax+1):
            assignment=_context_partition(prototypes,k)
            clusters=defaultdict(list)
            for c in contexts:
                clusters[assignment[c]].append(c)
            cluster_protos={
                group:np.mean([prototypes[c] for c in members],axis=0)
                for group,members in clusters.items()
            }
            prediction_error=float(np.mean([
                np.mean((centered[test_bank][c]-cluster_protos[assignment[c]])**2)
                for c in contexts
            ]))
            errors[k].append(prediction_error)
            fold["k_results"].append({
                "k":k,
                "mse":round(prediction_error,9),
                "training_banks_only_context_groups":[sorted(g) for g in clusters.values()],
            })
        fold_detail.append(fold)
    primary=float(np.mean(errors[1]))
    summary["results"]=[
        {
            "k":k,
            "bank_heldout_transductive_mse":round(float(np.mean(vals)),9),
            "k1_minus_k_mse":round(primary-float(np.mean(vals)),9),
            "fold_count":len(vals),
            "per_bank_heldout_mse":{
                b:round(val,9) for b,val in zip(banks,vals)
            }
        }
        for k,vals in errors.items()
    ]
    summary["fold_details"]=fold_detail
    summary["status"]="CORRELATED_CROSS_BANK_PROXY_ONLY"
    summary["test_bank_centering"]="label_unaware_transductive_mean_across_same_known_CV_contexts"
    summary["independent_human_singer_sessions"]=0
    summary["split_or_merge_approved"]=False
    return summary


def analyze_bank_stability(rows: list[dict]) -> dict:
    report=[]
    for role in ("C","V"):
        by_phoneme=_aggregate(rows,role)
        for phoneme, items in sorted(by_phoneme.items()):
            if len({o["context"] for o in items})<2:
                continue
            report.append(_evaluate_one(role,phoneme,items))
    return {
        "schema":SCHEMA,
        "source":"CV-only OTO-timed acoustic profiles",
        "evaluation":"heldout_pitch_subbank_with_transductive_context_matched_centering",
        "phonemes_screened":len(report),
        "phonemes_with_proxy_scores":sum(r["status"]=="CORRELATED_CROSS_BANK_PROXY_ONLY" for r in report),
        "phonemes_insufficient":sum(r["status"]!="CORRELATED_CROSS_BANK_PROXY_ONLY" for r in report),
        "items":report,
        "new_aliases":[],
        "new_recording_lines":[],
        "existing_VC_used_to_train":False,
        "limitations":[
            "Training and test banks may share original utterance templates, recording sessions, voice model, and singer.",
            "A bank-held-out result is NOT source-template-heldout evidence and cannot replace v4.1 abstentions.",
            "Test-bank centering is transductive and uses known CV contexts; not a blind waveform classifier.",
            "OTO timing may be imperfect and normal C/V coarticulation is present.",
            "This tests spectral-context repeatability, NOT whether extra recorded phonetic units improve synthesis.",
            "K=1..4 labels are candidate complexity probes, not a proven optimal number of allophones.",
            "VC unit retention/removal requires separate controlled resynthesis.",
        ],
    }


def export_report(root:Path,out:Path,data:dict,scan:dict)->Path:
    root=Path(root).expanduser().resolve(strict=True)
    out=Path(out).expanduser().resolve()
    if out==root or out.is_relative_to(root):
        raise ValueError("output must be outside source voicebank")
    if out.exists():
        raise FileExistsError("output already exists")
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pw-bank-proxy-",dir=out.parent) as td:
        stage=Path(td)/"export"
        stage.mkdir()
        (stage/"cross_bank_proxy.json").write_text(
            json.dumps({**data,"scan":scan},ensure_ascii=False,allow_nan=False,indent=2)+"\n",
            encoding="utf-8")
        with (stage/"cross_bank_review.csv").open("x",newline="",encoding="utf-8") as f:
            writer=csv.writer(f)
            writer.writerow(("role","phoneme","banks","contexts","status",
                             "k1_mse","k2_gain_proxy","k3_gain_proxy","k4_gain_proxy",
                             "split_decision"))
            for row in data["items"]:
                observed={x["k"]:x for x in row["results"]}
                values=[row["phoneme_role"],row["author_phoneme"],
                        row["bank_count"],row["common_context_count"],row["status"],
                        observed.get(1,{}).get("bank_heldout_transductive_mse"),
                        *(observed.get(k,{}).get("k1_minus_k_mse") for k in (2,3,4)),
                        "UNRESOLVED"]
                writer.writerow([
                    ("'"+v) if isinstance(v,str) and v.lstrip().startswith(("=","+","-","@")) else v
                    for v in values])
        os.rename(stage,out)
    return out


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(
        description="OTO-timed CV allophones: weaker bank-heldout proxy, never actual split approval")
    p.add_argument("voicebank",type=Path)
    p.add_argument("--adapter",choices=("mandarin","author"),default="mandarin")
    p.add_argument("--cv-lexicon",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    try:
        rows,scan=extract_cv(args.voicebank,adapter=args.adapter,
                             lexicon_path=args.cv_lexicon)
        if not rows:
            raise ValueError("no accepted CV-OTO segments")
        result=analyze_bank_stability(rows)
        export_report(args.voicebank,args.output,result,scan)
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
        p.error(str(exc))
    print(json.dumps({
        "phonemes":result["phonemes_screened"],
        "proxy_scores":result["phonemes_with_proxy_scores"],
        "real_split_decisions":0,
        "out":str(args.output)},ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
