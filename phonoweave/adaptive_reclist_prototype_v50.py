#!/usr/bin/env python3
"""Generate a *prototype* singer-adaptive CV recording seed from v4.2 acoustic proxy.

Unlike earlier iterations, this actually emits a recordable CSV and a syllable
list. It is a coverage experiment, NOT a validated optimal recording scheme.
OTO-timed C/V spectral evidence was produced upstream; legacy VC splits do not
contribute to the grouping objective. No voicebank sources are edited.

Input: cross_bank_proxy.json from CV/OTO v4.2.
Output: a core CV edge-cover, extra context-probe CVs, grouped allophone
hypotheses, and preserved current-VC inventory status (untested).

Uses SciPy (a declared PhonoWeave project dependency) for exact integer
set-cover minimization. Stable tie-breaks are not phonological claims.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

SCHEMA = "phonoweave.accent_adaptive_reclist.preview.v0.1"
SOURCE_SCHEMA = "phonoweave.cv_oto_bank_stability.v4.2"


def _ari(first, second):
    """Permutation-invariant adjusted Rand agreement of context partitions."""
    a = {ctx: idx for idx, group in enumerate(first) for ctx in group}
    b = {ctx: idx for idx, group in enumerate(second) for ctx in group}
    if set(a) != set(b):
        raise ValueError("inconsistent context labels between heldout bank folds")
    n = len(a)
    if n < 2:
        return 0.0
    counts_a, counts_b, joint = Counter(a.values()), Counter(b.values()), Counter((a[c], b[c]) for c in a)
    choose = lambda z: z * (z-1) / 2
    observed = sum(choose(x) for x in joint.values())
    aa, bb = sum(choose(x) for x in counts_a.values()), sum(choose(x) for x in counts_b.values())
    expected = aa * bb / choose(n)
    ceiling = (aa + bb) / 2
    return (observed - expected) / (ceiling - expected) if ceiling > expected else 1.0


def _normalized_partition(groups):
    return tuple(sorted(tuple(sorted(group)) for group in groups))


def _fold_partitions(item, k):
    found = []
    for fold in item["fold_details"]:
        entry = next((z for z in fold["k_results"] if z["k"] == k), None)
        if entry is None:
            return []
        found.append(_normalized_partition(entry["training_banks_only_context_groups"]))
    return found


def _stability(item, k):
    folds = _fold_partitions(item, k)
    if len(folds) < 3:
        return 0.0
    return sum(_ari(a,b) for a,b in itertools.combinations(folds,2)) / (len(folds)*(len(folds)-1)/2)


def _winner(item,k):
    base = item["results"][0]
    candidate = next((r for r in item["results"] if r["k"] == k), None)
    if not candidate:
        return None
    mse = base["bank_heldout_transductive_mse"]
    if mse is None or mse <= 0:
        return None
    wins = sum(candidate["per_bank_heldout_mse"][bank] < base["per_bank_heldout_mse"][bank]
               for bank in item["banks"])
    return {"k": k, "relative_proxy_gain": float(candidate["k1_minus_k_mse"] / mse),
            "folds_better": wins, "fold_count": len(item["banks"]),
            "partition_ari": float(_stability(item,k))}


def choose_hypotheses(report):
    """Conservative data-driven acoustic hypotheses (not categorical phone labels).

    K=2 requires >=10% transductive proxy error reduction, >=3/4 heldout
    banks lower error, ARI>=.7 across training-bank-only partitions, and >=2
    preceding/following contexts in each group. K=3 needs an EXTRA >=6pp
    relative gain over K=2 and the same stability constraints.
    """
    chosen = []
    audit = []
    for item in report["items"]:
        role, name = item["phoneme_role"], item["author_phoneme"]
        key = role + ":" + name
        if not item["results"]:
            audit.append({"phoneme": key, "status": "abstain", "reason": "no_proxy_score"})
            continue
        scored2 = _winner(item, 2)
        best = None
        for k in (2,3):
            w = _winner(item,k)
            if w is None:
                continue
            folds = _fold_partitions(item,k)
            if not folds or min(len(group) for p in folds for group in p) < 2:
                continue
            if (w["relative_proxy_gain"] < .10 or w["folds_better"] < 3
                    or w["partition_ari"] < .70):
                continue
            if k == 3:
                if best is None or w["relative_proxy_gain"] - best["relative_proxy_gain"] < .06:
                    continue
            best = w
        if best is None:
            audit.append({"phoneme":key, "status":"hold_shared_for_pilot",
                          "reason":"insufficient_proxy_gain_consistency_or_context_coverage",
                          "K2":scored2})
            continue
        folds = _fold_partitions(item,best["k"])
        frequency = Counter(folds)
        # Stable most frequent fold proposal; no existing VC alias input.
        canonical, votes = sorted(frequency.items(), key=lambda pair:(-pair[1],pair[0]))[0]
        groups = [list(group) for group in canonical]
        chosen.append({"phoneme":name,"role":role,"k":best["k"],
                       "groups":groups,"relative_proxy_gain":round(best["relative_proxy_gain"],4),
                       "folds_better":best["folds_better"],
                       "fold_count":best["fold_count"],
                       "partition_ari":round(best["partition_ari"],3),
                       "modal_partition_votes":votes,
                       "status":"PILOT_SPLIT_HYPOTHESIS_NOT_VERIFIED"})
        audit.append({"phoneme":key,"status":"pilot_split_hypothesis", "K":best["k"]})
    return chosen,audit


def _attested_graph(report):
    C={r["author_phoneme"]: set(r["common_contexts"]) for r in report["items"]
       if r["phoneme_role"]=="C"}
    V={r["author_phoneme"]: set(r["common_contexts"]) for r in report["items"]
       if r["phoneme_role"]=="V"}
    edges=sorted((c,v) for c,vl in C.items() for v in vl if c in V.get(v,set()))
    if not edges or any(not any(edge[0]==c for edge in edges) for c in C) or any(
        not any(edge[1]==v for edge in edges) for v in V):
        raise ValueError("cannot build CV cover: unattested C or V")
    return C,V,edges


def _minimum_cover(edges, required, edge_covers, fixed=()):
    """Solve exact binary set cover, with fixed preselected CVs when requested."""
    constraints=np.zeros((len(required),len(edges)), dtype=np.float64)
    index={n:i for i,n in enumerate(required)}
    for j,edge in enumerate(edges):
        for node in edge_covers[edge]:
            if node in index:
                constraints[index[node],j]=1
    if np.any(constraints.sum(axis=1)==0):
        absent=[required[i] for i in np.where(constraints.sum(axis=1)==0)[0]]
        raise ValueError("no attested CV represents required context: " + str(absent[:5]))
    lower=np.zeros(len(edges))
    for edge in fixed:
        lower[edges.index(edge)] = 1
    # All coefficient jitter is < .001 total to retain exact minimal LINE count.
    weights=np.array([1.0 + (int(hashlib.sha256((c+":"+v).encode()).hexdigest()[:6],16)
                           % 1000) / 1e8 for c,v in edges])
    answer=milp(c=weights, integrality=np.ones(len(edges)),
                bounds=Bounds(lower,np.ones(len(edges))),
                constraints=LinearConstraint(constraints,np.ones(len(required)),np.full(len(required),np.inf)),
                options={"time_limit":15, "mip_rel_gap":0.001})
    if answer.x is None or not answer.success:
        raise ValueError("CV cover integer program did not solve")
    selection={edge for edge,used in zip(edges,answer.x) if used>.5}
    if any(not any(node in edge_covers[e] for e in selection) for node in required):
        raise AssertionError("unsatisfied cover")
    return selection


def compile_scheme(report):
    if report.get("schema") != SOURCE_SCHEMA or not isinstance(report.get("items"),list):
        raise ValueError("expected PhonoWeave CV OTO cross-bank v4.2 JSON")
    C,V,edges=_attested_graph(report)
    chosen,audit=choose_hypotheses(report)
    phoneme_nodes={"C:"+c for c in C} | {"V:"+v for v in V}
    coverage={edge:{"C:"+edge[0],"V:"+edge[1]} for edge in edges}
    pilot_groups={}
    for row in chosen:
        p=row["role"]+":"+row["phoneme"]
        for group_id,contexts in enumerate(row["groups"],start=1):
            node=p+":variant_"+str(group_id)
            pilot_groups[node]=contexts
            for c,v in edges:
                if (row["role"]=="C" and c==row["phoneme"] and v in contexts
                    or row["role"]=="V" and v==row["phoneme"] and c in contexts):
                    coverage[(c,v)].add(node)
    baseline=_minimum_cover(edges,sorted(phoneme_nodes),coverage)
    adaptive=_minimum_cover(edges,sorted(phoneme_nodes|set(pilot_groups)),coverage,baseline)
    output=[]
    for index,(c,v) in enumerate(sorted(adaptive,key=lambda e:(e not in baseline,e[0],e[1])),1):
        output.append({"order":index,"recording_token":c+v,
                       "consonant":c,"vowel":v,
                       "phase":"CORE" if (c,v) in baseline else "ACCENT_EXTRA",
                       "covered_variants":" | ".join(sorted(coverage[(c,v)] & set(pilot_groups))),
                       "status":"REUSE_EXISTING_CV_OR_RECORD_FOR_NEXT_SINGER",
                       "notes":"Original C/V orthographic label only; verify pronunciation before new session"})
    return {
        "schema":SCHEMA,
        "source_summary":{"phonemes_screened":report["phonemes_screened"],
                          "with_correlated_proxy_scores":report["phonemes_with_proxy_scores"],
                          "VC_existing_unit_retain_policy":"KEEP_ALL_UNTIL_INDEPENDENT_RENDER_ABLATION",
                          "original_source_is_OTO_labeled_CV_only":True},
        "thresholds":{"K2_min_proxy_gain":.10,"min_banks_improved":3,
                      "min_partition_mean_ARI":.7,"min_group_contexts":2,
                      "K3_extra_proxy_gain_over_K2":.06},
        "core_CV_count":len(baseline),"accent_extra_CV_count":len(adaptive-baseline),
        "total_recording_tokens":len(adaptive),"observed_CV_tokens":len(edges),
        "author_C_coverage":len(C),"author_V_coverage":len(V),
        "pilot_hypotheses":chosen,
        "candidates_not_selected":audit,
        "recording_lines":output,
        "cannot_claim":"This is a pilot CV coverage + context-condition sampling plan, not validated minimum usable full singing voicebank or established accent-dependent allophone taxonomy. No existing VC deleted.",
    }


def export(scheme,output):
    output=Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite output directory")
    output.mkdir(parents=True)
    (output/"scheme.json").write_text(json.dumps(scheme,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for name,rows,keys in [
        ("recording_plan.csv",scheme["recording_lines"],list(scheme["recording_lines"][0])),
        ("phoneme_split_proposals.csv",[
            {"role":r["role"],"phoneme":r["phoneme"],"K":r["k"],
             "groups":" / ".join("{"+", ".join(g)+"}" for g in r["groups"]),
             "proxy_gain_percent":round(100*r["relative_proxy_gain"],1),
             "banks_improved":str(r["folds_better"])+"/"+str(r["fold_count"]),
             "group_stability_ARI":r["partition_ari"],"status":r["status"]}
            for r in scheme["pilot_hypotheses"]],
         ["role","phoneme","K","groups","proxy_gain_percent","banks_improved","group_stability_ARI","status"]),
    ]:
        with (output/name).open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow({k:("'"+str(r.get(k,"")) if str(r.get(k,"")).lstrip().startswith(("=","+","-","@")) else r.get(k,"")) for k in keys})
    (output/"recording_tokens.txt").write_text("\n".join(r["recording_token"] for r in scheme["recording_lines"]) +"\n",encoding="utf-8")
    lines=["# PhonoWeave — 个人 CV 录音方案草案 v0.1", "",
           "**这是已经生成的录音 token 清单，而不是又一份频谱报告。**", "",
           f"- 基础音位覆盖：{scheme['core_CV_count']} 条 CV",
           f"- 口音相关声学分组的额外覆盖：{scheme['accent_extra_CV_count']} 条 CV",
           f"- 合计：{scheme['total_recording_tokens']} 条（每种声区需另行安排，不代表成品声库仅需这么多条）。",
           f"- 用于选择的已见 CV：{scheme['observed_CV_tokens']} 种；覆盖 {scheme['author_C_coverage']} 个 C 标签和 {scheme['author_V_coverage']} 个 V 标签。",
           "", "## 试验性拆音建议（声学假设，非已经确认的同位音）", "",
           "| 音位 | 暂定分组 | K=1→K 改善（代理误差） | 音高声库中改善 |", "|---|---|---:|---:|"]
    for r in scheme["pilot_hypotheses"]:
        groups="; ".join("("+", ".join(g)+")" for g in r["groups"])
        lines.append(f"| {r['role']} / {r['phoneme']} | {groups} | {r['relative_proxy_gain']*100:.1f}% | {r['folds_better']}/{r['fold_count']} |")
    lines += ["", "## 录音表（复用现有 CV 或给下一位歌手作为初录目标）", "",
              "Core 先录；Accent Extra 是补齐代理模型建议的发音环境，**不代表一定要新增独立音素命名**。", "",
              "| 顺序 | CV | 阶段 | 对应的额外同位音假设覆盖 |", "|---:|---|---|---|"]
    for r in scheme["recording_lines"]:
        lines.append(f"| {r['order']} | `{r['recording_token']}` | {r['phase']} | {r['covered_variants'] or '基础覆盖'} |")
    lines += ["", "## 已有 VC", "", "全部保持原状。已有 VC 的录音价值尚未经过可靠的音频替代实验验证；本版本不删除任何 VC。",
             "", "## 科研边界", "", "候选分组来源于用户的 v4.2 同歌手跨音高频谱代理模型；这些音高声库不是独立歌手/录音。文中改善是代理误差改善，不是感知音质、真实合成效果或录音成本节省的百分比。",
             "", "此表是可以立刻使用的 CV 覆盖试验草案，但不是最终的最佳 CVVC/VCV 全录音表。后续通过真实合成结果替换假设后可重生成。"]
    (output/"README_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description="Generate an actual pilot CV recording list from CV/OTO accent proxy")
    p.add_argument("--proxy",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    report=json.loads(a.proxy.read_text(encoding="utf-8"))
    plan=compile_scheme(report)
    export(plan,a.output)
    print(json.dumps({k:plan[k] for k in ("core_CV_count","accent_extra_CV_count","total_recording_tokens")},ensure_ascii=False))


if __name__=="__main__":
    main()