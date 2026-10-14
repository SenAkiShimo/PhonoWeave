"""Budget-matched, WAV-held-out comparison of fixed and adaptive unit splits.

A benchmark protocol, not evidence that acoustic separability implies
rendering gains. A split means retaining both paired-final donor identities;
a merge means a single pooled donor. All strategies choose with the same
training-only data and are scored against held-out real spectral signatures.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import json
import numpy as np
from .synthesis_relevance_unit_merging_cost import extract, _distance
from .synthesis_relevance_final_matched import FINAL_PAIRS,PAIR_OF

SCHEMA="phonoweave.synthesis_relevance.budget_benchmark.v1.1"
SEED=20261013
# Predeclared control policy, not an empirically optimized phonological rank.
RULE_ORDER=("a/ua","an/uan","ai/uai","ang/uang")
PAIR_NAMES=tuple("/".join(pair) for pair in FINAL_PAIRS)

def allocate(units, *, base="sh", min_train_per_final=2):
    """Return observable cells, no sound generation or data-dependent tuning."""
    rng=np.random.default_rng(SEED)
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for u in units:
        if u.final in PAIR_OF:
            pair=PAIR_NAMES[PAIR_OF[u.final][0]]
            groups[u.group][pair][u.final].append(u)
    partitions=[]
    for subbank,by_pair in sorted(groups.items()):
        # WAVs held out *across all pairs in a subbank*, not row-wise.
        wavs=sorted({u.wav_id for finals in by_pair.values() for ls in finals.values() for u in ls})
        shuffled=list(rng.permutation(wavs))
        test_ids=set(shuffled[::5])
        for pair in PAIR_NAMES:
            a,b=pair.split("/")
            finals=by_pair.get(pair,{})
            train={f:[u for u in finals.get(f,[]) if u.wav_id not in test_ids] for f in (a,b)}
            test={f:[u for u in finals.get(f,[]) if u.wav_id in test_ids] for f in (a,b)}
            n_train={f:len({u.wav_id for u in train[f]}) for f in (a,b)}
            n_test={f:len({u.wav_id for u in test[f]}) for f in (a,b)}
            if min(n_train.values())<min_train_per_final or min(n_test.values())<1:
                partitions.append({"subbank":subbank,"pair":pair,"status":"insufficient",
                    "reason":"requires_two_training_WAVs_and_one_heldout_WAV_per_final",
                    "train_WAVs":n_train,"test_WAVs":n_test})
                continue
            partitions.append({"subbank":subbank,"pair":pair,"status":"evaluable",
                "train_WAVs":n_train,"test_WAVs":n_test,
                "train":train,"test":test})
    return partitions

def _mean_signature(rows):
    return np.mean(np.asarray([r.core for r in rows]),axis=0)

def _cell_losses(cell):
    """Paired test-domain predictions using *training-only* centroid donors."""
    left,right=cell["pair"].split("/")
    train=cell["train"]
    signatures={f:_mean_signature(train[f]) for f in (left,right)}
    pooled=_mean_signature(train[left]+train[right])
    bywav=defaultdict(lambda:defaultdict(list))
    for f in (left,right):
        for u in cell["test"][f]:
            bywav[u.wav_id][f].append(u)
    losses=[]
    for wav,finals in sorted(bywav.items()):
        # Prefer equal weighting of two contexts when they share the WAV;
        # otherwise retain explicit per-WAV average, never count OTO aliases
        # as independent in the benchmark statistic.
        diffs=[]
        for final,events in finals.items():
            for u in events:
                merged=_distance(u.core,pooled)
                split=_distance(u.core,signatures[final])
                diffs.append((merged,split))
        losses.append({"wav_id":wav,"merged":float(np.mean([v[0] for v in diffs])),
                       "split":float(np.mean([v[1] for v in diffs]))})
    # Holdout improvement from keeping both contextual donor identities.
    gain=float(np.mean([r["merged"]-r["split"] for r in losses]))
    # Training gain is allowed for adaptive decisions, never for outcome scores.
    tr=[]
    for f in (left,right):
        for u in train[f]:
            tr.append(_distance(u.core,pooled)-_distance(u.core,signatures[f]))
    train_gain=float(np.mean(tr))
    return losses,train_gain,gain

def evaluate_bank(units,*,budgets=(0,1,2,3,4)):
    partitions=allocate(units)
    valid=[p for p in partitions if p["status"]=="evaluable"]
    invalid=[{k:v for k,v in p.items() if k not in ("train","test")} for p in partitions if p["status"]!="evaluable"]
    # Fair strategy comparisons require all four predeclared final pairs in
    # *the same held-out subbank*. Never assign an artificially easy zero loss.
    per_subbank=defaultdict(dict)
    for c in valid:
        losses,tr_gain,te_gain=_cell_losses(c)
        per_subbank[c["subbank"]][c["pair"]]={
            "train_gain":tr_gain,"test_gain":te_gain,
            "test_WAV_losses":losses}
    curves=[]
    for subbank,cells in sorted(per_subbank.items()):
        if set(cells)!=set(PAIR_NAMES):
            continue
        # All methods use identical target WAVs and fixed budget. The
        # data-adaptive model may rank by local TRAIN-only gain.
        strategies={
            "fixed_lexicographic":sorted(PAIR_NAMES),
            "linguistic_rule":list(RULE_ORDER),
            "speaker_adaptive_train_only":sorted(PAIR_NAMES,key=lambda x:(-cells[x]["train_gain"],x)),
        }
        for budget in budgets:
            for name,rank in strategies.items():
                selected=set(rank[:budget])
                wav_losses=defaultdict(list)
                for pair in PAIR_NAMES:
                    for row in cells[pair]["test_WAV_losses"]:
                        loss=row["split"] if pair in selected else row["merged"]
                        wav_losses[row["wav_id"]].append(loss)
                metric=float(np.mean([np.mean(v) for v in wav_losses.values()]))
                curves.append({"subbank":subbank,"budget":budget,"method":name,
                               "mean_heldout_log_spectral_distance":round(metric,7),
                               "test_WAVs":len(wav_losses),
                               "selected_split_pairs":sorted(selected)})
    return {
        "status":"evaluable" if curves else "insufficient",
        "extracted_candidate_units":len(units),
        "evaluable_pair_cells":len(valid),
        "unevaluable_pair_cells":len(invalid),
        "complete_four_pair_subbanks":len({r["subbank"] for r in curves}),
        "unavailable":invalid,"budget_curves":curves,
        "methods":{
            "fixed_lexicographic":"Predefined alphabetic split order; no train selection",
            "linguistic_rule":"Predefined a/ua, an/uan, ai/uai, ang/uang split order; not fit to data",
            "speaker_adaptive_train_only":"Rank split gains using that pitch-subbank's training WAVs only",
        },
    }

def run_manifest(manifest:Path,output:Path,*,base="sh"):
    data=json.loads(manifest.expanduser().resolve(strict=True).read_text("utf-8"))
    if data.get("schema")!="phonoweave.voicebank_replication_manifest.v1":
        raise ValueError("invalid manifest schema")
    reports=[]
    for entry in data["voicebanks"]:
        units,meta=extract(Path(entry["path"]),base=base)
        item=evaluate_bank(units)
        item.update(bank_id=entry["bank_id"],speaker_group_id=entry["speaker_group_id"],
                    "extraction":meta)
        reports.append(item)
    result={"schema":SCHEMA,"base":base,"banks":reports,
        "budget_units":"extra paired-context donor identity; 0 merged / 4 split",
        "reference":"Held-out WAV spectral signature, not rendered synthesis",
        "unverified_speaker_identity":True,
        "no_audio_written":True,"source_OTO_WAV_modified":False,
        "critical_limits":[
            "A zero eligible cohort is a data-coverage failure; no advantage can be claimed.",
            "Same physical WAV is never shared across train and held-out within a subbank.",
            "Adaptation uses training-only data, but sample size may be insufficient to learn per-speaker priorities.",
            "The fixed rule order is a transparent heuristic, not a universally established linguistic rule.",
            "No synthesis-resampler loss, real-cost allocation or statistically independent speaker holdout.",
            "OTO-marked core can contain vowel/glide material; model gains are acoustic proxies.",
        ]}
    dest=output.expanduser().resolve();dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return result
