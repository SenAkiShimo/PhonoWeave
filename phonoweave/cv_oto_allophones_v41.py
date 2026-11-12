"""CV-OTO guided allophone candidates for one voicebank.

OTO timing supplies approximate C and V locations from existing CV rows.
Existing VC rows/aliases NEVER define phoneme variants or split ground truth.
A selected language adapter interprets full CV spellings only; other languages
supply their author CV lexicon. No automatic inventory edit is possible here.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from .audio import AudioReadError, read_wav
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .source_aware_recording_v17 import pcm_identity

SCHEMA = "phonoweave.cv_oto_allophone.v4.1"
LEXICON_SCHEMA = "phonoweave.author_cv_lexicon.v1"
NOTE_PREFIX = re.compile(r"^[A-Ga-g](?:#|b)?-?\d+[- ]*")
FREQ_EDGES = np.geomspace(180.0, 7800.0, 13)


def _label(value: Any, name: str) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or len(value) > 128 or any(ord(c) < 32 for c in value)):
        raise ValueError(f"invalid author {name}")
    return value


def read_lexicon(path: Path) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """User describes CV -> C,V, NOT old CVVC/VC unit roles."""
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if (not isinstance(config, dict) or config.get("schema") != LEXICON_SCHEMA
            or set(config) != {"schema", "cv", "alias_prefixes"}):
        raise ValueError("invalid CV lexicon schema")
    rows = config["cv"]
    prefixes = config["alias_prefixes"]
    if not isinstance(rows, list) or not rows or not isinstance(prefixes, list):
        raise ValueError("CV lexicon must contain nonempty cv and prefix list")
    data: dict[str, tuple[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"cv", "consonant", "vowel"}:
            raise ValueError("CV lexicon entries require exactly cv,consonant,vowel")
        cv = _label(row["cv"], "CV")
        if cv in data:
            raise ValueError("duplicate CV lexicon entry")
        data[cv] = (_label(row["consonant"], "consonant"),
                    _label(row["vowel"], "vowel"))
    if any(not isinstance(p, str) or any(ord(ch)<32 for ch in p) for p in prefixes):
        raise ValueError("invalid OTO alias prefix")
    return data, prefixes


def _cv_from_entry(alias: str, *, lexicon: dict[str, tuple[str, str]] | None,
                   prefixes: list[str], adapter: str) -> tuple[str, str, str] | None:
    """Only one full syllable / CV token; no multi-token or VC interpretation."""
    options = [alias.strip()]
    for prefix in sorted(set(prefixes), key=len, reverse=True):
        if prefix and alias.startswith(prefix):
            options.append(alias[len(prefix):].strip())
    # Prefix notation from bank map is metadata, not a split label.
    if adapter == "mandarin":
        options.extend(NOTE_PREFIX.sub("", s).strip() for s in tuple(options))
    matched = []
    for cv in set(options):
        if lexicon is not None:
            if cv in lexicon:
                c,v = lexicon[cv]
                matched.append((cv,c,v))
        elif adapter == "mandarin" and re.fullmatch(r"[a-züv]+", cv):
            # This is an optional, language-specific *orthographic* adapter.
            # It provides C/V identifiers, never an expected allophone count.
            from .mandarin import _split_syllable
            parsed = _split_syllable(cv)
            if parsed is not None:
                matched.append((cv,parsed[0],parsed[1]))
    distinct = list(dict.fromkeys(matched))
    if len(distinct) > 1:
        raise ValueError("ambiguous CV alias/prefix mapping")
    return distinct[0] if distinct else None


def _region_profile(audio: np.ndarray, rate: int, begin_ms: float,
                    end_ms: float) -> list[float]:
    """Three 20ms within-region spectral snapshots, without edge frames."""
    if not (math.isfinite(begin_ms) and math.isfinite(end_ms)):
        raise ValueError("non-finite OTO interval")
    frame = round(rate*.020)
    if rate < 18000 or end_ms-begin_ms < 55 or frame < 64:
        raise ValueError("C/V region too short for stable spectral sampling")
    freqs = np.fft.rfftfreq(frame, 1/rate)
    profiles = []
    for portion in (0.2,0.5,0.8):
        center_ms = begin_ms + (end_ms-begin_ms)*portion
        index = int(round(center_ms*rate/1000)) - frame//2
        if index < 0 or index+frame > len(audio):
            raise ValueError("C/V feature window outside source WAV")
        signal = audio[index:index+frame].astype(np.float64, copy=False)
        if np.sqrt(np.mean(signal**2)) < .001:
            raise ValueError("silent C/V region")
        power = np.abs(np.fft.rfft((signal-signal.mean())*np.hanning(frame)))**2
        bands = []
        for lo, hi in zip(FREQ_EDGES[:-1],FREQ_EDGES[1:]):
            mask = (freqs>=lo)&(freqs<hi)
            if not np.any(mask):
                raise ValueError("insufficient FFT bins")
            bands.append(math.log10(float(np.mean(power[mask]))+1e-10))
        vals = np.array(bands)
        vals -= vals.mean()
        profiles.extend(vals.tolist())
    return profiles


def _vowel_limit(entry, wav_duration_ms: float) -> float:
    # Standard UTAU OTO: negative cutoff is duration from offset;
    # positive cutoff is time trimmed from source WAV end.
    if entry.cutoff < 0:
        return entry.offset - entry.cutoff
    return wav_duration_ms-entry.cutoff


def extract_cv(root: Path, *, adapter: str = "mandarin",
               lexicon_path: Path | None = None) -> tuple[list[dict], dict]:
    """Use CV OTO timing as localization, ignore existing VC split inventory."""
    root=Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("voicebank root must be directory")
    if adapter not in ("mandarin", "author"):
        raise ValueError("adapter must be mandarin or author")
    lexicon = None
    prefixes: list[str] = []
    if adapter == "author":
        if lexicon_path is None:
            raise ValueError("author mode needs --cv-lexicon")
        lexicon,prefixes = read_lexicon(lexicon_path)
    elif lexicon_path is not None:
        raise ValueError("Mandarin adapter does not accept an author lexicon")
    # Already supplied bank affix configuration may help identify full CV,
    # but no author VC split label is a target or supervised category.
    prefixes.extend([p for p,s in affix_pairs(load_prefix_maps(root)) if p and not s])
    entries, warnings=load_voicebank(root)

    @lru_cache(maxsize=3)
    def pcm(path: Path) -> tuple[np.ndarray,int]:
        return read_wav(path)

    hash_cache={}
    rows=[]
    counts=Counter()
    for entry in entries:
        try:
            match=_cv_from_entry(entry.alias,lexicon=lexicon,prefixes=prefixes,
                                 adapter=adapter)
        except ValueError:
            counts["ambiguous_cv"]+=1
            continue
        if match is None:
            counts["not_a_declared_full_CV"]+=1
            continue
        cv,c,v=match
        wav=entry.wav_path.resolve()
        folder=entry.oto_path.parent.resolve()
        if not wav.is_relative_to(root) or not folder.is_relative_to(root):
            counts["wav_outside_voicebank"]+=1
            continue
        if not wav.is_file():
            counts["missing_CV_wav"]+=1
            continue
        try:
            audio,rate=pcm(wav)
            if wav not in hash_cache:
                hash_cache[wav]=pcm_identity(wav)
            t0=float(entry.offset)
            prior=float(entry.offset+entry.preutterance)
            # OTO offset and preutterance are usable approximate CV anchors;
            # no appeal to existing VC units or old split alias identities.
            if t0<0 or prior-t0<72:
                raise ValueError("invalid consonant time span")
            c_start=t0+max(5, (prior-t0)*.13)
            c_end=prior-max(5,(prior-t0)*.13)
            limit=min(_vowel_limit(entry,len(audio)*1000/rate),
                      prior+155)
            v_start=prior+15
            v_end=limit-10
            c_feature=_region_profile(audio,rate,c_start,c_end)
            try:
                v_feature=_region_profile(audio,rate,v_start,v_end)
            except ValueError:
                v_feature=None
                counts["vowel_region_unusable"]+=1
        except (AudioReadError,ValueError,OSError,EOFError):
            counts["consonant_region_unusable"]+=1
            continue
        rows.append({
            "cv":cv, "C":c, "V":v,
            "bank":folder.relative_to(root).as_posix() or ".",
            "wav":wav.relative_to(root).as_posix(),
            "source_template":wav.name.casefold(),
            "pcm_sha256":hash_cache[wav],
            "oto_alias_for_provenance":entry.alias,
            "oto_line":entry.line_number,
            "C_interval_ms":[round(c_start,3),round(c_end,3)],
            "V_interval_ms":[round(v_start,3),round(v_end,3)] if v_feature is not None else None,
            "C_feature":c_feature,
            "V_feature":v_feature,
        })
    return rows,{
        "original_oto_rows":len(entries),
        "selected_full_CV_oto_rows":len(rows),
        "oto_parse_warnings":len(warnings),
        "selection_source":"author_CV_lexicon" if adapter=="author" else "opt_in_Mandarin_CV_orthography",
        "excluded":dict(sorted(counts.items())),
        "no_VC_inventory_in_split_model":True,
    }


def _aggregate(rows: list[dict], role: str) -> dict[str,list[dict]]:
    """One vote per (source bank, source PCM, fixed phoneme, context)."""
    col="C" if role=="C" else "V"
    other="V" if role=="C" else "C"
    key=role+"_feature"
    grouped=defaultdict(list)
    for row in rows:
        if row.get(key) is None:
            continue
        x=np.asarray(row[key],dtype=float)
        if x.shape!=(36,) or not np.isfinite(x).all():
            raise ValueError("invalid acoustic vector")
        group=(row[col],row[other],row["bank"],row["pcm_sha256"])
        grouped[group].append(row)
    per_fixed=defaultdict(list)
    for (fixed,ctx,bank,pcm),part in grouped.items():
        per_fixed[fixed].append({
            "context":ctx,"bank":bank,"pcm":pcm,
            "source_family":part[0]["source_template"],
            "vector":np.mean([np.asarray(r[key],dtype=float) for r in part],axis=0),
            "aliases_collapsed":len(part),
            "source_examples":[{"wav":r["wav"],"CV":r["cv"],
                "alias":r["oto_alias_for_provenance"],
                "oto_line":r["oto_line"]} for r in part[:2]]
        })
    return per_fixed


def _context_partition(centers:dict[str,np.ndarray],k:int)->dict[str,int]:
    """Deterministic agglomeration; number of context groups NOT pre-assumed."""
    groups=[(key,) for key in sorted(centers)]
    if k<1 or k>len(groups):
        raise ValueError("invalid number of groups")
    while len(groups)>k:
        options=[]
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                x=np.mean([centers[s] for s in groups[i]],axis=0)
                y=np.mean([centers[s] for s in groups[j]],axis=0)
                options.append((float(np.mean((x-y)**2)),groups[i],groups[j],i,j))
        _,_,_,i,j=min(options)
        merged=tuple(sorted(groups[i]+groups[j]))
        groups=[group for q,group in enumerate(groups) if q not in (i,j)]
        groups.append(merged)
        groups.sort()
    return {context:num for num,g in enumerate(groups) for context in g}


def _fold_scores(items:list[dict],max_k:int=3)->dict:
    """Train context-grouping only on other full physical source templates.

    Test audio never chooses its own subtype. Source family held out across
    all banks. Only bank-matched prediction is scored.
    """
    families=sorted({o["source_family"] for o in items})
    context_set=sorted({o["context"] for o in items})
    results={k:[] for k in range(1,max_k+1)}
    for family in families:
        train=[o for o in items if o["source_family"]!=family]
        test=[o for o in items if o["source_family"]==family]
        centroids={}
        for context in context_set:
            matches=[o["vector"] for o in train if o["context"]==context]
            if matches:
                centroids[context]=np.mean(matches,axis=0)
        if len(centroids)!=len(context_set):
            continue
        for k in results:
            partition=_context_partition(centroids,k)
            bank_vectors=defaultdict(list)
            for o in train:
                bank_vectors[(o["bank"],partition[o["context"]])].append(o["vector"])
            targets=defaultdict(list)
            for o in test:
                candidates=bank_vectors.get((o["bank"],partition[o["context"]]),[])
                if not candidates:
                    continue
                prediction=np.mean(candidates,axis=0)
                targets[o["bank"]].append(float(np.mean((o["vector"]-prediction)**2)))
            if targets:
                # One vote per physical source template fold, not per OTO
                # or pitch subbank; bank errors averaged within fold.
                results[k].append((family,float(np.mean([
                    float(np.mean(v)) for v in targets.values()]))))
    # Compare on the INTERSECTION of the same eligible source templates.
    common=set(f for f,_ in results[1])
    for k in results:
        common &= {f for f,_ in results[k]}
    if len(common)<3:
        return {"holdout_source_templates":len(common),
                "k_results":[],"status":"insufficient_source_template_replication"}
    models=[]
    for k in results:
        mse=float(np.mean([score for f,score in results[k] if f in common]))
        models.append({"k":k,"heldout_mse":round(mse,8),
                       "same_eligible_source_templates":len(common)})
    return {"holdout_source_templates":len(common),
            "k_results":models,
            "status":"exploratory_heldout_context_prediction_only"}


def analyze_candidates(rows: list[dict]) -> dict:
    """Compare K=1/2/3 same-C and same-V realization groupings.

    Never use existing VC alias names or old unit counts for any analysis.
    """
    results=[]
    for role in ("C","V"):
        families=_aggregate(rows,role)
        for symbol in sorted(families):
            obs=families[symbol]
            contexts=sorted({o["context"] for o in obs})
            source_families=sorted({o["source_family"] for o in obs})
            if len(contexts)<2:
                continue
            by_context=defaultdict(list)
            for o in obs:
                by_context[o["context"]].append(o)
            means={c:np.mean([o["vector"] for o in by_context[c]],axis=0)
                   for c in contexts}
            trials=_fold_scores(obs,min(3,len(contexts)))
            # Context groups are descriptive FULL-DATA fit; heldout folds
            # independently re-fit them without seeing each target family.
            full_partitions=[]
            for k in range(1,min(3,len(contexts))+1):
                part=_context_partition(means,k)
                grouped=defaultdict(list)
                for c,g in part.items():
                    grouped[g].append(c)
                full_partitions.append({
                    "k":k,
                    "context_groups":[sorted(g) for _,g in sorted(grouped.items())],
                    "trained_on_full_data_for_review_only":True
                })
            evaluated={x["k"]:x["heldout_mse"] for x in trials["k_results"]}
            proxy_delta={str(k):round(evaluated[1]-evaluated[k],8)
                         for k in evaluated if k>1 and 1 in evaluated}
            results.append({
                "phoneme_role":role,"author_phoneme":symbol,
                "context_axis":"following_V" if role=="C" else "preceding_C",
                "contexts":contexts,
                "original_cv_rows_collapsed":sum(x["aliases_collapsed"] for x in obs),
                "source_group_observations":len(obs),
                "distinct_source_templates":len(source_families),
                "context_source_templates":{
                    c:len({o["source_family"] for o in by_context[c]})
                    for c in contexts},
                "candidate_groupings":full_partitions,
                "source_template_holdout":trials,
                "k1_minus_k_more_mse":proxy_delta,
                "allophone_split_status":"CANDIDATE_ONLY_NOT_VALIDATED",
                "new_units_authorized":0,
                "examples":[{
                    "context":c,
                    "source":by_context[c][0]["source_examples"][0]
                } for c in contexts[:6]],
            })
    results.sort(key=lambda x:(x["phoneme_role"],x["author_phoneme"]))
    return {
        "schema":SCHEMA,
        "method":"OTO_locates_C_V_from_CV_only__VC_inventory_excluded_from_inference",
        "candidate_phonemes":len(results),
        "results":results,
        "new_aliases":[],
        "recording_list_changes":[],
        "VC_marginal_value":"not_yet_evaluated_requires_counterfactual_synthesis",
        "limitations":[
            "OTO offset/preutterance are approximate author timing, not all gold phone boundaries.",
            "Consonant region and vowel onset samples can contain normal coarticulation.",
            "Stored CV pronunciation is from an opt-in language adapter or an author CV lexicon, NOT existing VC splits.",
            "Pitch/style banks from one singer are not independent human performances.",
            "Source filename-template holdout is conservative but not proof of independent sessions.",
            "Full-data K-grouping is a review hypothesis; source-heldout scores are exploratory, not synthesis quality.",
            "No existing VC can be removed or new allophone added without controlled singing rendering and recording cost testing.",
        ]
    }


def export_report(root:Path,out:Path,report:dict,meta:dict)->Path:
    root=Path(root).expanduser().resolve(strict=True)
    out=Path(out).expanduser().resolve()
    if out==root or out.is_relative_to(root):
        raise ValueError("output must be outside source voicebank")
    if out.exists():
        raise FileExistsError("refusing to overwrite existing output")
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pw-allophone-",dir=out.parent) as tmp:
        stage=Path(tmp)/"export"
        stage.mkdir()
        (stage/"allophone_candidates.json").write_text(
            json.dumps({**report,"scan":meta},ensure_ascii=False,indent=2,
                       allow_nan=False)+"\n",encoding="utf-8")
        with (stage/"allophone_review.csv").open("x",newline="",encoding="utf-8") as f:
            writer=csv.writer(f)
            writer.writerow(("role","phoneme","contexts","source_templates",
                             "source_holdout_status","K1_minus_K2_MSE",
                             "K1_minus_K3_MSE","split_status"))
            for r in report["results"]:
                fields=(r["phoneme_role"],r["author_phoneme"],
                        " | ".join(r["contexts"]),r["distinct_source_templates"],
                        r["source_template_holdout"]["status"],
                        r["k1_minus_k_more_mse"].get("2"),
                        r["k1_minus_k_more_mse"].get("3"),
                        r["allophone_split_status"])
                writer.writerow([
                    "'"+v if isinstance(v,str) and v.lstrip().startswith(("=","+","-","@")) else v
                    for v in fields
                ])
        os.rename(stage,out)
    return out


def main(argv:list[str]|None=None)->int:
    ap=argparse.ArgumentParser(description="CV/OTO-timed allophone hypotheses; VC split scheme is never used")
    ap.add_argument("voicebank",type=Path)
    ap.add_argument("--adapter",choices=("mandarin","author"),default="mandarin")
    ap.add_argument("--cv-lexicon",type=Path)
    ap.add_argument("--output",required=True,type=Path)
    args=ap.parse_args(argv)
    try:
        observations,meta=extract_cv(args.voicebank,adapter=args.adapter,
                                     lexicon_path=args.cv_lexicon)
        if not observations:
            raise ValueError("no usable CV rows; verify language adapter or author CV lexicon")
        result=analyze_candidates(observations)
        export_report(args.voicebank,args.output,result,meta)
    except (ValueError,OSError,TypeError,KeyError,json.JSONDecodeError,EOFError) as e:
        ap.error(str(e))
    print(json.dumps({
        "CV_OTO_rows":meta["selected_full_CV_oto_rows"],
        "candidate_phonemes":result["candidate_phonemes"],
        "VC_split_scheme_used":False,
        "approved_new_units":0,
        "output":str(args.output),
    },ensure_ascii=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
