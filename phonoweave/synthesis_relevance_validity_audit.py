"""Read-only diagnostic controls for v0.5 OTO fricative separability.

Do not relax extraction thresholds, alter source OTO/WAV, generate sound,
or make a synthesis split/merge decision. All statistical sampling units
are physical WAVs, not OTO aliases.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .audio import AudioReadError, read_wav
from .mandarin import collect_observations, context_for
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .synthesis_relevance_acoustic_separability import (
    Observation, SEED, _features, _nearest_centroid_predict, extract_bank, grouped_cv,
)

SCHEMA = "phonoweave.synthesis_relevance.acoustic_validity_audit.v0.6"
TRIMS_MS = (0.0, 20.0, 40.0)
MIN_REGION_MS = 110.0


def _feature_failure(samples: np.ndarray, rate: int, begin: float, end: float, extra_trim: float) -> str | None:
    if not np.isfinite(begin) or not np.isfinite(end):
        return "nonfinite_oto_timing"
    if rate < 16000:
        return "unsupported_rate_below_16khz"
    if end - extra_trim - begin < MIN_REGION_MS:
        return "oto_interval_shorter_than_110ms_after_trim"
    start = begin + 12.0
    finish = end - extra_trim - 22.0
    centers = np.linspace(start + 14.0, finish - 14.0, 3)
    for c in centers:
        lo = int(round((c - 12.5) * rate / 1000))
        hi = int(round((c + 12.5) * rate / 1000))
        if lo < 0 or hi > len(samples) or hi - lo < 200:
            return "spectral_window_outside_wav"
    return None


def _duration_bin(value: float) -> str:
    if not np.isfinite(value):
        return "nonfinite"
    if value <= 0: return "zero_or_negative"
    if value < 40: return "0_to_40ms"
    if value < 80: return "40_to_80ms"
    if value < 110: return "80_to_110ms"
    if value < 150: return "110_to_150ms"
    if value < 250: return "150_to_250ms"
    return "250ms_plus"


def audit_extraction(root: Path, *, base: str = "sh") -> dict:
    if base not in ("sh", "s", "x"):
        raise ValueError("supported bases: sh, s, x")
    root = root.expanduser().resolve(strict=True)
    entries, warnings = load_voicebank(root)
    observations = collect_observations(
        (entry for entry in entries if entry.wav_path.is_file()),
        affix_pairs(load_prefix_maps(root)),
    )
    wave_cache = {}
    successful_keys = set()
    baseline_rows = {}
    trimmed = {extra: {} for extra in TRIMS_MS if extra}
    reasons = Counter()
    by_context = defaultdict(Counter)
    by_subbank = defaultdict(Counter)
    duration_bins = Counter()
    sample_rates = Counter()
    for obs in observations:
        if obs.base_unit != base:
            continue
        context = context_for(base, obs.final)
        if context not in ("plain", "rounded"):
            continue
        e = obs.entry
        subbank = str(e.oto_path.parent.relative_to(root))
        interval = e.preutterance
        duration_bins[_duration_bin(interval)] += 1
        wav = e.wav_path.resolve()
        identity = (
            str(wav), round(e.offset + e.preutterance, 1), context
        )
        # Deduplicate successful events exactly as v0.5 does.
        if identity in successful_keys:
            reason = "duplicate_successful_oto_event"
            reasons[reason] += 1
            by_context[context][reason] += 1
            by_subbank[subbank][reason] += 1
            continue
        try:
            if wav not in wave_cache:
                wave_cache[wav] = read_wav(wav)
            audio, rate = wave_cache[wav]
            sample_rates[str(rate)] += 1
        except (AudioReadError, OSError, ValueError):
            reason = "wav_read_error"
            reasons[reason] += 1
            by_context[context][reason] += 1
            by_subbank[subbank][reason] += 1
            continue
        reason = _feature_failure(audio, rate, e.offset, e.offset+interval, 0.0)
        features = None
        if reason is None:
            try:
                features = _features(audio,rate,e.offset,e.offset+interval)
            except (ValueError, AudioReadError, OverflowError, FloatingPointError) as exc:
                message = str(exc)
                reason = (
                    "insufficient_spectral_energy" if "energy" in message
                    else "other_feature_error"
                )
        if reason:
            reasons[reason] += 1
            by_context[context][reason] += 1
            by_subbank[subbank][reason] += 1
            continue
        successful_keys.add(identity)
        wav_id=sha256(str(wav.relative_to(root)).encode("utf-8")).hexdigest()[:24]
        row=Observation(wav_id,subbank,context,features)
        baseline_rows[identity]=row
        reasons["accepted"] += 1
        by_context[context]["accepted"] += 1
        by_subbank[subbank]["accepted"] += 1
        for extra in trimmed:
            if _feature_failure(audio,rate,e.offset,e.offset+interval,extra):
                continue
            try:
                new_features=_features(audio,rate,e.offset,e.offset+interval-extra)
            except (ValueError, AudioReadError, OverflowError, FloatingPointError):
                continue
            trimmed[extra][identity]=Observation(wav_id,subbank,context,new_features)

    # An exact v0.5 extraction replay prevents a diagnostic-only code path
    # from silently creating a different baseline subset.
    previous, old_meta = extract_bank(root, base=base)
    if Counter(previous) != Counter(baseline_rows.values()):
        raise AssertionError("v0.5 extraction replay mismatch")
    matching = {}
    for extra, candidates in trimmed.items():
        common=[key for key in baseline_rows if key in candidates]
        before = [baseline_rows[key] for key in common]
        after = [candidates[key] for key in common]
        old_cv=grouped_cv(before)
        new_cv=grouped_cv(after)
        if len(before) != len(after):
            raise AssertionError("matched-window comparison count differs")
        matching[str(int(extra))] = {
            "retained_events":len(common),
            "retained_fraction_of_baseline":round(len(common)/max(len(baseline_rows),1),6),
            "original_on_identical_events":old_cv,
            "trimmed_on_identical_events":new_cv,
            "balanced_accuracy_delta_trimmed_minus_original":(
                round(new_cv["balanced_accuracy"]-old_cv["balanced_accuracy"],6)
                if old_cv["status"] == new_cv["status"] == "evaluated" else None
            ),
        }
    return {
        "candidate_contexts":dict(sorted((k,dict(v)) for k,v in by_context.items())),
        "candidate_subbanks":dict(sorted((k,dict(v)) for k,v in by_subbank.items())),
        "oto_interval_ms_histogram":dict(sorted(duration_bins.items())),
        "sample_rate_event_counts":dict(sorted(sample_rates.items())),
        "extraction_reasons":dict(sorted(reasons.items())),
        "v05_replay": {
            "matches":True, "valid_events":old_meta["events"],
            "independent_WAVs":old_meta["physical_wavs"],
        },
        "extra_right_edge_trimming_ms":matching,
    }


def _folds(rows: list[Observation], n_splits: int=5):
    """Reproduce the v0.5 grouped split exactly, without a feature transform."""
    groups=sorted({r.wav_id for r in rows})
    if len(rows)<10 or len(groups)<n_splits:
        return None
    y=np.asarray([int(r.context=="rounded") for r in rows], dtype=np.int8)
    identifiers=np.asarray([r.wav_id for r in rows])
    strata=defaultdict(list)
    for g in groups:
        labels=y[identifiers==g]
        key=(bool(np.any(labels==1)),bool(np.any(labels==0)))
        strata[key].append(g)
    rng=np.random.default_rng(SEED)
    held=[set() for _ in range(n_splits)]
    for key in sorted(strata):
        bucket=sorted(strata[key])
        rng.shuffle(bucket)
        for i,g in enumerate(bucket):
            held[i%n_splits].add(g)
    masks=[np.isin(identifiers, list(group)) for group in held]
    if any(not m.any() or len(set(y[m]))!=2 or len(set(y[~m]))!=2 for m in masks):
        return None
    return masks


def _score_labels(x:np.ndarray,y:np.ndarray,masks:list[np.ndarray]) -> float | None:
    truth=[]
    predictions=[]
    for test in masks:
        train=~test
        if len(set(y[train]))!=2 or len(set(y[test]))!=2:
            return None
        p=_nearest_centroid_predict(x[train],y[train],x[test])
        truth.extend(y[test])
        predictions.extend(p)
    truth=np.asarray(truth)
    predictions=np.asarray(predictions)
    if len(set(truth))!=2:
        return None
    return float((np.mean(predictions[truth==0]==0)+np.mean(predictions[truth==1]==1))/2)


def wav_group_label_null(rows:list[Observation], *, permutations:int=399) -> dict:
    """Flip entire physical-WAV class labels; refit within every unchanged fold.

    All labels within each physical WAV flip together, preserving the
    original grouping and relative same-WAV structure. Rejections due to
    absent class support are reported, not counted as negative trials.
    """
    if permutations<20:
        raise ValueError("at least 20 permutations required")
    folds=_folds(rows)
    if folds is None:
        return {"status":"insufficient","reason":"v05_grouped_split_not_evaluable"}
    raw=grouped_cv(rows)
    if raw["status"]!="evaluated":
        raise AssertionError("v0.5 CV and audit split disagree")
    x=np.asarray([r.features for r in rows],dtype=float)
    y=np.asarray([int(r.context=="rounded") for r in rows],dtype=np.int8)
    identities=np.asarray([r.wav_id for r in rows])
    unique=sorted(set(identities))
    group_lookup={g:np.flatnonzero(identities==g) for g in unique}
    observed=_score_labels(x,y,folds)
    if observed is None or abs(observed-raw["balanced_accuracy"])>1e-6:
        raise AssertionError("v0.5 CV metric replay mismatch")
    rng=np.random.default_rng(SEED+59)
    accepted=[]
    attempts=0
    maximum_attempts=permutations*20
    while len(accepted)<permutations and attempts<maximum_attempts:
        attempts+=1
        new_y=y.copy()
        flags=rng.integers(0,2,size=len(unique),dtype=np.int8)
        for flag,g in zip(flags,unique):
            if flag:
                ids=group_lookup[g]
                new_y[ids]=1-new_y[ids]
        statistic=_score_labels(x,new_y,folds)
        if statistic is not None:
            accepted.append(statistic)
    if len(accepted)<permutations:
        return {
            "status":"insufficient","reason":"permutation_class_support",
            "valid_null_trials":len(accepted),
            "requested_null_trials":permutations,"attempts":attempts,
        }
    values=np.asarray(accepted)
    return {
        "status":"evaluated",
        "observed_balanced_accuracy":round(observed,6),
        "whole_WAV_label_flip_null_mean":round(float(np.mean(values)),6),
        "null_p05":round(float(np.percentile(values,5)),6),
        "null_p95":round(float(np.percentile(values,95)),6),
        "permutation_p_upper_tail":round(
            (1+int(np.sum(values>=observed-1e-12)))/(len(values)+1),6
        ),
        "valid_null_trials":len(values),"attempts":attempts,
        "physical_WAV_groups":len(unique),
        "method":"joint_WAV_context_label_flip; train-only standardization and classifier refit each fold",
        "warning":"Fixed original folds and conditional trial acceptances; descriptive null, not a calibrated cross-speaker p-value.",
    }


def leave_one_subbank_out(rows:list[Observation]) -> dict:
    subbanks=sorted({r.subbank for r in rows})
    if len(subbanks)<2:
        return {"status":"insufficient","reason":"only_one_subbank"}
    x=np.asarray([r.features for r in rows],dtype=float)
    y=np.asarray([int(r.context=="rounded") for r in rows])
    group=np.asarray([r.subbank for r in rows])
    reports=[]
    for key in subbanks:
        test=group==key
        train=~test
        if not test.any() or len(set(y[train]))!=2 or len(set(y[test]))!=2:
            reports.append({"subbank":key,"status":"insufficient","reason":"context_missing"})
            continue
        prediction=_nearest_centroid_predict(x[train],y[train],x[test])
        score=(np.mean(prediction[y[test]==0]==0)
               +np.mean(prediction[y[test]==1]==1))/2
        reports.append({"subbank":key,"status":"evaluated",
                        "test_events":int(test.sum()),
                        "balanced_accuracy":round(float(score),6)})
    evaluated=[r["balanced_accuracy"] for r in reports if r["status"]=="evaluated"]
    return {"status":"evaluated" if evaluated else "insufficient",
            "subbanks_evaluable":len(evaluated),
            "subbanks_total":len(subbanks),
            "mean_heldout_subbank_accuracy":round(float(np.mean(evaluated)),6) if evaluated else None,
            "subbanks":reports}


def evaluate_bank(root: Path, *, base:str="sh", permutations:int=399) -> dict:
    rows,metadata=extract_bank(root,base=base)
    audit=audit_extraction(root,base=base)
    return {
        "baseline_events":metadata["events"],
        "baseline_WAVs":metadata["physical_wavs"],
        "baseline_grouped_cv":grouped_cv(rows),
        "extraction_audit":audit,
        "grouped_label_permutation":wav_group_label_null(rows,permutations=permutations),
        "leave_one_subbank_out":leave_one_subbank_out(rows),
    }


def run_manifest(manifest_path:Path,output_path:Path,*,base:str="sh",permutations:int=399) -> dict:
    manifest=json.loads(manifest_path.expanduser().resolve(strict=True).read_text("utf-8"))
    if manifest.get("schema")!="phonoweave.voicebank_replication_manifest.v1":
        raise ValueError("unexpected replication manifest schema")
    ids=set()
    speaker_ids=set()
    reports=[]
    for entry in manifest["voicebanks"]:
        bank_id=entry["bank_id"]
        speaker_id=entry["speaker_group_id"]
        if bank_id in ids or speaker_id in speaker_ids:
            raise ValueError("duplicate bank or independent speaker identifier")
        ids.add(bank_id);speaker_ids.add(speaker_id)
        result=evaluate_bank(Path(entry["path"]),base=base,permutations=permutations)
        result["bank_id"]=bank_id
        result["speaker_group_id"]=speaker_id
        reports.append(result)
    result={
        "schema":SCHEMA,"base_unit":base,
        "permutations_per_evaluable_bank":permutations,
        "banks":reports,"evaluated_banks":len(reports),
        "no_audio_written":True,
        "speaker_profile_or_inventory_modified":False,
        "interpretation":"Objective OTO-aligned acoustic association only; not UTAU unit necessity.",
    }
    output=output_path.expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return result
