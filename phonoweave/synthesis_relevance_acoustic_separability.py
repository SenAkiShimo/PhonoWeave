"""Read-only sh plain/rounded frication separability with grouped validation.

OTO offset/preutterance supplies an *approximate* consonant interval, not
ground-truth CV boundaries. Frames near either end are deliberately excluded.
This study tests acoustic separability, NOT the need for distinct UTAU aliases.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .audio import AudioReadError, read_wav
from .mandarin import collect_observations, context_for
from .oto import load_voicebank
from .prefixmap import load_prefix_maps, affix_pairs

SCHEMA = "phonoweave.synthesis_relevance.acoustic_separability.v0.5"
FEATURE_NAMES = (
    "cog_early_khz", "highband_fraction_early", "spectral_flatness_early",
    "cog_middle_khz", "highband_fraction_middle", "spectral_flatness_middle",
    "cog_late_khz", "highband_fraction_late", "spectral_flatness_late",
)
SEED = 20261011


@dataclass(frozen=True)
class Observation:
    wav_id: str
    subbank: str
    context: str
    features: tuple[float, ...]


def _features(samples: np.ndarray, rate: int, start_ms: float, end_ms: float) -> tuple[float, ...]:
    """Three interior 25-ms spectra within the OTO-defined candidate region."""
    if not (np.isfinite(start_ms) and np.isfinite(end_ms)):
        raise ValueError("nonfinite OTO region")
    if rate < 16000 or end_ms - start_ms < 110:
        raise ValueError("short OTO region")
    # Avoid the initial onset and the presumed CV junction.
    start_ms += 12
    end_ms -= 22
    centers = np.linspace(start_ms + 14, end_ms - 14, 3)
    outcomes = []
    for center in centers:
        lo = int(round((center - 12.5) * rate / 1000))
        hi = int(round((center + 12.5) * rate / 1000))
        if lo < 0 or hi > len(samples) or hi - lo < 200:
            raise ValueError("truncated acoustic frame")
        x = samples[lo:hi].astype(float)
        x = (x - x.mean()) * np.hanning(len(x))
        spec = np.abs(np.fft.rfft(x)) ** 2
        freq = np.fft.rfftfreq(len(x), 1 / rate)
        band = (freq >= 1500) & (freq <= min(11000, 0.46 * rate))
        high = (freq >= 5000) & band
        power = spec[band]
        if power.size < 4 or power.sum() <= 1e-12:
            raise ValueError("insufficient high-frequency energy")
        cog = float(np.dot(freq[band], power) / power.sum() / 1000)
        high_fraction = float(spec[high].sum() / power.sum())
        flatness = float(np.exp(np.mean(np.log(power + 1e-15))) / (np.mean(power) + 1e-15))
        outcomes.append((cog,high_fraction,flatness))
    # Feature order grouped by channel, not frame.
    values = tuple(float(outcomes[t][k]) for k in range(3) for t in range(3))
    if not all(np.isfinite(values)):
        raise ValueError("nonfinite acoustic features")
    return values


def extract_bank(root: Path, *, base: str = "sh") -> tuple[list[Observation], dict]:
    if base not in {"sh","s","x"}:
        raise ValueError("supported sibilants: sh, s, x")
    root = root.expanduser().resolve(strict=True)
    entries, warnings = load_voicebank(root)
    observations = collect_observations(
        (e for e in entries if e.wav_path.is_file()),
        affix_pairs(load_prefix_maps(root)),
    )
    # Deduplicate aliases resolving to the same physical WAV, location and context.
    events = {}
    cache = {}
    skipped = Counter()
    for obs in observations:
        if obs.base_unit != base:
            continue
        context = context_for(base, obs.final)
        if context not in {"plain","rounded"}:
            continue
        e = obs.entry
        wav = e.wav_path.resolve()
        subbank = str(e.oto_path.parent.relative_to(root))
        point = round(e.offset + e.preutterance, 1)
        event = (str(wav),point,context)
        if event in events:
            skipped["duplicate_alias_event"] += 1
            continue
        try:
            if wav not in cache:
                cache[wav] = read_wav(wav)
            x, rate = cache[wav]
            features = _features(x,rate,e.offset,e.offset + e.preutterance)
        except (AudioReadError,ValueError,OSError,OverflowError):
            skipped["invalid_OTO_candidate_or_audio"] += 1
            continue
        key = sha256(str(wav.relative_to(root)).encode("utf-8")).hexdigest()[:24]
        events[event] = Observation(key,subbank,context,features)
    return list(events.values()), {
        "oto_entries":len(entries), "oto_warnings":len(warnings),
        "events":len(events), "physical_wavs":len({o.wav_id for o in events.values()}),
        "skipped_reasons":dict(skipped)
    }


def _nearest_centroid_predict(train_x,train_y,test_x):
    """Standardization fitted exclusively to training WAV groups."""
    mu=train_x.mean(axis=0)
    sd=train_x.std(axis=0)
    sd=np.where(sd<1e-6,1.,sd)
    a=(train_x-mu)/sd
    b=(test_x-mu)/sd
    centroids=np.stack([a[train_y==i].mean(axis=0) for i in (0,1)])
    distances=np.sum((b[:,None,:]-centroids[None,:,:])**2,axis=2)
    return np.argmin(distances,axis=1)


def grouped_cv(rows: list[Observation], *, n_splits:int=5) -> dict:
    """Group by physical WAV. Report pooled class recalls, not naive row accuracy."""
    if len(rows)<10:
        return {"status":"insufficient","reason":"fewer_than_10_events"}
    groups=sorted({r.wav_id for r in rows})
    if len(groups)<n_splits:
        return {"status":"insufficient","reason":"too_few_independent_WAVs"}
    x=np.asarray([r.features for r in rows],dtype=float)
    y=np.asarray([int(r.context=="rounded") for r in rows])
    group=np.asarray([r.wav_id for r in rows])
    # Stratify *groups* approximately by their context composition.
    counts={g: Counter(y[group==g]) for g in groups}
    strata=defaultdict(list)
    for g in groups:
        key=(counts[g][1]>0,counts[g][0]>0)
        strata[key].append(g)
    rng=np.random.default_rng(SEED)
    fold_groups=[set() for _ in range(n_splits)]
    for key in sorted(strata):
        bucket=sorted(strata[key])
        rng.shuffle(bucket)
        for i,g in enumerate(bucket):
            fold_groups[i%n_splits].add(g)
    true=[];pred=[]; folds=[]
    for fold,held in enumerate(fold_groups):
        mask=np.isin(group,list(held))
        training=~mask
        if (not mask.any() or len(set(y[training]))!=2 or len(set(y[mask]))!=2):
            return {"status":"insufficient","reason":"grouped_fold_missing_a_context"}
        out=_nearest_centroid_predict(x[training],y[training],x[mask])
        true.extend(y[mask].tolist())
        pred.extend(out.tolist())
        folds.append({"fold":fold,"test_wavs":len(held),"test_events":int(mask.sum())})
    true=np.asarray(true);pred=np.asarray(pred)
    recall_plain=float(np.mean(pred[true==0]==0))
    recall_rounded=float(np.mean(pred[true==1]==1))
    return {"status":"evaluated","balanced_accuracy":round((recall_plain+recall_rounded)/2,6),
            "recall_plain":round(recall_plain,6),
            "recall_rounded":round(recall_rounded,6),"folds":folds,
            "n_events":len(rows),"n_wavs":len(groups)}


def paired_wav_control(rows: list[Observation], *, permutations:int=2000) -> dict:
    """Same-WAV within-subbank paired difference and WAV-level sign-flip null."""
    grouped=defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[(r.subbank,r.wav_id)][r.context].append(r.features)
    pairs=[]
    for key,states in sorted(grouped.items()):
        if not states["plain"] or not states["rounded"]:
            continue
        # One vector per context per WAV: not treating OTO aliases as independent.
        pairs.append(np.mean(states["rounded"],axis=0)-np.mean(states["plain"],axis=0))
    if len(pairs)<6:
        return {"status":"insufficient","paired_wavs":len(pairs),
                "reason":"fewer_than_six_within_WAV_pairs"}
    arr=np.asarray(pairs)
    # Per-feature standardized paired mean magnitude, descriptive.
    scale=np.std(arr,axis=0,ddof=1)
    scale=np.maximum(scale,1e-6)
    observed=float(np.linalg.norm(np.mean(arr,axis=0)/scale))
    rng=np.random.default_rng(SEED)
    exceeding=0
    for _ in range(permutations):
        signs=rng.choice([-1.,1.],size=(len(arr),1))
        statistic=float(np.linalg.norm(np.mean(arr*signs,axis=0)/scale))
        exceeding+=statistic>=observed-1e-12
    return {
        "status":"evaluated","paired_wavs":len(pairs),
        "standardized_multivariate_mean_norm":round(observed,6),
        "paired_sign_flip_p":round((exceeding+1)/(permutations+1),6),
        "permutations":permutations,
        "mean_rounded_minus_plain_by_feature":{
            name:round(float(v),6) for name,v in zip(FEATURE_NAMES,np.mean(arr,axis=0))
        },
        "method":"within_physical_WAV_pair, fixed paired scale, joint WAV-level sign-flip",
    }


def evaluate_bank(root: Path, *, base:str="sh") -> dict:
    rows,meta=extract_bank(root,base=base)
    overall=grouped_cv(rows)
    subbanks={}
    for subbank in sorted({row.subbank for row in rows}):
        own=[r for r in rows if r.subbank==subbank]
        subbanks[subbank]={
            "observations":len(own),
            "grouped_cv":grouped_cv(own),
            "paired":paired_wav_control(own),
        }
    result={
        "schema":SCHEMA,"base_unit":base,
        "extraction":meta,"grouped_cv_all_subbanks":overall,
        "paired_same_WAV_all_subbanks":paired_wav_control(rows),
        "subbanks":subbanks,
        "interpretation":"acoustic_separability_only_not_unit_merge_or_synthesis_quality",
        "source_wavs_modified":False,
        "no_audio_output":True,
        "limitations":[
            "OTO-defined interval is an approximate candidate, not verified frication-only segmentation.",
            "Grouped classifier may exploit phonetic-final differences and subbank distribution; paired same-WAV analysis mitigates but does not remove these confounds.",
            "No held-out new speakers: each bank is modeled and tested internally.",
            "Events within a physical WAV are correlated; naive per-event significance is invalid.",
            "A low permutation p-value detects spectral association, not perceptual or synthesis-necessity evidence.",
            "No split/merge of synthesis units is automatically supported by this study.",
        ],
    }
    return result


def evaluate_manifest(manifest_path:Path,output_path:Path,*,base:str="sh")->dict:
    manifest=json.loads(manifest_path.expanduser().resolve().read_text("utf-8"))
    if manifest.get("schema")!="phonoweave.voicebank_replication_manifest.v1":
        raise ValueError("unsupported voicebank manifest")
    banks=[]
    for bank in manifest["voicebanks"]:
        row=evaluate_bank(Path(bank["path"]),base=base)
        row["bank_id"]=bank["bank_id"]
        row["speaker_group_id"]=bank["speaker_group_id"]
        banks.append(row)
    report={"schema":SCHEMA,"base_unit":base,"banks":banks,
            "evaluated_banks":len(banks),
            "speaker_group_count":len({b["speaker_group_id"] for b in banks}),
            "aggregate_note":"Report per-speaker effects; do not pool dependent OTO events as independent speakers."}
    output_path=output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n","utf-8")
    return report
