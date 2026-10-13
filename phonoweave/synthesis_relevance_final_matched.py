"""Final-paired donor counterfactual with explicit identifiability audit.

No sound is generated. A matched control donor must have exactly the same
phonetic final as its target, while a cross donor must have the predefined
rounded/plain counterpart. Both must be from another physical WAV within
the same pitch subbank. Recording-list confounds and OTO alignment remain.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from .synthesis_relevance_unit_merging_cost import (
    Unit,extract,_distance,paired_stats
)

SCHEMA="phonoweave.synthesis_relevance.final_matched_counterfactual.v0.9"
# Only transparent orthographic base/rounded finals, excluding pairs with
# potentially different monophthong/diphthong phonological nuclei (e.g. en/un).
FINAL_PAIRS=(("a","ua"),("ai","uai"),("an","uan"),("ang","uang"))
PAIR_OF={f:(i,side) for i,p in enumerate(FINAL_PAIRS) for side,f in enumerate(p)}
CALIPER_MS=10.0
BALANCE_MS=5.0

def _score(d:Unit,t:Unit):
    return (
        _distance(d.core,t.core),
        _distance(d.core,t.late),
        abs(d.core_rms_db-t.late_rms_db),
    )

def _closest(pool:list[Unit],target:Unit):
    valid=[u for u in pool if u.wav_id!=target.wav_id
           and abs(u.duration_ms-target.duration_ms)<=CALIPER_MS]
    return min(valid,key=lambda u:(abs(u.duration_ms-target.duration_ms),
                                    u.wav_id,u.duration_ms)) if valid else None

def matched_trials(units:list[Unit]):
    buckets=defaultdict(list)
    for u in units:
        if u.final in PAIR_OF:
            buckets[(u.group,PAIR_OF[u.final][0],u.final)].append(u)
    reasons=Counter()
    trials=[]
    for target in units:
        if target.final not in PAIR_OF:
            reasons["final_without_prespecified_pair"]+=1
            continue
        pair_index,side=PAIR_OF[target.final]
        opposite=FINAL_PAIRS[pair_index][1-side]
        same=_closest(buckets[(target.group,pair_index,target.final)],target)
        cross=_closest(buckets[(target.group,pair_index,opposite)],target)
        if same is None:
            reasons["no_different_WAV_same_final_within_10ms"]+=1
        if cross is None:
            reasons["no_different_WAV_matched_opposite_final_within_10ms"]+=1
        if same is None or cross is None:
            continue
        ds=abs(same.duration_ms-target.duration_ms)
        dc=abs(cross.duration_ms-target.duration_ms)
        if abs(ds-dc)>BALANCE_MS:
            reasons["arm_duration_balance_exceeds_5ms"]+=1
            continue
        a=_score(same,target)
        b=_score(cross,target)
        trials.append({
            "wav_id":target.wav_id,"subbank":target.group,
            "pair":"/".join(FINAL_PAIRS[pair_index]),
            "target_final":target.final,
            "target_context":target.context,
            "same_donor_wav_id":same.wav_id,
            "cross_donor_wav_id":cross.wav_id,
            "same_duration_mismatch_ms":round(ds,4),
            "cross_duration_mismatch_ms":round(dc,4),
            "delta_reconstruction":round(b[0]-a[0],7),
            "delta_fixed_right":round(b[1]-a[1],7),
            "delta_energy_db":round(b[2]-a[2],7),
        })
    return trials,dict(sorted(reasons.items()))

def co_recorded_pairs(units:list[Unit]):
    """Within-WAV corresponding finals: diagnostic, not an eligible donor."""
    groups=defaultdict(lambda:defaultdict(list))
    for u in units:
        if u.final not in PAIR_OF:continue
        pair_id,_=PAIR_OF[u.final]
        groups[(u.group,u.wav_id,pair_id)][u.final].append(u)
    pairs=[]
    for (subbank,wav_id,pair_id),finals in sorted(groups.items()):
        plain,rounded=FINAL_PAIRS[pair_id]
        if not finals.get(plain) or not finals.get(rounded):continue
        # Aggregate aliases within a physical WAV/final to avoid OTO pseudoreplication.
        x=np.mean([u.core for u in finals[plain]],axis=0)
        y=np.mean([u.core for u in finals[rounded]],axis=0)
        pairs.append({"wav_id":wav_id,"subbank":subbank,
                      "pair":f"{plain}/{rounded}",
                      "within_WAV_core_distance":round(_distance(tuple(x),tuple(y)),7),
                      "plain_aliases":len(finals[plain]),
                      "rounded_aliases":len(finals[rounded])})
    return pairs

def analyze_bank(root:Path,*,base:str="sh"):
    units,extraction=extract(root,base=base)
    trials,reasons=matched_trials(units)
    same_wav=co_recorded_pairs(units)
    per_pair={}
    for a,b in FINAL_PAIRS:
        key=f"{a}/{b}"
        group=[t for t in trials if t["pair"]==key]
        reference=[p for p in same_wav if p["pair"]==key]
        per_pair[key]={
            "strict_targets":len(group),
            "strict_target_WAVs":len({t["wav_id"] for t in group}),
            "co_recorded_reference_pairs":len(reference),
            "co_recorded_WAVs":len({p["wav_id"] for p in reference}),
            "strict_reconstruction":paired_stats(group,"delta_reconstruction"),
            "strict_fixed_right":paired_stats(group,"delta_fixed_right"),
        }
    grouped=defaultdict(list)
    for t in trials:grouped[t["target_context"]].append(t)
    def stats(rows,field):
        return paired_stats(rows,field)
    donor_use=Counter()
    for t in trials:
        donor_use[t["same_donor_wav_id"]]+=1
        donor_use[t["cross_donor_wav_id"]]+=1
    return {
        "extraction":extraction,
        "strict_target_events":len(trials),
        "strict_independent_target_WAVs":len({t["wav_id"] for t in trials}),
        "excluded_reason_counts":reasons,
        "paired_final_groups":per_pair,
        "strict_overall":{
            "reconstruction":stats(trials,"delta_reconstruction"),
            "fixed_right":stats(trials,"delta_fixed_right"),
            "energy":stats(trials,"delta_energy_db"),
        },
        "target_contexts":{
            ctx:{"events":len(group),
                 "reconstruction":stats(group,"delta_reconstruction")}
            for ctx,group in sorted(grouped.items())
        },
        "donor_reuse":{
            "distinct_donor_WAVs":len(donor_use),
            "max_target_arms_per_donor_WAV":max(donor_use.values(),default=0),
            "donor_WAVs_used_more_than_once":sum(v>1 for v in donor_use.values()),
        },
        "same_WAV_diagnostic":{
            "paired_units":len(same_wav),
            "physical_WAVs":len({p["wav_id"] for p in same_wav}),
            "mean_core_distance":round(float(np.mean([p["within_WAV_core_distance"] for p in same_wav])),7) if same_wav else None,
            "comparative_status":"not_a_heldout_substitution_test",
            "pairs":same_wav,
        },
        "strict_trials":trials,
    }

def run_manifest(manifest:Path,output:Path,*,base:str="sh"):
    data=json.loads(manifest.expanduser().resolve(strict=True).read_text("utf-8"))
    if data.get("schema")!="phonoweave.voicebank_replication_manifest.v1":
        raise ValueError("invalid voicebank manifest")
    ids=set();speakers=set();banks=[]
    for b in data["voicebanks"]:
        if b["bank_id"] in ids or b["speaker_group_id"] in speakers:
            raise ValueError("duplicated bank/speaker identifier")
        ids.add(b["bank_id"]);speakers.add(b["speaker_group_id"])
        bank=analyze_bank(Path(b["path"]),base=base)
        bank.update(bank_id=b["bank_id"],speaker_group_id=b["speaker_group_id"])
        banks.append(bank)
    report={
        "schema":SCHEMA,"base":base,
        "final_pairs":[list(p) for p in FINAL_PAIRS],
        "duration_caliper_ms":CALIPER_MS,
        "duration_arm_balance_ms":BALANCE_MS,
        "banks":banks,
        "no_audio_created":True,"source_WAV_or_OTO_modified":False,
        "limitations":[
            "Identical-final same-arm vs paired-rounded-final cross-arm matching controls gross vowel-family differences, not the actual glide/vowel acoustic contrast.",
            "Available repetitions may be sparse; insufficient status is not a null effect.",
            "A physical WAV can contain multiple OTO events; within-WAV final contrasts are descriptive only.",
            "Donor WAV reuse induces dependent target contrasts; WAV-level bootstrap does not address all donor dependence.",
            "OTO offset/preutterance remains an engine-control approximation rather than verified frication-only boundaries.",
            "A positive difference is an offline log-spectrum proxy, not proof of real synthesis impairment, audibility or mandatory unit splitting.",
        ],
    }
    output=output.expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n","utf-8")
    return report
