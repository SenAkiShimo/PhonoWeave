"""Read-only physical-WAV inventory and exact-final counterfactual feasibility.

Counts are observational; additions are lower-bound planning quantities
under explicitly stated no-existing-donor-reuse requirements, NOT optimal
recording prescriptions or evidence of synthesis necessity.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import json
from pathlib import Path
from .synthesis_relevance_unit_merging_cost import extract
from .synthesis_relevance_final_matched import (
    FINAL_PAIRS,PAIR_OF,CALIPER_MS,BALANCE_MS,matched_trials
)

SCHEMA="phonoweave.synthesis_relevance.inventory_feasibility.v1.0"

def _different_wavs(pool,target):
    return [d for d in pool if d.wav_id!=target.wav_id]

def _best_distance(pool,target):
    xs=_different_wavs(pool,target)
    return min((abs(u.duration_ms-target.duration_ms) for u in xs),default=None)

def feasibility(units):
    """Enumerate distinct source WAVs and identify mutually exclusive blockers.

    Inventory reports all eligible v0.8 extracted units. Same-final target
    multiplicity within one WAV never constitutes independent replicates.
    """
    by_final=defaultdict(list)
    by_group=defaultdict(list)
    for u in units:
        by_final[(u.group,u.final)].append(u)
        by_group[u.group].append(u)
    groups=[]
    totals=Counter()
    for group in sorted(by_group):
        events=by_group[group]
        finals=[]
        for final in sorted({u.final for u in events}):
            rows=by_final[(group,final)]
            wavs={u.wav_id for u in rows}
            pair_info=PAIR_OF.get(final)
            finals.append({
                "final":final,"context":rows[0].context,
                "oto_events":len(rows),"physical_WAVs":len(wavs),
                "aliased_or_repeated_events_in_same_WAV":len(rows)-len(wavs),
                "preutterance_duration_ms":{
                    "minimum":round(min(u.duration_ms for u in rows),3),
                    "maximum":round(max(u.duration_ms for u in rows),3)},
                "prespecified_pair":"/".join(FINAL_PAIRS[pair_info[0]]) if pair_info else None,
            })
        pairs=[]
        for a,b in FINAL_PAIRS:
            left=by_final[(group,a)];right=by_final[(group,b)]
            left_ids={u.wav_id for u in left}
            right_ids={u.wav_id for u in right}
            # One held-out target from each side needs >=2 independent WAVs
            # for that side and >=1 independent WAV for opposite side.
            # Planning counts target both directions, so >=2 per final.
            # The two sides might be co-recorded on the same physical WAV.
            n_left=len(left_ids);n_right=len(right_ids)
            minimum_each=max(0,2-n_left)+max(0,2-n_right)
            union=len(left_ids|right_ids)
            # Number of NEW WAV files is at least max(per-final deficits);
            # a new WAV recording both finals could satisfy both deficits.
            lower_bound_new_WAVs=max(max(0,2-n_left),max(0,2-n_right))
            pairs.append({
                "pair":f"{a}/{b}",
                "plain_WAVs":n_left,"rounded_WAVs":n_right,
                "union_WAVs":union,
                "co_recorded_WAVs":len(left_ids&right_ids),
                "bidirectional_two_WAV_per_final_feasible":n_left>=2 and n_right>=2,
                "minimum_additional_final_recordings_for_two_per_side":minimum_each,
                "lower_bound_additional_WAVs_if_both_finals_can_share_each_new_WAV":lower_bound_new_WAVs,
                "limitations":"Structural replication lower bound only; no guarantee of 10ms caliper or 5ms balance.",
            })
        groups.append({
            "subbank":group,
            "oto_events":len(events),
            "physical_WAVs":len({u.wav_id for u in events}),
            "finals":finals,"pairs":pairs,
        })
        totals["subbanks"]+=1
        totals["final_cells"]+=len(finals)
        totals["pair_subbank_cells"]+=len(pairs)
        totals["structurally_bidirectional_pair_cells"]+=sum(
            p["bidirectional_two_WAV_per_final_feasible"] for p in pairs)
        totals["final_recording_deficit_two_each"]+=sum(
            p["minimum_additional_final_recordings_for_two_per_side"] for p in pairs)
        totals["new_WAV_lower_bound_per_pair_sum"]+=sum(
            p["lower_bound_additional_WAVs_if_both_finals_can_share_each_new_WAV"] for p in pairs)

    reasons=Counter()
    by_pair=defaultdict(Counter)
    examples=[]
    for t in units:
        pair_info=PAIR_OF.get(t.final)
        if pair_info is None:
            reasons["not_in_prespecified_final_pairs"]+=1;continue
        pair_index,side=pair_info
        alternative=FINAL_PAIRS[pair_index][1-side]
        same=by_final[(t.group,t.final)]
        cross=by_final[(t.group,alternative)]
        sd=_best_distance(same,t);cd=_best_distance(cross,t)
        same_exists=sd is not None
        cross_exists=cd is not None
        if not same_exists:state="missing_independent_same_final"
        elif not cross_exists:state="missing_independent_opposite_final"
        elif sd>CALIPER_MS:state="same_final_duration_caliper_failure"
        elif cd>CALIPER_MS:state="opposite_final_duration_caliper_failure"
        elif abs(sd-cd)>BALANCE_MS:state="arm_duration_imbalance"
        else:state="eligible"
        reasons[state]+=1
        pairname="/".join(FINAL_PAIRS[pair_index])
        by_pair[pairname][state]+=1
        if len(examples)<32 and state!="eligible":
            examples.append({
                "subbank":t.group,"target_final":t.final,
                "target_WAV":t.wav_id,
                "status":state,
                "closest_independent_same_final_ms":round(sd,3) if sd is not None else None,
                "closest_independent_opposite_final_ms":round(cd,3) if cd is not None else None,
            })
    eligible,_=matched_trials(units)
    if len(eligible)!=reasons["eligible"]:
        raise AssertionError("feasibility count differs from frozen strict matching")
    return {
        "extracted_unit_events":len(units),
        "physical_WAVs":len({u.wav_id for u in units}),
        "subbanks":groups,
        "aggregate_counts":dict(totals),
        "mutually_exclusive_target_statuses":dict(sorted(reasons.items())),
        "pair_target_statuses":{p:dict(sorted(c.items())) for p,c in sorted(by_pair.items())},
        "strict_trial_replay":len(eligible),
        "diagnostic_examples":examples,
    }


def run_manifest(manifest:Path,output:Path,*,base:str="sh"):
    data=json.loads(manifest.expanduser().resolve(strict=True).read_text("utf-8"))
    if data.get("schema")!="phonoweave.voicebank_replication_manifest.v1":
        raise ValueError("unsupported voicebank manifest")
    seen=set();result=[]
    for entry in data["voicebanks"]:
        if entry["bank_id"] in seen:raise ValueError("duplicate bank ID")
        seen.add(entry["bank_id"])
        units,metadata=extract(Path(entry["path"]),base=base)
        report=feasibility(units)
        report.update(bank_id=entry["bank_id"],
                      speaker_group_id=entry["speaker_group_id"],
                      extraction=metadata)
        result.append(report)
    report={
        "schema":SCHEMA,"base":base,"final_pairs":[list(pair) for pair in FINAL_PAIRS],
        "strict_duration_caliper_ms":CALIPER_MS,
        "strict_arm_balance_ms":BALANCE_MS,
        "banks":result,"no_audio_written":True,
        "source_WAV_OTO_modified":False,
        "cautions":[
            "Minimum additional recordings measure coverage targets, not guaranteed counterfactual eligibility.",
            "Lower-bound new WAV counts presume both finals may be co-recorded in one new independent WAV.",
            "Pair-specific new-WAV lower bounds cannot be summed into a proven global minimum because recording scripts can share files.",
            "Physical WAV independence is not the same as independent recording sessions or speakers.",
            "The inventory cannot certify a frication-only segment or synthesis quality.",
        ],
    }
    path=output.expanduser().resolve();path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report
