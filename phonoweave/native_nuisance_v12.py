"""v12: paired-family bank-effect hypotheses with explicit protected acoustic geometry.

This is exploratory acoustic nuisance auditing, NOT a learned phoneme representation.
Training fits only source-family-matched bank translations. Metadata is used for
pairing/splitting only; no phone labels, IPA, OTO times, or transcripts enter model.
Corrections are exported separately and never applied to production inventories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .native_motif_link_v10 import load_v9, extract_generic_full_wav_zip, group_split, normalize
from .native_context_transfer_v11 import validate_frozen_v10, neighboring_acoustics_transfer, ridge_context_control

SCHEMA = 'phonoweave.paired_bank_context_decomposition.v1.2'
SEED = 20261112
STRENGTHS = (0., 0.25, 0.5, 0.75, 1.)
PERMUTATIONS = 32
MIN_GROUPS_PER_BANK = 6
MIN_VALIDATION_TOP1_GAIN = 0.025
MIN_VALIDATION_RATIO_GAIN = 0.04
MAX_NULL_P = 0.05


def _style(bank: str) -> str:
    return bank.split('_', 1)[0]


def _records(rows: list[dict], X: np.ndarray, mask: np.ndarray):
    """Aggregate repeat clips within carrier × bank, equally weighting families."""
    data = defaultdict(list)
    for i in np.flatnonzero(mask):
        r=rows[int(i)]
        data[(str(r['clip_group']),str(r['bank']))].append(X[int(i)])
    return {key:np.mean(vec,axis=0) for key,vec in data.items()}


def validate_source(rows: list[dict], X: np.ndarray) -> None:
    if X.shape!=(len(rows),48) or not np.isfinite(X).all() or not rows:
        raise ValueError('requires nonempty finite aligned 48D audio features')
    identities=set(); pcm_families=defaultdict(set)
    for row in rows:
        bank=row.get('bank'); family=row.get('clip_group'); digest=row.get('source_pcm_sha256');wav=row.get('wav_archive_entry')
        if not all(isinstance(x,str) and x.strip() for x in (bank,family,wav,digest)):
            raise ValueError('source metadata missing bank, group, wav or PCM digest')
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest.lower()):
            raise ValueError('invalid source PCM SHA256')
        key=(wav,digest,int(row['region_index']),float(row['start_ms']),float(row['end_ms']))
        if key in identities: raise ValueError('duplicate region identity')
        identities.add(key)
        pcm_families[digest].add(family)
        if not (0<=float(row['start_ms'])<float(row['end_ms'])):
            raise ValueError('invalid acoustic interval')
    if any(len(x)!=1 for x in pcm_families.values()):
        raise ValueError('same PCM in multiple source families')


def fit_offsets(rows: list[dict], X: np.ndarray, train: np.ndarray,
                shuffle_seed: int | None = None) -> tuple[dict[str,np.ndarray],dict]:
    """Source-paired, train-only bank effects, optionally permuting bank identities.

    Every carrier gets equal weight. A paired group's center is formed from
    equally weighted banks rather than allowing more repeats to dominate.
    """
    pairs=_records(rows,X,train)
    by_group=defaultdict(dict)
    for (g,b),y in pairs.items(): by_group[g][b]=y
    banks=sorted({b for _,b in pairs})
    styles=sorted({_style(b) for b in banks})
    if len(styles)<2 or len(banks)<3:
        raise ValueError('abstain: no paired multi-style evidence')
    rng=None if shuffle_seed is None else np.random.default_rng(shuffle_seed)
    deltas=defaultdict(list); used_groups=0
    for group in sorted(by_group):
        bank_features=by_group[group]
        if len({_style(b) for b in bank_features})<2:continue
        ordered_banks=sorted(bank_features)
        vectors=np.stack([bank_features[b] for b in ordered_banks])
        center=vectors.mean(axis=0)
        assigned_banks=ordered_banks if rng is None else [ordered_banks[i] for i in rng.permutation(len(ordered_banks))]
        for bank,vector in zip(assigned_banks,vectors):
            deltas[bank].append(vector-center)
        used_groups+=1
    coverage={b:len(deltas[b]) for b in banks}
    if any(n<MIN_GROUPS_PER_BANK for n in coverage.values()):
        raise ValueError('abstain: paired training coverage insufficient for some bank')
    offsets={b:np.mean(deltas[b],axis=0) for b in banks}
    # Exactly zero mean bank effect (balanced across bank types); support unequal counts.
    return offsets,{'paired_training_carriers':used_groups,'bank_support':coverage,
                    'styles':styles, 'bank_count':len(banks),
                    'original_style_and_pitch_retained_in_raw_features':True,
                    'source_pair_matching_only_on_training_groups':True}


def transform(X: np.ndarray, rows: list[dict], offsets: dict[str,np.ndarray], strength: float) -> np.ndarray:
    if not (0 <= strength <= 1):raise ValueError('strength must lie in [0,1]')
    if any(r['bank'] not in offsets for r in rows):
        raise ValueError('unseen bank cannot be silently corrected')
    return X-np.stack([offsets[r['bank']] for r in rows])*strength


def retrieval(rows: list[dict], X: np.ndarray, mask: np.ndarray) -> dict:
    """Same-carrier top1 across style-different banks vs distractor carriers.

    Group names are an evaluation-only 'same recording script' proxy, not
    known phone identities. Each (query-bank, target-bank, carrier) is one vote.
    """
    pairs=_records(rows,X,mask)
    carriers=sorted({g for g,b in pairs});banks=sorted({b for g,b in pairs})
    hit=0;total=0;pos=[];neg=[]
    for b in banks:
        for other in banks:
            if _style(b)==_style(other):continue
            targets=[(g,pairs[g,other]) for g in carriers if (g,other) in pairs]
            if len(targets)<2:continue
            for g in carriers:
                if (g,b) not in pairs or (g,other) not in pairs:continue
                distances=[(float(np.sum((pairs[g,b]-v)**2)),h) for h,v in targets]
                distances.sort(key=lambda q:(q[0],q[1]))
                hit+=int(distances[0][1]==g);total+=1
                same=next(d for d,h in distances if h==g)
                different=[d for d,h in distances if h!=g]
                pos.append(same);neg.append(float(np.median(different)))
    if total<10 or not neg or min(np.mean(neg),np.mean(pos))<1e-12:
        return {'status':'abstain_insufficient_cross_style_pairs','queries':total,
                'top1_fraction':None,'matched_to_distractor_distance_ratio':None}
    return {'status':'acoustic_carrier_retrieval_diagnostic','queries':total,
            'carriers':len(carriers),'top1_fraction':round(hit/total,6),
            'matched_to_distractor_distance_ratio':round(float(np.mean(pos)/np.mean(neg)),6),
            'mean_matched_distance2':round(float(np.mean(pos)),6),
            'mean_distractor_median_distance2':round(float(np.mean(neg)),6)}


def relative_gain(baseline: dict, alternative: dict) -> dict:
    if baseline['top1_fraction'] is None or alternative['top1_fraction'] is None:
        return {'top1_gain':None,'ratio_gain':None,'combined_score_gain':None}
    top=alternative['top1_fraction']-baseline['top1_fraction']
    ratio=1-alternative['matched_to_distractor_distance_ratio']/baseline['matched_to_distractor_distance_ratio']
    # Combining two evaluation scales only for candidate ranking (not a probability).
    return {'top1_gain':round(top,6),'ratio_gain':round(ratio,6),
            'combined_score_gain':round(top+0.25*ratio,6)}


def within_bank_geometry_preserved(rows: list[dict], raw: np.ndarray, adj: np.ndarray, mask: np.ndarray) -> float:
    """A bank-wise translation should preserve ALL within-bank pairwise distances."""
    if len(raw)!=len(rows) or adj.shape!=raw.shape:return float('inf')
    max_diff=0.
    for bank in sorted({r['bank'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if mask[i] and r['bank']==bank]
        for j in range(len(ids)):
            for k in range(j+1,len(ids)):
                u,v=ids[j],ids[k]
                before=float(np.sum((raw[u]-raw[v])**2))
                after=float(np.sum((adj[u]-adj[v])**2))
                max_diff=max(max_diff,abs(before-after))
    return max_diff


def _select(rows, X, train, validation, offsets) -> tuple[float, list[dict],dict]:
    base=retrieval(rows,X,validation)
    if base['top1_fraction'] is None:raise ValueError('insufficient validation carrier pairing')
    variants=[]
    for alpha in STRENGTHS:
        transformed=transform(X,rows,offsets,alpha)
        metrics=retrieval(rows,transformed,validation)
        gain=relative_gain(base,metrics)
        variants.append({'strength':alpha,**metrics,**gain})
    eligible=[x for x in variants if x['strength']>0]
    chosen=max(eligible,key=lambda s:(s['combined_score_gain'],-s['strength']))
    return chosen['strength'],variants,base


def phase_audit(rows:list[dict],X:np.ndarray,phase:int,full_rows:list[dict],full_X:np.ndarray) -> dict:
    train,val,audit=group_split(rows)
    Y,norm=normalize(X,train)
    offsets,support=fit_offsets(rows,Y,train)
    strength,variants,raw_validation=_select(rows,Y,train,val,offsets)
    chosen=next(q for q in variants if q['strength']==strength)
    calibrated=[]
    for perm in range(PERMUTATIONS):
        random_offsets,_=fit_offsets(rows,Y,train,shuffle_seed=SEED+perm)
        _,candidate_null,_=_select(rows,Y,train,val,random_offsets)
        calibrated.append(max(q['combined_score_gain'] for q in candidate_null if q['strength']>0))
    chosen_gain=chosen['combined_score_gain']
    p=(1+sum(null>=chosen_gain-1e-12 for null in calibrated))/(1+PERMUTATIONS)
    corrected=transform(Y,rows,offsets,strength)
    max_geom=within_bank_geometry_preserved(rows,Y,corrected,audit)
    allowed=bool(chosen['top1_gain']>=MIN_VALIDATION_TOP1_GAIN and
                 chosen['ratio_gain']>=MIN_VALIDATION_RATIO_GAIN and
                 p<=MAX_NULL_P and max_geom<1.e-7)
    raw_audit=retrieval(rows,Y,audit)
    corrected_audit=retrieval(rows,corrected,audit)
    # Does a stage-zero acoustic predecessor predict this phase on new carriers?
    context_spec=neighboring_acoustics_transfer(full_rows,full_X,rows,train,val,audit,phase)
    context_result=(ridge_context_control(context_spec['matrix'],X,rows,train,val,audit)
                    if context_spec['status']=='context_features_ready'
                    else {'status':context_spec['status']})
    report={
        'acoustic_stage_index':phase,'regions':len(rows),'source_groups_train_validation_audit':
        [len({r['clip_group'] for r,t in zip(rows,m) if t}) for m in (train,val,audit)],
        'normalization_train_only':norm,'paired_bank_fit':support,
        'validation_raw':raw_validation,'validation_strength_search':variants,
        'chosen_strength_diagnostic':strength,'matched_bank_identity_permutation_null_count':PERMUTATIONS,
        'permutation_p_exploratory_not_phone_significance':round(float(p),6),
        'validation_gate':{'top1_gain_min':MIN_VALIDATION_TOP1_GAIN,'ratio_gain_min':MIN_VALIDATION_RATIO_GAIN,
                           'null_p_max':MAX_NULL_P},
        'candidate_bank_translation_passes_development_gate':allowed,
        'audit_raw_frozen':raw_audit,'audit_corrected_descriptive':corrected_audit,
        'audit_within_bank_squared_distance_max_abs_change':round(max_geom,12),
        'adjacent_acoustic_context_predictability':context_result,
        'acoustic_context_removal_authorized':False,
        'automatic_new_phone_or_alias_authorized':False,
        'output_is_only_acoustic_bank_translation_hypothesis':True,
    }
    frozen_offsets={b:[round(float(z),8) for z in y] for b,y in offsets.items()}
    return report, {'phase':phase, 'strength':strength if allowed else 0.,
                    'development_gate_pass':allowed,
                    'train_only_bank_offsets':frozen_offsets,
                    'apply_only_if_manually_research_authorized':True}


def run(v9_results:Path|None,output:Path,v10_results:Path|None=None,
        full_wav_zip:Path|None=None)->dict:
    output=Path(output)
    if output.exists():raise FileExistsError('refusing to overwrite previous research output')
    if (v9_results is None)==(full_wav_zip is None):
        raise ValueError('exactly one of --v9-results or --full-wav-zip is required')
    if v9_results is not None:
        rows,X,source=load_v9(Path(v9_results))
        source_hash=source['archive_sha256']
        provenance='historical_OTO_curated_600ms_audio_context_clips'
        if v10_results is not None:
            v10=validate_frozen_v10(rows,Path(v10_results))
            if v10['v10_input_source_zip_sha256']!=source_hash:
                raise ValueError('source archive hash differs from frozen v10')
        else:v10={'v10_region_keys_and_pcm_verified':False}
    else:
        if v10_results is not None:
            raise ValueError('v10 historical results cannot validate a new full WAV archive')
        archive=Path(full_wav_zip)
        rows,X=extract_generic_full_wav_zip(archive)
        source_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
        provenance='unanchored_full_wav_user_supplied_archive'
        v10={'v10_region_keys_and_pcm_verified':False}
    X=np.asarray(X,dtype=np.float64)
    validate_source(rows,X)
    diagnostics=[]; proposals=[]
    for phase in sorted({int(r['region_index']) for r in rows}):
        ids=[i for i,r in enumerate(rows) if int(r['region_index'])==phase]
        part=[rows[i] for i in ids]
        features=X[ids]
        try:
            if len(part)<45:raise ValueError('too few acoustic regions in phase')
            report,sidecar=phase_audit(part,features,phase,rows,X)
            diagnostics.append(report);proposals.append(sidecar)
        except ValueError as err:
            diagnostics.append({'acoustic_stage_index':phase,'regions':len(part),
                                'status':'abstain_no_paired_context_evidence','reason':str(err),
                                'automatic_new_phone_or_alias_authorized':False})
    summary={'schema':SCHEMA,'source_archive_sha256':source_hash,'source_provenance':provenance,
             'v10_identity_checked':v10,'regions_total':len(rows),
             'source_family_count':len({r['clip_group'] for r in rows}),
             'bank_count':len({r['bank'] for r in rows}),
             'stage_results':diagnostics,
             'candidate_bank_corrections_passing_development_gate':sum(p['development_gate_pass'] for p in proposals),
             'context_residualization_deployed':False,'confirmed_phone_identity_or_boundary':False,
             'recording_inventory_split_merge_or_auto_labels':False,
             'ipa_oto_timestamps_or_aliases_used_in_features':False,
             'retrospective_exploration_not_confirmatory':True}
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (output/'research_bank_offsets.json').write_text(json.dumps({'schema':SCHEMA,'proposals':proposals,
                                                                'NOT_production_voicebank_configuration':True},
                                                               ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return summary


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(prog='phonoweave v12 paired bank nuisance hypothesis')
    inp=p.add_mutually_exclusive_group(required=True)
    inp.add_argument('--v9-results',type=Path)
    inp.add_argument('--full-wav-zip',type=Path)
    p.add_argument('--v10-results',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv)
    print(json.dumps(run(a.v9_results,a.output,a.v10_results,a.full_wav_zip),ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
