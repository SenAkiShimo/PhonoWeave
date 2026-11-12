"""v11: source-disjoint, cross-style acoustic-pattern transfer audit.

Acoustic embeddings are taken from the v9 audio-only extractor. Carrier keys and
bank/style metadata are NEVER feature inputs, used only to split or audit sources.
All language/phoneme interpretation is explicitly excluded: a region is not a phone.
The historical audio snippets were originally OTO-curated, although no OTO times
or language symbols are consulted during inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .native_acoustic_subtypes_v9 import adjusted_rand_index
from .native_motif_link_v10 import (
    distance2, extract_generic_full_wav_zip, fit_kmeans, group_split, load_v9, model_audit,
)

SCHEMA = 'phonoweave.context_transfer_audit.v1.1'
SEED = 20261111
STYLE_ARI_GATE = 0.65  # fixed exploratory guard, not phone accuracy
CONTEXT_SUPPORT_COUNT = 2
RANDOM_CONTROLS = 24
ALPHAS = (10., 100., 1000.)


def _style(row: dict) -> str:
    return str(row['bank']).split('_', 1)[0]


def _keys(row: dict):
    return (str(row['wav_archive_entry']), str(row['source_pcm_sha256']),
            int(row['region_index']), float(row['start_ms']), float(row['end_ms']))


def validate_frozen_v10(v9_rows: list[dict], v10_dir: Path) -> dict:
    """Check that we didn't silently change original regions or PCM identities.

    v10 cluster IDs are checked for alignment but not fed into v11 fitting.
    """
    x = json.loads((v10_dir/'regions.json').read_text(encoding='utf8'))
    s = json.loads((v10_dir/'summary.json').read_text(encoding='utf8'))
    if x.get('schema') != 'phonoweave.cross_recording_acoustic_motifs.v1.0' or x.get('schema') != s.get('schema'):
        raise ValueError('unrecognized v10 result schema')
    observed = x.get('regions', [])
    if len(v9_rows) != len(observed) or any(_keys(a) != _keys(b) for a,b in zip(v9_rows,observed)):
        raise ValueError('frozen v9/v10 region or source PCM identity mismatch')
    if s.get('regions_total') != len(v9_rows):
        raise ValueError('v10 summary region count mismatch')
    return {'v10_region_keys_and_pcm_verified': True,
            'v10_input_source_zip_sha256': s.get('input_source_zip_sha256')}


def style_replicability(Y: np.ndarray, rows: list[dict], train: np.ndarray,
                         validation: np.ndarray, audit: np.ndarray, k: int) -> dict:
    """Fit each available style separately using training carrier families ONLY.

    Agreement on validation is the predeclared gate. Audit is descriptive and
    cannot rescue a model that failed on validation.
    """
    styles = sorted({_style(r) for r in rows})
    if k <= 1:
        return {'status': 'one_broad_acoustic_mode_no_subtype_claim',
                'validation_style_ari': None, 'audit_style_ari_diagnostic': None,
                'styles': styles, 'gate_pass': False}
    if len(styles) < 2:
        return {'status': 'abstain_less_than_two_styles',
                'validation_style_ari': None, 'audit_style_ari_diagnostic': None,
                'styles': styles, 'gate_pass': False}
    if len(styles) != 2:
        return {'status': 'abstain_style_pair_not_predeclared',
                'validation_style_ari': None, 'audit_style_ari_diagnostic': None,
                'styles': styles, 'gate_pass': False}
    centers = []
    counts = {}
    for style in styles:
        mask = train & np.asarray([_style(r) == style for r in rows], dtype=bool)
        groups = {r['clip_group'] for r,m in zip(rows,mask) if m}
        counts[style] = {'train_regions':int(mask.sum()),'independent_carriers':len(groups)}
        if mask.sum() < max(15, 3*k) or len(groups) < 5:
            return {'status':'abstain_insufficient_cross_style_training',
                    'validation_style_ari':None,'audit_style_ari_diagnostic':None,
                    'styles':styles,'training_by_style':counts,'gate_pass':False}
        try:
            centers.append(fit_kmeans(Y[mask],k,seed=SEED+5,restarts=9))
        except ValueError:
            return {'status':'abstain_style_model_degenerate',
                    'validation_style_ari':None,'audit_style_ari_diagnostic':None,
                    'styles':styles,'training_by_style':counts,'gate_pass':False}
    if int(validation.sum()) < 10:
        raise ValueError('style gate validation group empty')
    predicted_val = [distance2(Y[validation], c).argmin(axis=1) for c in centers]
    val_ari = adjusted_rand_index(predicted_val[0], predicted_val[1])
    # Unseen audit is only used AFTER the gate has been computed.
    predicted_audit = [distance2(Y[audit],c).argmin(axis=1) for c in centers]
    audit_ari = adjusted_rand_index(predicted_audit[0],predicted_audit[1]) if int(audit.sum())>=2 else None
    passed = bool(val_ari >= STYLE_ARI_GATE)
    return {'status':'exploratory_cross_style_replicable' if passed else 'abstain_cross_style_disagreement',
            'validation_style_ari':round(float(val_ari),5),
            'audit_style_ari_diagnostic':None if audit_ari is None else round(float(audit_ari),5),
            'styles':styles,'training_by_style':counts,
            'validation_gate_threshold':STYLE_ARI_GATE,'gate_pass':passed,
            'audit_not_used_for_model_or_gate_selection':True}


def _other_carrier_hits(Y: np.ndarray, rows: list[dict], labels: np.ndarray,
                        known: np.ndarray, development: np.ndarray,
                        audit: np.ndarray) -> dict:
    """Two independent carriers and two styles required for an audit motif echo.

    A distance radius is calibrated on development-only cross-carrier geometry.
    It is not a label equivalence or phoneme prediction score.
    """
    motifs = sorted({int(labels[i]) for i in np.flatnonzero(known & development)})
    radius = {}
    for k in motifs:
        ids = np.flatnonzero(known & development & (labels == k))
        second = []
        for i in ids:
            pairs = []
            for j in ids:
                if i==j or rows[i]['clip_group'] == rows[j]['clip_group']:
                    continue
                pairs.append((float(np.sum((Y[i]-Y[j])**2)),str(rows[j]['clip_group'])))
            pairs.sort()
            unique = []
            seen = set()
            for d,group in pairs:
                if group not in seen:
                    seen.add(group); unique.append(d)
                    if len(unique)>=CONTEXT_SUPPORT_COUNT:break
            if len(unique)>=CONTEXT_SUPPORT_COUNT:
                second.append(unique[CONTEXT_SUPPORT_COUNT-1])
        radius[k] = float(np.quantile(second,0.90))*1.10 if len(second)>=6 else 0.
    results=[]
    for i in np.flatnonzero(audit):
        k = int(labels[i]);hits=[]
        if known[i] and radius.get(k,0.)>0:
            ids=np.flatnonzero(known & development & (labels==k))
            for j in ids:
                if rows[i]['clip_group']==rows[j]['clip_group'] or rows[i]['source_pcm_sha256']==rows[j]['source_pcm_sha256']:
                    continue
                dist = float(np.sum((Y[i]-Y[j])**2))
                if dist <= radius[k]: hits.append((dist,str(rows[j]['clip_group']),_style(rows[j])))
        hits.sort()
        closest_carriers={}
        for d,c,style in hits:
            closest_carriers.setdefault(c,(d,style))
        passed = (known[i] and len(closest_carriers)>=CONTEXT_SUPPORT_COUNT and
                  len({s for _,s in closest_carriers.values()})>=2)
        results.append({'v9_region_key':_keys(rows[i]), 'v11_motif':k if known[i] else None,
                        'candidate_known':bool(known[i]),
                        'distinct_dev_carrier_contexts':len(closest_carriers),
                        'distinct_dev_styles':len({s for _,s in closest_carriers.values()}),
                        'cross_context_supported_NOT_phone_identity':bool(passed),
                        'status':'acoustic_echo_candidate_NOT_phone' if passed else 'unknown_or_context_support_insufficient'})
    return {'heldout_regions':len(results),
            'heldout_known_regions':sum(r['candidate_known'] for r in results),
            'heldout_cross_context_echo_candidates':sum(r['cross_context_supported_NOT_phone_identity'] for r in results),
            'development_distance_radii':{str(k):round(v,5) for k,v in radius.items()},
            'carrier_support_minimum':CONTEXT_SUPPORT_COUNT,
            'style_support_minimum':2,
            'heldout_records':results}


def _ridge_predict(xtr, ytr, xev, alpha):
    """Single lambda-isotropic ridge, with intercept, no test-time fitting."""
    xm=xtr.mean(axis=0);ym=ytr.mean(axis=0)
    a=xtr-xm; b=ytr-ym
    beta=np.linalg.solve(a.T@a+alpha*np.eye(xtr.shape[1]),a.T@b)
    return (xev-xm)@beta+ym


def _scores(pred, baseline, truth):
    denom=float(np.sum((truth-baseline)**2))
    if denom<1e-12:return None
    return float(1-np.sum((truth-pred)**2)/denom)


def neighboring_acoustics_transfer(all_rows: list[dict], all_X: np.ndarray,
                                  phase_rows: list[dict], train: np.ndarray,
                                  validation: np.ndarray, audit: np.ndarray,
                                  target_phase: int) -> dict:
    """Exploratory contextual effect: can phase-0 acoustics predict target region?

    Style/pitch baseline and within-bank shuffled-neighbor negative controls
    distinguish this test from a blind end-to-end neural phone recognizer.
    This is correlation among adjacent acoustic stages, not causal evidence.
    """
    if target_phase==0:
        return {'status':'no_prior_acoustic_stage_available'}
    origin={}
    for j,r in enumerate(all_rows):
        if int(r['region_index'])==0:
            key=str(r['wav_archive_entry'])
            if key in origin: raise ValueError('duplicate phase zero in one audio clip')
            origin[key]=all_X[j]
    target_idx=[];predictors=[]
    for j,r in enumerate(phase_rows):
        key=str(r['wav_archive_entry'])
        if key not in origin:return {'status':'abstain_missing_acoustic_context'}
        target_idx.append(j); predictors.append(origin[key])
    predictors=np.asarray(predictors,dtype=np.float64)
    if predictors.shape!=(len(phase_rows),48):raise ValueError('invalid context features')
    # Note that phase_rows and the target matrix are supplied together below.
    return {'status':'context_features_ready','matrix':predictors}


def ridge_context_control(context: np.ndarray, target: np.ndarray,
                          rows: list[dict], train: np.ndarray,
                          validation: np.ndarray, audit: np.ndarray) -> dict:
    if context.shape!=target.shape or context.shape[1]!=48 or len(rows)!=len(context):
        raise ValueError('invalid paired acoustic contexts')
    for mask in (train,validation,audit):
        if int(mask.sum()) < 12: return {'status':'abstain_insufficient_context_pairs'}
    if not np.isfinite(context).all() or not np.isfinite(target).all():
        raise ValueError('nonfinite context acoustic vectors')
    cx=context[train].mean(axis=0);sx=np.maximum(context[train].std(axis=0),max(.05,float(context[train].std(axis=0).max())*.2))
    ty=target[train].mean(axis=0);sy=np.maximum(target[train].std(axis=0),max(.05,float(target[train].std(axis=0).max())*.2))
    X=np.clip((context-cx)/sx,-6,6)
    T=np.clip((target-ty)/sy,-6,6)
    # Baseline per bank, trained without validation or audit. Bank is a nuisance
    # covariate used only for matching/control, not an acoustic label/phone ID.
    by_bank={b:T[train & np.array([r['bank']==b for r in rows])].mean(axis=0)
             for b in sorted(set(r['bank'] for r,m in zip(rows,train) if m))}
    baseline=np.stack([by_bank.get(r['bank'],T[train].mean(axis=0)) for r in rows])
    chosen=None;best=-math.inf;estimates={}
    for alpha in ALPHAS:
        pred=_ridge_predict(X[train],T[train],X[validation],alpha)
        score=_scores(pred,baseline[validation],T[validation])
        if score is None: return {'status':'abstain_degenerate_bank_baseline'}
        estimates[str(alpha)]=round(score,5)
        if score>best:best=score;chosen=alpha
    negative=[]
    for rep in range(RANDOM_CONTROLS):
        rng=np.random.default_rng(SEED+rep)
        altered=X[train].copy();banks=np.array([r['bank'] for r in rows])[train]
        for b in np.unique(banks):
            idx=np.flatnonzero(banks==b)
            if len(idx)>=2: altered[idx]=altered[rng.permutation(idx)]
        # Each null candidate is allowed to choose its own alpha on validation;
        # this avoids unfairly fixing the null to the observed best alpha.
        null_scores=[]
        for alpha in ALPHAS:
            pp=_ridge_predict(altered,T[train],X[validation],alpha)
            null_scores.append(_scores(pp,baseline[validation],T[validation]))
        negative.append(max(null_scores))
    # Score audit only after lambda and null test are fixed on validation.
    pred_audit=_ridge_predict(X[train],T[train],X[audit],chosen)
    audit_score=_scores(pred_audit,baseline[audit],T[audit])
    p=(1+sum(x>=best for x in negative))/(1+RANDOM_CONTROLS)
    validated=bool(best>0.03 and p<=0.05)
    return {'status':('exploratory_context_predictability_on_validation' if validated else
                     'abstain_no_robust_context_prediction_on_validation'),
            'development_validation_improvement_vs_bank_baseline':round(best,5),
            'audit_improvement_vs_bank_baseline_descriptive':None if audit_score is None else round(audit_score,5),
            'validation_lambda_candidates':estimates, 'selected_lambda':chosen,
            'shuffled_within_bank_null_median':round(float(np.median(negative)),5),
            'shuffled_within_bank_null_max':round(float(np.max(negative)),5),
            'null_comparison_p_exploratory':round(p,5),
            'negative_control_repetitions':RANDOM_CONTROLS,
            'valid_on_validation_not_causal':validated,
            'bank_baseline_train_only':True, 'audit_not_used_for_selection':True}


def run(v9:Path|None, output:Path, v10:Path|None=None, max_k:int=4,
        full_wav_zip:Path|None=None) -> dict:
    output=Path(output)
    if output.exists():raise FileExistsError('refusing to overwrite prior research output')
    if not 1<=max_k<=7:raise ValueError('max_k outside supported diagnostic range')
    if (v9 is None)==(full_wav_zip is None):
        raise ValueError('provide exactly one of v9 results or full WAV archive')
    if v9 is not None:
        rows,X,source=load_v9(Path(v9))
        source_sha=source['archive_sha256']
        provenance='historical_OTO_curated_600ms_snippets_retroactive'
        input_schema=source['schema']
    else:
        archive=Path(full_wav_zip)
        rows,X=extract_generic_full_wav_zip(archive)
        source_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
        provenance='unanchored_full_recordings_from_user_supplied_WAV_archive'
        input_schema='phonoweave.full_wav_audio_only_extractor.v1.0'
    if not rows:raise ValueError('no acoustic regions')
    if any(len(str(r.get('source_pcm_sha256','')))!=64 or
           any(c not in '0123456789abcdef' for c in str(r['source_pcm_sha256']).lower()) for r in rows):
        raise ValueError('missing or invalid 64-digit source PCM SHA256')
    if len(rows)!=len({_keys(r) for r in rows}):raise ValueError('duplicate acoustic region identity')
    if any(not (isinstance(r.get('clip_group'),str) and r['clip_group'] and
                isinstance(r.get('bank'),str) and r['bank']) for r in rows):
        raise ValueError('missing carrier or bank grouping identifiers')
    source_shas=defaultdict(set)
    for r in rows:source_shas[r['source_pcm_sha256']].add(r['clip_group'])
    if any(len(v)>1 for v in source_shas.values()):raise ValueError('same PCM across carrier groups risks leakage')
    frozen=validate_frozen_v10(rows,Path(v10)) if v10 is not None else {'v10_region_keys_and_pcm_verified':False}
    if v10 is not None and frozen['v10_input_source_zip_sha256'] != source_sha:
        raise ValueError('v10 source ZIP hash mismatches supplied acoustic input')
    all_X=np.asarray(X,dtype=np.float64)
    result=[]; heldout=[]
    for phase in sorted(set(int(r['region_index']) for r in rows)):
        ids=[i for i,r in enumerate(rows) if int(r['region_index'])==phase]
        sub=[rows[i] for i in ids]; features=all_X[ids]
        item={'acoustic_stage_index':phase,'regions':len(ids),
              'carrier_context_groups':len({r['clip_group'] for r in sub}),
              'style_count':len({_style(r) for r in sub}),
              'phoneme_identity_established':False}
        if len(sub)<45 or item['carrier_context_groups']<10:
            item['status']='abstain_insufficient_stage_data';result.append(item);continue
        try:
            tr,val,aud=group_split(sub)
            m,labels,known,hidden=model_audit(features,sub,tr,val,aud,max_k=max_k)
            style=style_replicability(hidden['normalized'],sub,tr,val,aud,m['selected_k'])
            echo=_other_carrier_hits(hidden['normalized'],sub,labels,known,tr|val,aud)
            context_spec=neighboring_acoustics_transfer(rows,all_X,sub,tr,val,aud,phase)
            context=(ridge_context_control(context_spec['matrix'],features,sub,tr,val,aud)
                     if context_spec['status']=='context_features_ready' else {'status':context_spec['status']})
            qualified=bool(m['selected_k']>1 and style['gate_pass'])
            item.update({'status':('exploratory_multiple_cross_style_repeated_acoustic_modes'
                                   if qualified else 'no_cross_style_supported_multiple_modes'),
                         'global_within_stage_model_selection':m['selection_status'],
                         'selected_within_stage_k_before_style_gate':m['selected_k'],
                         'group_resample_validation_ari':m['chosen_stability_median_ari'],
                         'source_context_train_validation_audit':m['source_groups_train_validation_audit'],
                         'style_replicability':style,
                         'cross_carrier_recurrence':{k:v for k,v in echo.items() if k!='heldout_records'},
                         'neighbor_context_predictability':context,
                         'automatic_new_user_units_authorized':False})
            heldout.extend([{'acoustic_stage_index':phase,**r} for r in echo['heldout_records']])
        except ValueError as e:
            item.update(status='abstain_model_or_input_invalid',reason=str(e))
        result.append(item)
    report={'schema':SCHEMA,'acoustic_input_schema':input_schema,
            'source_archive_sha256':source_sha,
            'source_provenance':provenance,
            'regions_total':len(rows),'carrier_groups_total':len({r['clip_group'] for r in rows}),
            'styles_total':len({_style(r) for r in rows}),
            'v10_provenance_check':frozen,
            'stage_results':result,
            'stages_approving_new_phone_or_alias':0,
            'independent_phone_boundaries_available':False,
            'cross_style_thresholds_are_exploratory':True,
            'not_seven_independent_speakers':True,
            'no_ipa_g2p_oto_timestamp_or_transcript_in_model':True,
            'no_voicebank_or_reclist_modified':True}
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (output/'audit_region_recurrence.json').write_text(json.dumps({'schema':SCHEMA,'heldout_regions':heldout},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return report


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(prog='phonoweave v11 cross-context acoustic audit')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--v9-results',type=Path)
    source.add_argument('--full-wav-zip',type=Path)
    p.add_argument('--v10-results',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--max-k',type=int,default=4)
    a=p.parse_args(argv)
    result=run(a.v9_results,a.output,a.v10_results,a.max_k,a.full_wav_zip)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())