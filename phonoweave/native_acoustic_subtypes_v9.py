"""Exploratory IPA-free, within-user-label spectral-context subtype discovery.

IMPORTANT provenance boundary: v3 36D context vectors were extracted at frozen
OTO-derived reference positions. This is automatic clustering of available
acoustic embeddings, NOT unanchored phone segmentation, novel-phone discovery,
or sufficient evidence to split a synthesis inventory. No source audio is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

SCHEMA = 'phonoweave.exploratory_acoustic_subtypes.v0.9'
FROZEN_HELDOUT = frozenset(('a.wav', 'ai.wav', 'ang.wav', 'e0.wav', 'en0.wav', 'eriin.wav', 'uvvn.wav', 'vcsection2.wav'))
FEATURE_MODE = 'late_context'
SEED = 20261110
MIN_TRAIN = 60
MIN_HELDOUT = 15


def read_frozen_v3(path: str | Path) -> tuple[list[dict], np.ndarray, str]:
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        report = json.loads(z.read('results/evaluation.json'))
        rows = list(csv.DictReader(io.StringIO(z.read('v2_frozen_inputs/source_index.csv').decode('utf-8-sig'))))
        with np.load(io.BytesIO(z.read('results/temporal_features.npz')), allow_pickle=False) as arrays:
            X = np.asarray(arrays['feat_' + FEATURE_MODE], dtype=np.float64)
    if set(report.get('frozen_test_groups', ())) != FROZEN_HELDOUT or report.get('selected_mode') != FEATURE_MODE:
        raise ValueError('v3 source split / selected feature mode has changed')
    if X.shape != (len(rows), 36) or not np.isfinite(X).all():
        raise ValueError('wrong shape or non-finite acoustic features')
    for row in rows:
        for field in ('bank', 'source_wav', 'symbol', 'context', 'source_pcm_sha256'):
            if not row.get(field):
                raise ValueError('missing source identity or user symbol: ' + field)
        if len(row['source_pcm_sha256']) != 64:
            raise ValueError('invalid source SHA256')
    return rows, X, hashlib.sha256(path.read_bytes()).hexdigest()


def bank_normalization(rows: list[dict], X: np.ndarray, heldout: frozenset[str]):
    """Fit all parameters from DEVELOPMENT templates only; applies to heldout unchanged."""
    dev = np.array([r['source_wav'] not in heldout for r in rows], bool)
    if not dev.any():
        raise ValueError('no development rows')
    centers = {}
    Y = X.copy()
    for b in sorted({r['bank'] for r in rows}):
        bank_idx = np.array([r['bank'] == b for r in rows])
        subset = dev & bank_idx
        if not subset.any():
            raise ValueError('bank has no development rows: '+b)
        centers[b] = X[subset].mean(axis=0)
        Y[bank_idx] -= centers[b]
    scale = np.std(Y[dev], axis=0)
    # Shrink per-axis whitening: do not amplify tiny-noise dimensions to unit scale.
    scale = np.maximum(scale, max(0.02, 0.15 * float(np.max(scale))))
    Y = np.clip(Y / scale, -6., 6.)
    assert np.isfinite(Y).all()
    return Y, dev, {'bank_count': len(centers), 'training_only': True, 'scale_floor_absolute': 0.02, 'scale_floor_relative_to_largest_axis': 0.15}


def kmeans2(X: np.ndarray, seed: int, repeats: int = 12) -> tuple[np.ndarray, np.ndarray, float]:
    """Deterministic two-centroid Euclidean k-means with multiple starts."""
    if X.ndim != 2 or len(X) < 4 or not np.isfinite(X).all():
        raise ValueError('kmeans needs >=4 finite row vectors')
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(repeats):
        first = int(rng.integers(len(X)))
        dist = np.sum((X - X[first])**2, axis=1)
        if dist.max() <= 1.e-12:
            raise ValueError('constant acoustic features')
        second = int(rng.choice(len(X), p=dist / dist.sum()))
        c = np.stack([X[first], X[second]])
        for _ in range(70):
            d = np.sum((X[:, None, :] - c[None, :, :])**2, axis=2)
            y = np.argmin(d, axis=1)
            if min(np.bincount(y, minlength=2)) == 0:
                break
            new = np.stack([X[y == k].mean(axis=0) for k in range(2)])
            if np.allclose(new, c, atol=1.e-8, rtol=0):
                c = new
                break
            c = new
        d = np.sum((X[:, None, :] - c[None, :, :])**2, axis=2)
        y = np.argmin(d, axis=1)
        if min(np.bincount(y, minlength=2)) == 0:
            continue
        loss = float(d[np.arange(len(X)), y].sum())
        if best is None or loss < best[2]:
            best = (c, y, loss)
    if best is None:
        raise ValueError('cannot form nonempty two-means clusters')
    return best


def assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    d = np.sum((X[:, None, :] - centroids[None, :, :])**2, axis=2)
    return np.argmin(d, axis=1)


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or not len(a):
        raise ValueError('ARI requires matched nonempty rows')
    a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
    tab = np.zeros((len(np.unique(a)),len(np.unique(b))), dtype=np.int64)
    ai = {x:i for i,x in enumerate(np.unique(a))}; bi = {x:i for i,x in enumerate(np.unique(b))}
    for x,y in zip(a,b):tab[ai[x],bi[y]] += 1
    choose2 = lambda x: np.sum(x*(x-1)/2)
    n = len(a); total = n*(n-1)/2
    if total == 0:return 1.0
    rows, cols = tab.sum(axis=1),tab.sum(axis=0)
    expected = float(choose2(rows)*choose2(cols)/total)
    maxv = float((choose2(rows)+choose2(cols))/2)
    denom=maxv-expected
    return float((choose2(tab)-expected)/denom) if denom > 1.e-12 else (1. if np.array_equal(a,b) else 0.)


def normalized_mutual_information(group: list[str], clusters: np.ndarray) -> float:
    """Fraction of cluster entropy explained by a metadata group, for confound warning."""
    if len(group) != len(clusters) or not len(group):return 0.
    n=len(group); gs=Counter(group); cs=Counter(map(int,clusters))
    if len(cs) < 2:return 0.
    joint=Counter((g,int(c)) for g,c in zip(group,clusters))
    mi=sum((count/n)*math.log((count*n)/(gs[g]*cs[c])) for (g,c),count in joint.items())
    h=-sum((p/n)*math.log(p/n) for p in cs.values())
    return float(max(0,min(1,mi/h))) if h>0 else 0.


def _support(groups: list[str], y: np.ndarray) -> tuple[int, int]:
    return tuple(len({g for g, yi in zip(groups, y) if yi==k}) for k in (0,1))


def _symbol_seed(symbol: str) -> int:
    return SEED + int(hashlib.sha256(symbol.encode('utf-8')).hexdigest()[:8],16)


def discover(rows: list[dict], X: np.ndarray) -> tuple[list[dict], list[dict], dict]:
    """No IPA or metadata enters the 2-means acoustic clustering itself.

    User labels define groups; contextual labels and source identities are used
    AFTER assigning clusters to audit confounding and source separation.
    """
    if len(rows)!=len(X):raise ValueError('row / feature count differs')
    Y, dev, normalization = bank_normalization(rows, X, FROZEN_HELDOUT)
    banks = sorted(set(r['bank'] for r in rows))
    bank_styles = {b: b.split('_')[0] for b in banks}
    styles = sorted(set(bank_styles.values()))
    if len(styles) < 2:
        raise ValueError('at least two recording styles required for style-replicability check')
    summaries=[]; memberships=[]
    for symbol in sorted({r['symbol'] for r in rows}):
        ids=np.array([i for i,r in enumerate(rows) if r['symbol']==symbol],dtype=int)
        train_ids=ids[dev[ids]]; test_ids=ids[~dev[ids]]
        base={'symbol': symbol, 'dev_samples':len(train_ids), 'heldout_samples':len(test_ids),
              'training_template_count': len(set(rows[i]['source_wav'] for i in train_ids)),
              'heldout_template_count':len(set(rows[i]['source_wav'] for i in test_ids)),
              'status':'insufficient_data_or_reproducibility', 'makes_new_user_alias':False,
              'phone_boundary_verified':False, 'context_anchor_type':'oto_derived_reference_not_phone_boundary'}
        if len(train_ids)<MIN_TRAIN or len(test_ids)<MIN_HELDOUT:
            base['reason']='insufficient_dev_or_frozen_heldout';summaries.append(base);continue
        style_centers={}
        insufficient=False
        for style in styles:
            sub=np.array([i for i in train_ids if bank_styles[rows[i]['bank']]==style],dtype=int)
            if len(sub)<20:
                insufficient=True;break
            try:style_centers[style]=kmeans2(Y[sub], _symbol_seed(symbol)+styles.index(style))[0]
            except ValueError:insufficient=True;break
        if insufficient:
            base['reason']='insufficient_style_training_or_constant_features';summaries.append(base);continue
        try:full_centers, _, full_loss=kmeans2(Y[train_ids],_symbol_seed(symbol)+101)
        except ValueError:
            base['reason']='constant_features';summaries.append(base);continue
        y_train=assign(Y[train_ids],full_centers)
        y_test=assign(Y[test_ids],full_centers)
        proportions=np.bincount(y_test,minlength=2)/len(y_test)
        test_predictions=[assign(Y[test_ids],style_centers[style]) for style in styles]
        style_agreement=adjusted_rand_index(test_predictions[0],test_predictions[1]) if len(styles)==2 else min(adjusted_rand_index(test_predictions[0],q) for q in test_predictions[1:])
        dev_centroid=Y[train_ids].mean(axis=0)
        heldout_loss_1=float(np.sum((Y[test_ids]-dev_centroid)**2))
        heldout_loss_2=float(np.sum((Y[test_ids]-full_centers[y_test])**2))
        gain=(heldout_loss_1-heldout_loss_2)/max(heldout_loss_1,1.e-12)
        # A two-means SSE gain alone is not independent evidence of 2 phones.
        axis=full_centers[1]-full_centers[0]
        length=float(np.linalg.norm(axis))
        projected=((Y[test_ids]-(full_centers[0]+full_centers[1])*0.5) @ axis / max(length,1.e-10))
        # Counts near centroid midpoint, relative to full 2-means separation.
        midpoint_fraction=float(np.mean(np.abs(projected)<=0.25*length))
        dev_wavs=[rows[i]['source_wav'] for i in train_ids]
        wav_support=_support(dev_wavs,y_train)
        bank_support=_support([rows[i]['bank'] for i in train_ids],y_train)
        context_nmi=normalized_mutual_information([rows[i]['context'] for i in train_ids],y_train)
        template_nmi=normalized_mutual_information(dev_wavs,y_train)
        bank_nmi=normalized_mutual_information([rows[i]['bank'] for i in train_ids],y_train)
        # Initial geometry screen; this can flag pitch/bank artifacts. A repeated
        # waveform template across seven subbanks is NOT seven independent tests.
        geometric_signal=(style_agreement>=0.6 and proportions.min()>=0.15
                          and min(wav_support)>=3 and min(bank_support)>=5
                          and midpoint_fraction<=0.45 and gain>=0.07)
        confound_guard=(bank_nmi>=0.25 or context_nmi>=0.50
                        or template_nmi>=0.50 or base['heldout_template_count']<2)
        repeatable=geometric_signal and not confound_guard
        status=('exploratory_repeatable_spectral_context_pattern' if repeatable
                else 'exploratory_confound_or_underreplicated' if geometric_signal
                else 'insufficient_data_or_reproducibility')
        base.update(reason=('cross_style_repeatable_context_pattern' if repeatable
                            else 'preliminary_signal_fails_provenance_or_bank_confound_guard' if geometric_signal
                            else 'no_robust_two_subtype_evidence'),
                    status=status,
                    style_heldout_ari=round(style_agreement,4), heldout_sse_gain_vs_one_centroid=round(gain,4),
                    heldout_midpoint_fraction=round(midpoint_fraction,4),
                    heldout_cluster_min_fraction=round(float(proportions.min()),4),
                    dev_unique_wav_count_by_cluster=list(wav_support),
                    dev_unique_bank_count_by_cluster=list(bank_support),
                    context_nmi=round(context_nmi,4), source_template_nmi=round(template_nmi,4),
                    bank_nmi=round(bank_nmi,4),
                    context_confound_warning=bool(context_nmi>=0.5 or template_nmi>=0.5),
                    bank_confound_warning=bool(bank_nmi>=0.25),
                    underreplicated_heldout_warning=base['heldout_template_count']<2,
                    geometry_screen_passed=bool(geometric_signal),
                    subtype_count_hypothesis=2, hypothetical_split_not_authorized=True,
                    full_centroids_fit_dev_only=True)
        summaries.append(base)
        for subset,subset_ids,assignment in [('development',train_ids,y_train),('frozen_heldout',test_ids,y_test)]:
            for j,k in zip(subset_ids,assignment):
                r=rows[j]
                memberships.append({'bank':r['bank'], 'source_wav':r['source_wav'],
                                    'source_pcm_sha256':r['source_pcm_sha256'],
                                    'source_oto_line_development_reference':r.get('oto_line',''),
                                    'symbol':symbol, 'context_for_diagnostics_only':r['context'],
                                    'data_split':subset,'context_pattern_candidate_id':f'{symbol}__ctxcluster_{int(k)}',
                                    'status':'cluster_on_oto_derived_context_NOT_verified_phone'})
    counts=Counter(r['status'] for r in summaries)
    report={'schema':SCHEMA,'feature_mode':FEATURE_MODE,'feature_dim':36,
            'source':'v3 frozen OTO-derived late-context acoustics',
            'embedding_source_is_unanchored':False,'ipa_used':False,
            'frozen_holdout_templates':sorted(FROZEN_HELDOUT),
            'heldout_used_to_fit_centroids':False,
            'heldout_inspected_during_iterative_development':True,
            'heldout_is_independent_confirmatory_test':False,
            'bank_normalization':normalization,
            'recordings':len(set((r['bank'],r['source_wav']) for r in rows)),
            'source_wav_templates':len(set(r['source_wav'] for r in rows)),
            'bank_count':len(banks),'different_speakers_established':False,
            'sample_count':len(rows),'label_count':len(summaries),
            'status_counts':dict(counts),
            'interpretation':'Hypotheses about repeating spectral CONTEXT only; no verified phoneme/subtype, true boundary, or split/merge.',
            'no_auto_aliases_or_inventory_edits':True,'no_phone_boundary_accuracy':True,
            'read_only_acoustic_hypothesis_exploration':True}
    return summaries,memberships,report


def run(v3: str | Path, output: str | Path) -> dict:
    output=Path(output)
    if output.exists():raise FileExistsError('refusing to overwrite existing experiment')
    rows,X,digest=read_frozen_v3(v3)
    result,memberships,report=discover(rows,X)
    report['input_v3_zip_sha256']=digest
    output.mkdir(parents=True)
    for name,records in [('subtype_diagnostics.csv',result),('sample_membership.csv',memberships)]:
        columns=list(dict.fromkeys(k for rec in records for k in rec))
        with (output/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns,extrasaction='raise');w.writeheader()
            for rec in records:
                w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in rec.items()})
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(prog='python -m phonoweave.native_acoustic_subtypes_v9')
    parser.add_argument('--v3',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(run(args.v3,args.output),ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
