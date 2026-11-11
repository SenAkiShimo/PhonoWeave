"""Small research-evidence gate before suggesting *new* recording units.

Does not replace the existing evidence_gap/supplement_plan/diagnostic_selector
coverage workflow. Historical unlabeled acoustic motifs are not phones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

V13 = 'phonoweave.equal_budget_continuation_proxy.v1.3'
V14 = 'phonoweave.waveform_splice_dynamics.v1.4'
SCHEMA = 'phonoweave.native_recording_gate.v1.5'


def _require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError('unusable evidence: ' + why)


def _hash(s: object) -> str:
    _require(isinstance(s, str) and len(s) == 64 and
             all(c in '0123456789abcdef' for c in s.lower()), 'bad SHA256')
    return s.lower()


def _number(obj: dict, key: str) -> float:
    v = obj.get(key)
    _require(type(v) in (int, float) and math.isfinite(v), 'invalid '+key)
    return float(v)


def _gain(obj: dict, baseline: str, alternative: str, gain_name: str | None = None) -> float:
    a, b = _number(obj, baseline), _number(obj, alternative)
    _require(a > 0 and b >= 0, 'invalid losses')
    result = (a - b) / a
    if gain_name is not None:
        _require(abs(result - _number(obj, gain_name)) <= 1e-4,
                 'contradictory reported metric: '+gain_name)
    return round(result, 6)


def assess(v13: dict, v14: dict, *, digest13: str, digest14: str) -> dict:
    """Only assess this retrospective, unverified motif line, not an entire inventory."""
    _require(isinstance(v13, dict) and isinstance(v14, dict), 'reports must be objects')
    _require(v13.get('schema') == V13 and v14.get('schema') == V14, 'wrong schema')
    src = _hash(v13.get('source_archive_sha256'))
    _require(src == _hash(v14.get('v9_archive_sha256')), 'different audio sources')
    _require(v13.get('equal_donor_budget_and_train_only_donors') is True and
             v13.get('development', {}).get('selection_used_audit') is False,
             'v13 donor isolation or leakage')
    _require(v13.get('verified_phone_identity_or_boundary') is False and
             v13.get('automatic_unit_split_merge_authorized') is False and
             v13.get('audit_previously_explored_in_prior_studies_not_confirmatory') is True,
             'v13 unsupported provenance claims')
    _require(v14.get('no_automatic_inventory_split_merge') is True and
             v14.get('not_perceptual_or_singing_synthesis_quality') is True and
             v14.get('audit_previous_iterations_already_seen_retrospective_only') is True,
             'v14 unsupported provenance claims')
    _require(v14.get('audio_source', {}).get('source_zip_sha256_verified') is True and
             v14.get('frozen_rule', {}).get('feature_fit') == 'train_only' and
             v14.get('frozen_rule', {}).get('selection_access_to_target_right') is False,
             'v14 source isolation or leakage')
    g13, g14 = v13.get('source_groups_by_split', {}), v14.get('source_groups_by_partition', {})
    _require(isinstance(g13, dict) and g13 == g14 and
             all(type(g13.get(k)) is int and g13[k] > 0 for k in ('train', 'validation', 'audit')),
             'different group split')
    _require(v13.get('clip_inventory', {}).get('complete_three_stage_clips') == v14.get('complete_clips'),
             'different eligible clip count')
    a13, a14 = v13.get('audit_descriptive', {}), v14.get('audit', {})
    _require(type(a13.get('paired_source_groups')) is int and
             a13['paired_source_groups'] == a14.get('groups') == g13['audit'] and
             type(a13.get('matched_observations')) is int and
             a13['matched_observations'] == a14.get('observations') and
             a13['matched_observations'] > 0, 'audit denominator mismatch')
    _require(a14.get('donor_budget_equal') == v14.get('frozen_rule', {}).get('candidate_budget'),
             'v14 candidate budget mismatch')
    r13 = _gain(a13, 'pooled_group_mean_mse', 'context_group_mean_mse', 'relative_improvement')
    r14 = _gain(a14['trajectory_mse'], 'pooled', 'seam_aware', 'relative_gain_positive_is_better')
    seam = _gain(a14['predicted_seam_cost'], 'pooled', 'seam_aware', 'relative_gain_positive_is_better')
    transient = _gain(a14['splice_max_sample_step_over_rms'], 'pooled', 'seam_aware',
                      'relative_gain_positive_is_better')
    if r13 <= 0 and r14 <= 0:
        finding = 'current_context_split_not_supported'
    else:
        finding = 'mixed_or_exploratory_proxy_only'
    return {
        'schema': SCHEMA,
        'source_archive_sha256': src,
        'input_report_sha256': {'v13': _hash(digest13), 'v14': _hash(digest14)},
        'scope': 'historical_unverified_audio_regions_only',
        'metrics': {'v13_heldout_relative_gain': r13, 'v14_heldout_continuation_gain': r14,
                    'v14_self_optimized_seam_gain': seam,
                    'v14_rendered_transient_gain': transient,
                    'heldout_source_groups': g13['audit'],
                    'heldout_matched_observations': a13['matched_observations']},
        'finding': finding,
        'recording_advice': {
            'new_units_authorized': False,
            'automatic_alias_changes': [],
            'new_reclist_lines': [],
            'existing_voicebank': 'unchanged',
            'separate_existing_coverage_based_supplement_planner': 'unmodified',
            'reason': 'no independently verified phone and synthesis benefit',
            'next_priority': 'use existing recording scheme; do not add context split from this evidence',
        },
        'not_perceptual_test': True,
        'not_confirmatory': True,
    }


def run(v13_path: Path, v14_path: Path, output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError('refusing to overwrite existing report')
    b13, b14 = Path(v13_path).read_bytes(), Path(v14_path).read_bytes()
    report = assess(json.loads(b13), json.loads(b14),
                    digest13=hashlib.sha256(b13).hexdigest(),
                    digest14=hashlib.sha256(b14).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Scope-limited research evidence gate')
    p.add_argument('--v13-summary', type=Path, required=True)
    p.add_argument('--v14-summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    print(json.dumps(run(a.v13_summary, a.v14_summary, a.output), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
