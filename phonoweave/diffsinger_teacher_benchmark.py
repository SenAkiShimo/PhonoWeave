"""Frozen independent DiffSinger teacher intake and matched K=2 diagnostics.

No synthesis model is run here. Without fresh, externally attested teacher
waveforms, no K=2 reference-comparison scores may be produced. Provenance is
validated as a submitted *declaration*, never asserted to be machine-proven.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import wave

import numpy as np

SCHEMA = 'phonoweave.diffsinger_teacher_intake.v1'
REPORT_SCHEMA = 'phonoweave.diffsinger_teacher_matched_k2_diagnostics.v1'
RENDER_SCHEMA = 'phonoweave.diffsinger_note_phrase_ab.v1'
EXPECTED_POLICIES = ('k2_keep_i_ir', 'k2_keep_i_i0', 'k2_keep_ir_i0')
HIGH_BUDGET = 'k3_preserve_all_not_budget_matched'
TEACHER_SUBDIR = 'teacher'
SHA64 = re.compile(r'^[a-f0-9]{64}$')


def _sha_file(path: Path) -> str:
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _safe_child(root: Path, rel: str, *, must_exist: bool = True) -> Path:
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or '..' in Path(rel).parts or '\\' in rel:
        raise ValueError('invalid relative artifact path')
    root = root.resolve()
    path = (root / rel).resolve(strict=must_exist)
    if not path.is_relative_to(root):
        raise ValueError('artifact escapes its immutable source directory')
    return path


def _canonical_hash(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
       separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _load_renders(root: Path) -> tuple[dict, str, dict[tuple[str, str], dict[str, dict]]]:
    root = root.expanduser().resolve(strict=True)
    manifest = root / 'render_manifest.json'
    raw = manifest.read_bytes()
    data = json.loads(raw)
    if data.get('schema') != RENDER_SCHEMA:
        raise ValueError('incompatible note-phrase render manifest')
    if data.get('experiment_gates', {}).get('independent_DiffSinger_teacher_reference_present') is not False:
        raise ValueError('expected source-only preview manifest, not pre-scored teacher')
    frozen = data.get('frozen_score')
    if not isinstance(frozen, dict) or _canonical_hash(frozen) != data.get('score_sha256'):
        raise ValueError('frozen note-score checksum mismatch')
    if [p.get('id') for p in frozen.get('phrases', [])] != ['phrase_a', 'phrase_b']:
        raise ValueError('expected exactly two frozen phrases')
    groups: dict[tuple[str, str], dict[str, dict]] = {}
    seen_wavs: set[str] = set()
    for row in data.get('renders', []):
        bank, phrase, policy = row.get('bank'), row.get('score_phrase'), row.get('policy')
        if not isinstance(bank, str) or not re.fullmatch(r'(mang|lu)_[A-G][b#]?[0-8]', bank):
            raise ValueError('invalid render bank id')
        if phrase not in ('phrase_a', 'phrase_b') or policy not in (*EXPECTED_POLICIES, HIGH_BUDGET):
            raise ValueError('unexpected phrase or policy')
        if row.get('score_sha256') != data['score_sha256']:
            raise ValueError('render note score does not match immutable score')
        wav = row.get('rendered_wav')
        if wav in seen_wavs:
            raise ValueError('duplicate output WAV pathname')
        seen_wavs.add(wav)
        expected = f'{bank}/{policy}/{phrase}.wav'
        if wav != expected:
            raise ValueError('noncanonical output WAV location')
        path = _safe_child(root, wav)
        if _sha_file(path) != row.get('wav_sha256'):
            raise ValueError('rendered WAV source integrity mismatch')
        if row.get('test_target_was_independently_generated') is not False or row.get('teacher_quality_scored') is not False:
            raise ValueError('render falsely claims to be independent teacher')
        key = bank, phrase
        if policy in groups.setdefault(key, {}):
            raise ValueError('duplicate render policy for matched target')
        groups[key][policy] = row
    if len(groups) != 14 or any(set(policies) != {*EXPECTED_POLICIES, HIGH_BUDGET}
                                for policies in groups.values()):
        raise ValueError('frozen study requires 7 banks × 2 phrases × 4 arms')
    budgets = {(x['bank'], x['policy']): x['total_deduplicated_source_ms']
               for x in data.get('budget_ledger', [])}
    for bank, _ in groups:
        values = {budgets.get((bank, policy)) for policy in EXPECTED_POLICIES}
        if values != {2140} or budgets.get((bank, HIGH_BUDGET)) != 2380:
            raise ValueError('K=2 identical source-unit budget or higher-cost reference corrupted')
    return data, sha256(raw).hexdigest(), groups


def prepare_intake(render_root: Path, output_file: Path) -> dict:
    """Create immutable 14-slot reference request; no reference audio fabricated."""
    data, render_sha, groups = _load_renders(render_root)
    items = []
    for bank, phrase in sorted(groups):
        midi_path = f'score_midi/{bank}_{phrase}.mid'
        _safe_child(render_root, midi_path)
        items.append({
            'bank': bank, 'phrase': phrase,
            'relative_wav': f'{TEACHER_SUBDIR}/{bank}/{phrase}.wav',
            'midi_relative_path': midi_path,
            'midi_sha256': _sha_file(_safe_child(render_root, midi_path)),
            'wav_sha256': None,
            'model_or_checkpoint_identifier': None,
            'generator': 'DiffSinger',
            'new_generation_from_frozen_score': False,
            'source_wav_reused_or_spliced': None,
            'style_verified_by_operator': False,
            'note_and_lyric_sequence_verified': False,
            'onset_alignment_verified': False,
            'onset_shift_ms': 0,
            'settings_or_project_reference': None,
        })
    obj = {
        'schema': SCHEMA,
        'render_manifest_sha256': render_sha,
        'score_sha256': data['score_sha256'],
        'score_id': data['frozen_score']['score_id'],
        'reference_status': 'unprovided_no_teacher_scores',
        'study_design': 'DiffSinger_generated_inventory_to_concatenative_reconstruction',
        'teacher_reference_slots': items,
        'scientific_limits': [
            'Teacher WAV provenance is an operator attestation; hashes cannot prove model origin.',
            'MIDI pitch/lyrics alone do not freeze DiffSinger phoneme duration or all inference settings.',
            'These six VC targets were seen as units in the inventory; this is sequence reconstruction, not unseen-phone generalization.',
            'All comparisons share a Rubber Band preview renderer, not a validated OpenUtau engine.',
            'Without independent teacher WAVs there are zero measured reconstruction benefits.',
        ],
    }
    output_file = output_file.expanduser().resolve()
    if output_file == render_root.resolve() or output_file.is_relative_to(render_root.resolve()):
        raise ValueError('intake cannot overwrite or live inside source renders')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return obj


def scan_new_teacher_wavs(receipt: Path, reference_root: Path, output_file: Path) -> dict:
    """Populate WAV SHA-256 only; NEVER auto-attest fresh DiffSinger origin."""
    receipt = receipt.expanduser().resolve(strict=True)
    reference_root = reference_root.expanduser().resolve(strict=True)
    data = json.loads(receipt.read_text(encoding='utf-8'))
    if data.get('schema') != SCHEMA or not isinstance(data.get('teacher_reference_slots'),list):
        raise ValueError('unrecognized teacher reference intake')
    import copy
    updated = copy.deepcopy(data)
    found = 0
    seen = set()
    for row in updated['teacher_reference_slots']:
        bank, phrase = row.get('bank'), row.get('phrase')
        if (not isinstance(bank,str) or not isinstance(phrase,str)
            or not re.fullmatch(r'(mang|lu)_[A-G][b#]?[0-8]',bank)
            or phrase not in ('phrase_a','phrase_b')):
            raise ValueError('invalid teacher reference identity')
        key = bank,phrase
        if key in seen:
            raise ValueError('duplicate teacher reference identity')
        seen.add(key)
        rel = f'{TEACHER_SUBDIR}/{bank}/{phrase}.wav'
        if row.get('relative_wav') != rel:
            raise ValueError('teacher relative WAV path changed')
        file = _safe_child(reference_root,rel,must_exist=False)
        if file.is_file():
            _read_pcm(file)  # format check only, not an origin attestation
            row['wav_sha256'] = _sha_file(file)
            found += 1
    if len(seen) != 14:
        raise ValueError('frozen teacher intake must contain all fourteen slots')
    updated['reference_status'] = 'file_hashes_scanned_operator_attestation_still_required'
    updated['scanned_wav_slots'] = found
    output_file = output_file.expanduser().resolve()
    if output_file == receipt or output_file.suffix.lower() != '.json':
        raise ValueError('scan output must be a new JSON file without overwriting the input receipt')
    output_file.parent.mkdir(parents=True,exist_ok=True)
    with output_file.open('x',encoding='utf-8') as f:
        json.dump(updated,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    return updated


def _read_pcm(path: Path) -> tuple[np.ndarray, int, float, str]:
    try:
        with wave.open(str(path), 'rb') as w:
            if w.getcomptype() != 'NONE' or w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise ValueError('reference and preview WAVs must be mono PCM16')
            sr = w.getframerate()
            if sr not in (44100, 48000, 96000):
                raise ValueError('unsupported WAV sample rate')
            n = w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError, OSError) as e:
        raise ValueError('unreadable WAV') from e
    if len(raw) != n * 2:
        raise ValueError('truncated WAV audio')
    a = np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768.0
    return a, sr, n / sr * 1000, sha256(raw).hexdigest()


def _trailing_release_compatible(teacher: np.ndarray, rate: int, *, expected_ms: float, observed_ms: float) -> bool:
    """Allow a small *decaying final release*, not arbitrary extra note time.

    This only changes acceptance of 100-150 ms positive overhang. Comparison
    remains pinned to the first frozen-preview frames; the trailing release is
    never an extra training source or a policy-specific alignment parameter.
    """
    extra = observed_ms - expected_ms
    if not 100 < extra <= 150:
        return False
    first = round(expected_ms * rate / 1000)
    n = round(.020 * rate)
    if first < 0 or len(teacher) - first < 3 * n:
        return False
    begin = float(np.sqrt(np.mean(teacher[first:first+n] ** 2)))
    end = float(np.sqrt(np.mean(teacher[-n:] ** 2)))
    return begin > 1e-5 and end < .005 and end < .25 * begin


def _compare_spectra(teacher: np.ndarray, arm: np.ndarray, rate: int) -> dict:
    """Coarse fixed-time spectral diagnostics, not a singing quality metric.

    No per-arm time warping, independent gain matching or best-shift fitting.
    Strictly compare the same teacher frames against frozen rendered-note times.
    """
    frame = round(rate * .032)
    hop = round(rate * .010)
    count = min(len(teacher), len(arm))
    if count < frame * 4:
        raise ValueError('too-short aligned phrase')
    starts = np.arange(0, count-frame+1, hop)
    hann = np.hanning(frame)
    x = teacher[starts[:, None] + np.arange(frame)]
    y = arm[starts[:, None] + np.arange(frame)]
    rx = np.sqrt(np.mean(x*x, axis=1))
    ry = np.sqrt(np.mean(y*y, axis=1))
    active = (20*np.log10(np.maximum(rx, 1e-10))) > -48
    if np.sum(active) < 40:
        raise ValueError('teacher reference contains too little usable voiced signal')
    fx = np.abs(np.fft.rfft((x-x.mean(axis=1,keepdims=True))*hann,axis=1))**2
    fy = np.abs(np.fft.rfft((y-y.mean(axis=1,keepdims=True))*hann,axis=1))**2
    freq = np.fft.rfftfreq(frame, 1/rate)
    edges = np.geomspace(150, 8000, 13)
    bands_x=[];bands_y=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(freq>=lo)&(freq<hi)
        bands_x.append(np.log10(np.mean(fx[:,mask],axis=1)+1e-10))
        bands_y.append(np.log10(np.mean(fy[:,mask],axis=1)+1e-10))
    sx=np.stack(bands_x,axis=1)
    sy=np.stack(bands_y,axis=1)
    sx-=sx.mean(axis=1,keepdims=True)
    sy-=sy.mean(axis=1,keepdims=True)
    diff=np.sqrt(np.mean((sx[active]-sy[active])**2,axis=1))
    loudness=np.abs(20*np.log10(np.maximum(rx[active],1e-10))-
                    20*np.log10(np.maximum(ry[active],1e-10)))
    return {
       'spectral_log10_shape_rmse':round(float(np.mean(diff)),6),
       'frame_rms_absolute_difference_db':round(float(np.mean(loudness)),4),
       'active_teacher_frames':int(np.sum(active)),
       'metric_scope':'framewise_spectral_shape_and_loudness_diagnostic_not_perceptual_quality',
    }


def audit_and_compare(render_root: Path, receipt: Path, *,
                      reference_root: Path | None = None) -> dict:
    """Fail closed per reference slot and optionally compare verified K2 arms.

    A teacher in an existing preview WAV is always rejected. Incomplete slots
    remain blocked and do not create fabricated quality metrics.
    """
    data, render_sha, groups = _load_renders(render_root)
    receipt = receipt.expanduser().resolve(strict=True)
    obj = json.loads(receipt.read_text(encoding='utf-8'))
    if obj.get('schema') != SCHEMA or obj.get('render_manifest_sha256') != render_sha or obj.get('score_sha256') != data['score_sha256']:
        raise ValueError('teacher receipt is not bound to these frozen renders and score')
    rows=obj.get('teacher_reference_slots')
    if not isinstance(rows,list) or len(rows)!=len(groups):
        raise ValueError('missing or duplicated reference slots')
    root = reference_root.expanduser().resolve(strict=True) if reference_root else receipt.parent
    all_render_hashes={r['wav_sha256'] for pols in groups.values() for r in pols.values()}
    seen=set();seen_pcm=set()
    verified=[];provisional=[];blocked=[];diagnostics=[]
    for row in rows:
        key=row.get('bank'),row.get('phrase')
        if key not in groups or key in seen:
            raise ValueError('unexpected or duplicate reference bank/phrase')
        seen.add(key)
        bank,phrase=key
        ref_rel=f'{TEACHER_SUBDIR}/{bank}/{phrase}.wav'
        if row.get('relative_wav')!=ref_rel or row.get('generator')!='DiffSinger':
            raise ValueError('teacher WAV path or generator was altered')
        midi_rel=f'score_midi/{bank}_{phrase}.mid'
        if row.get('midi_relative_path')!=midi_rel or row.get('midi_sha256')!=_sha_file(_safe_child(render_root,midi_rel)):
            raise ValueError('teacher request MIDI identity does not match frozen score')
        candidate=_safe_child(root,ref_rel,must_exist=False)
        want=row.get('wav_sha256')
        if not candidate.is_file():
            blocked.append({'bank':bank,'phrase':phrase,'reason':'teacher_wav_missing'})
            continue
        teacher_hash=_sha_file(candidate)
        if not isinstance(want,str) or not SHA64.fullmatch(want) or want!=teacher_hash:
            blocked.append({'bank':bank,'phrase':phrase,'reason':'teacher_hash_missing_or_mismatch'})
            continue
        if teacher_hash in all_render_hashes:
            blocked.append({'bank':bank,'phrase':phrase,'reason':'teacher_is_existing_concatenative_preview'})
            continue
        teacher, rate, duration, pcm_sha=_read_pcm(candidate)
        if pcm_sha in seen_pcm:
            blocked.append({'bank':bank,'phrase':phrase,'reason':'identical_teacher_pcm_in_multiple_conditions'})
            continue
        seen_pcm.add(pcm_sha)
        operator_flags = (row.get('new_generation_from_frozen_score') is True
            and row.get('source_wav_reused_or_spliced') is False
            and row.get('style_verified_by_operator') is True
            and row.get('note_and_lyric_sequence_verified') is True
            and row.get('onset_alignment_verified') is True)
        reproducible_metadata = (isinstance(row.get('model_or_checkpoint_identifier'),str)
            and bool(row['model_or_checkpoint_identifier'].strip())
            and isinstance(row.get('settings_or_project_reference'),str)
            and bool(row['settings_or_project_reference'].strip()))
        attestation=row.get('author_attestation')
        explicitly_partial = (row.get('author_attested_partial_provenance') is True
            and row.get('model_or_checkpoint_identifier') is None
            and row.get('settings_or_project_reference') is None
            and row.get('unrecorded_technical_provenance') ==
                ['exact_model_checkpoint','inference_settings_or_editor_project']
            and isinstance(attestation,dict)
            and attestation.get('source')=='author_confirmed_in_conversation'
            and attestation.get('attestation_is_cryptographic_proof') is False
            and isinstance(attestation.get('assertion'),str)
            and bool(attestation['assertion'].strip()))
        if not operator_flags or not (reproducible_metadata or explicitly_partial):
            blocked.append({'bank':bank,'phrase':phrase,'reason':'fresh_render_identity_or_alignment_unattested'})
            continue
        onset=row.get('onset_shift_ms')
        if isinstance(onset,bool) or not isinstance(onset,(int,float)) or not np.isfinite(onset) or abs(onset)>150:
            blocked.append({'bank':bank,'phrase':phrase,'reason':'invalid_onset_shift'})
            continue
        actual=groups[key]
        sounds={}
        for policy in EXPECTED_POLICIES:
            a, ar, ad, apcm = _read_pcm(_safe_child(render_root,actual[policy]['rendered_wav']))
            sounds[policy]=(a,ar,ad,apcm)
        if len({(ar,round(ad,2)) for _,ar,ad,_ in sounds.values()})!=1:
            raise ValueError('rendered arms have inconsistent WAV formats/timing')
        if rate!=sounds[EXPECTED_POLICIES[0]][1]:
            blocked.append({'bank':bank,'phrase':phrase,'reason':'teacher_sample_rate_mismatch_resample_externally'})
            continue
        expected_duration=sounds[EXPECTED_POLICIES[0]][2]
        if (abs(duration-expected_duration)>100
            and not _trailing_release_compatible(
                teacher, rate, expected_ms=expected_duration, observed_ms=duration
            )):
            blocked.append({'bank':bank,'phrase':phrase,'reason':'teacher_score_duration_mismatch_align_externally'})
            continue
        nshift=round(onset*rate/1000)
        # Shift teacher only once from a human-reviewed score offset, NEVER fit
        # each candidate arm independently to improve its own measurement.
        if nshift>0:
            shifted=np.pad(teacher,(0,nshift))[nshift:]
        elif nshift<0:
            shifted=np.pad(teacher,(-nshift,0))[:len(teacher)]
        else:
            shifted=teacher
        tier=('reproducible_metadata_declared' if reproducible_metadata
              else 'author_attested_model_version_and_settings_unrecorded')
        for policy in EXPECTED_POLICIES:
            scores=_compare_spectra(shifted,sounds[policy][0],rate)
            diagnostics.append({'bank':bank,'phrase':phrase,'policy':policy,
                                'source_budget_ms':2140,'reference_provenance_tier':tier,
                                **scores})
        record={'bank':bank,'phrase':phrase,'reference_wav_sha256':teacher_hash,
                'reference_duration_ms':round(duration,2),
                'render_reference_role':'diffsinger_teacher_operator_attested_not_machine_proven',
                'reference_provenance_tier':tier}
        (verified if reproducible_metadata else provisional).append(record)
    if seen!=set(groups):
        raise ValueError('teacher receipt omitted frozen reference slots')
    return {
      'schema':REPORT_SCHEMA,
      'status':('partial_descriptive_reference_diagnostics_provenance_incomplete'
        if provisional else ('partial_descriptive_reference_diagnostics_not_efficacy'
        if verified else 'blocked_no_independent_reference_comparisons')),
      'score_sha256':data['score_sha256'],
      'render_manifest_sha256':render_sha,
      'expected_teacher_slots':len(groups),
      'reference_verified_slots':len(verified),
      'reference_provisional_author_attested_slots':len(provisional),
      'reference_diagnostic_eligible_slots':len(verified)+len(provisional),
      'complete_teacher_set':len(verified)==len(groups),
      'verified':verified,'provisional':provisional,
      'blocked':blocked,'diagnostics':diagnostics,
      'rankings_are_perceptual_quality_claims':False,
      'independent_source_generation_machine_proven':False,
      'split_share_efficacy_established':False,
      'K3_in_equal_budget_comparisons':False,
      'limitations':[
        'Teacher generation is operator-attested, not cryptographically proven from audio.',
        'Partial provenance permits only explicitly labeled descriptive diagnostics; absent model checkpoint/settings bars independently reproducible evaluation.',
        'No unseen-phoneme or cross-speaker generalization.',
        'Framewise log-spectral divergence is sensitive to timing and formant differences; not an intelligibility or listener-preference score.',
        'The Rubber Band preview engine is not a full OpenUtau singing renderer.',
        'These are matched-budget comparisons only between predeclared K=2 policies.',
      ]
    }


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(prog='phonoweave diffsinger-teacher')
    sub=parser.add_subparsers(dest='cmd',required=True)
    prep=sub.add_parser('prepare',help='prepare 14 unfilled DiffSinger teacher slots')
    prep.add_argument('--renders',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True)
    scan=sub.add_parser('scan',help='read and hash available teacher WAVs, without automatic origin approval')
    scan.add_argument('--receipt',type=Path,required=True)
    scan.add_argument('--reference-root',type=Path,required=True)
    scan.add_argument('--output',type=Path,required=True)
    audit=sub.add_parser('audit',help='check frozen teacher waveforms and compare usable matched K2 arms')
    audit.add_argument('--renders',type=Path,required=True)
    audit.add_argument('--receipt',type=Path,required=True)
    audit.add_argument('--reference-root',type=Path)
    audit.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.cmd=='prepare':
            obj=prepare_intake(args.renders,args.output)
            print(f"Prepared {len(obj['teacher_reference_slots'])} slots (no teacher WAVs fabricated).")
        elif args.cmd=='scan':
            scanned=scan_new_teacher_wavs(args.receipt,args.reference_root,args.output)
            print(f"Hashed {scanned['scanned_wav_slots']}/14 teacher WAVs; origin declarations remain unconfirmed.")
        else:
            report=audit_and_compare(args.renders,args.receipt,reference_root=args.reference_root)
            output=args.output.expanduser().resolve()
            root=args.renders.resolve()
            if output.is_relative_to(root) or output==args.receipt.resolve() or (args.reference_root and output.is_relative_to(args.reference_root.resolve())):
                raise ValueError('audit cannot overwrite source/teacher directories')
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.open('x',encoding='utf-8') as f:
                json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
                f.write('\n')
            print(f"Teacher slots with technical metadata: {report['reference_verified_slots']}/{report['expected_teacher_slots']}")
            print(f"Author-attested, metadata-incomplete slots: {report['reference_provisional_author_attested_slots']}")
            print(f"Status: {report['status']}; evidence for quality gain: NOT established")
    except (OSError,ValueError,KeyError,TypeError,OverflowError) as exc:
        parser.error(str(exc))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
