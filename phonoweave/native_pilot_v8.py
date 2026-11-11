"""Real-corpus pilot preflight. Does NOT infer phone boundaries or recording scripts.

Uses frozen v3 heldout metadata, v4 source inventories and v5 candidate provenance.
The script file and manually annotated boundaries are external and never inferred
from OTO aliases, v5 guesses, or source WAV filenames.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import wave
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = 'phonoweave.real_corpus_pilot.v0.8'
SEQUENCE_SCHEMA = 'phonoweave.native_sequences.v1'
PILOT_BANKS = ('mang_D4', 'lu_C4')
PILOT_TEMPLATES = ('e0.wav', 'en0.wav')
SMOKE_TEMPLATES = ('additional.wav',)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _csv(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def _safe_key(bank: str, wav: str) -> tuple[str, str]:
    if not re.fullmatch(r'[A-Za-z0-9_]+', bank) or not re.fullmatch(r'[A-Za-z0-9_.-]+\.wav', wav):
        raise ValueError('unsafe source key')
    return bank, wav


def _sha64(s: str) -> str:
    if not isinstance(s, str) or not re.fullmatch('[0-9a-fA-F]{64}', s):
        raise ValueError('invalid SHA256 value')
    return s.lower()


def _read_sources(v4: Path):
    sources = {}
    summaries = {}
    with zipfile.ZipFile(v4) as z:
        names = set(z.namelist())
        for name in sorted(n for n in names if re.fullmatch(r'results/[^/]+/sources\.csv', n)):
            bank = name.split('/')[1]
            summary_name = f'results/{bank}/summary.json'
            if summary_name not in names:
                raise ValueError('missing source summary ' + bank)
            summaries[bank] = json.loads(z.read(summary_name))
            for row in _csv(z.read(name)):
                key = _safe_key(bank, row['source_wav'])
                if key in sources:
                    raise ValueError('duplicate v4 source')
                sources[key] = {'source_pcm_sha256': _sha64(row['source_pcm_sha256']),
                                'duration_ms': float(row['duration_ms']),
                                'internal_events': int(row['internal_events'])}
    if not sources:
        raise ValueError('no v4 sources found')
    return sources, summaries


def _read_v5(v5: Path):
    with zipfile.ZipFile(v5) as z:
        rows = _csv(z.read('results/candidate_segment_unit_hints.csv'))
    groups = defaultdict(list)
    hashes = {}
    for row in rows:
        key = _safe_key(row['bank'], row['source_wav'])
        digest = _sha64(row['source_pcm_sha256'])
        if key in hashes and hashes[key] != digest:
            raise ValueError('inconsistent v5 PCM hashes')
        hashes[key] = digest
        groups[key].append(row)
    if not rows:
        raise ValueError('v5 candidate data is empty')
    return groups, hashes


def _read_v2_local_hints(v2: Path):
    """Read only archive leaf names + inside-ZIP roots; NEVER export local paths."""
    with zipfile.ZipFile(v2) as z:
        data = json.loads(z.read('runs/lynneqing_bulk_manifest_local_example.json'))
    hints = {}
    for x in data['banks']:
        bank = x['style'] + '_' + x['pitch']
        hints[bank] = {'archive_basename_hint': Path(x['archive']).name, 'root': x['root']}
    return hints


def prepare(v2: Path, v3: Path, v4: Path, v5: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError('destination exists; frozen pilot cannot be overwritten')
    sources, summaries = _read_sources(v4)
    groups, hashes = _read_v5(v5)
    if set(sources) != set(groups):
        raise ValueError('v4/v5 source-key sets differ')
    if any(sources[k]['source_pcm_sha256'] != hashes[k] for k in sources):
        raise ValueError('v4/v5 PCM hashes differ')
    with zipfile.ZipFile(v3) as z:
        heldout = set(json.loads(z.read('results/evaluation.json'))['frozen_test_groups'])
    if not set(PILOT_TEMPLATES) <= heldout:
        raise ValueError('pilot templates were not frozen v3 heldout groups')
    if set(SMOKE_TEMPLATES) & heldout:
        raise ValueError('smoke control must be a development template')
    hints = _read_v2_local_hints(v2)
    entries = []
    for bank in PILOT_BANKS:
        if bank not in summaries or bank not in hints:
            raise ValueError('unknown pilot bank')
        for role, templates in [('retrospective_pilot', PILOT_TEMPLATES),
                                ('development_smoke_only', SMOKE_TEMPLATES)]:
            for wav in templates:
                k = (bank, wav)
                if k not in sources:
                    raise ValueError('pilot source missing: ' + str(k))
                c = groups[k]
                row = sources[k]
                entries.append({
                    'bank': bank, 'source_wav': wav,
                    'source_pcm_sha256': row['source_pcm_sha256'],
                    'duration_ms': row['duration_ms'],
                    'v4_internal_events': row['internal_events'],
                    'v5_candidate_regions': len(c),
                    'v5_hint_regions': sum(x['status'] == 'candidate_UNVERIFIED_not_a_phone' for x in c),
                    'cohort_role': role,
                    'evidence_status': 'NO_INDEPENDENT_SCRIPT_OR_GOLD',
                    'source_archive_sha256': _sha64(summaries[bank]['archive_sha256']),
                })
    cohort = {
        'schema': SCHEMA, 'status': 'preflight_only_not_phone_validation',
        'method': 'Fixed two v3-heldout templates in each of two banks, plus one development smoke template per bank',
        'important_limitations': [
            'The same speaker may be represented across banks; this is not speaker-independent testing.',
            'v5 already ran OTO-based posthoc diagnostics on its v3-heldout templates: this cohort is retrospective, NOT untouched confirmatory data.',
            'The recording script is missing; do not infer order from filename, v5 candidates, OTO aliases, or reference anchors.',
            'No real phone boundaries have been annotated or scored.',
        ],
        'archive_fingerprints': {'v2': _file_sha(v2), 'v3': _file_sha(v3),
                                 'v4': _file_sha(v4), 'v5': _file_sha(v5)},
        'corpus': {'banks': len(summaries), 'source_wavs': len(sources),
                   'v5_candidate_regions': sum(len(v) for v in groups.values()),
                   'v5_hint_regions': sum(x['status'] == 'candidate_UNVERIFIED_not_a_phone'
                                          for g in groups.values() for x in g)},
        'sources': entries,
    }
    raw = (json.dumps(cohort, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    output = {
        'schema': SEQUENCE_SCHEMA, 'WARNING': 'TEMPLATE ONLY: MUST BE COMPLETED FROM AN INDEPENDENT SCRIPT, NOT OTO/V5',
        'entries': [{'bank': x['bank'], 'source_wav': x['source_wav'],
                     'source_pcm_sha256': x['source_pcm_sha256'],
                     'provenance': 'PENDING_INDEPENDENT_RECORDING_SCRIPT', 'units': []}
                    for x in entries if x['cohort_role'] == 'retrospective_pilot'],
    }
    audio = {'banks': {bank: {'archive': 'REPLACE_WITH_LOCAL_ABSOLUTE_PATH',
                             'root': hints[bank]['root'],
                             'archive_basename_hint': hints[bank]['archive_basename_hint']}
                       for bank in PILOT_BANKS}}
    out.mkdir(parents=True)
    (out / 'cohort_frozen.json').write_bytes(raw)
    (out / 'cohort_sha256.txt').write_text(_sha_bytes(raw) + '  cohort_frozen.json\n')
    (out / 'sequences_TEMPLATE.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    (out / 'audio_manifest_TEMPLATE.json').write_text(json.dumps(audio, ensure_ascii=False, indent=2) + '\n')
    report = {
        'verified_v4_v5_sources': len(sources),
        'source_hash_mismatches': 0,
        'pilot_files': sum(x['cohort_role'] == 'retrospective_pilot' for x in entries),
        'pilot_duration_seconds': round(sum(x['duration_ms'] for x in entries if x['cohort_role'] == 'retrospective_pilot') / 1000, 3),
        'development_smoke_files': sum(x['cohort_role'] == 'development_smoke_only' for x in entries),
        'cohort_sha256': _sha_bytes(raw),
        'can_run_v6': False,
        'blocker': 'Complete independent unit order and provide source audio for manual gold',
    }
    (out / 'preflight_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def _read_frozen_cohort(path: Path) -> dict:
    raw = path.read_bytes()
    receipt_path = path.parent / 'cohort_sha256.txt'
    if not receipt_path.is_file():
        raise ValueError('missing frozen cohort SHA-256 receipt')
    digest = receipt_path.read_text(encoding='utf8').split()[0]
    if _sha64(digest) != _sha_bytes(raw):
        raise ValueError('frozen cohort hash differs from receipt')
    cohort = json.loads(raw)
    if cohort.get('schema') != SCHEMA:
        raise ValueError('wrong cohort schema')
    return cohort


def verify_sequences(cohort_path: Path, sequence_path: Path) -> dict:
    cohort = _read_frozen_cohort(cohort_path)
    data = json.loads(sequence_path.read_text(encoding='utf8'))
    if cohort.get('schema') != SCHEMA or data.get('schema') != SEQUENCE_SCHEMA:
        raise ValueError('incorrect schema')
    expected = {(x['bank'], x['source_wav']): x for x in cohort['sources']
                if x['cohort_role'] == 'retrospective_pilot'}
    got = {}
    for x in data['entries']:
        key = _safe_key(x['bank'], x['source_wav'])
        if key in got:
            raise ValueError('duplicate script source')
        got[key] = x
    if set(expected) != set(got):
        raise ValueError('script coverage must exactly match frozen pilot source keys')
    for key in sorted(expected):
        item = got[key]
        if _sha64(item['source_pcm_sha256']) != expected[key]['source_pcm_sha256']:
            raise ValueError('independent script PCM hash mismatch')
        if item.get('provenance') != 'independent_recording_script':
            raise ValueError('independent recording-script provenance is required')
        if not isinstance(item.get('units'), list) or not item['units'] or not all(
                isinstance(x, str) and x.strip() for x in item['units']):
            raise ValueError('provide complete, nonempty ordered user-native symbols')
    return {'verified_sources': len(expected), 'units': sum(len(x['units']) for x in got.values()),
            'status': 'script_schema_passed_not_audio_or_boundary_validation',
            'warning': 'Provenance string is a human assertion; software cannot prove that script was independent.'}


def _pcm_sha(raw: bytes) -> str:
    with wave.open(io.BytesIO(raw), 'rb') as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getcomptype() != 'NONE':
            raise ValueError('voicebank audio must be mono uncompressed PCM16')
        return _sha_bytes(w.readframes(w.getnframes()))


def extract_audio(cohort_path: Path, audio_manifest: Path, out: Path) -> dict:
    """Opt-in, LOCAL ONLY. Avoid publishing private original voicebank audio."""
    if out.exists():
        raise FileExistsError('audio output already exists')
    cohort = _read_frozen_cohort(cohort_path)
    config = json.loads(audio_manifest.read_text(encoding='utf8'))
    if cohort.get('schema') != SCHEMA:
        raise ValueError('wrong cohort schema')
    verified = []
    # Verify *everything* before writing, so mismatched audio cannot leave a partial pack.
    for rec in cohort['sources']:
        bank, wav = _safe_key(rec['bank'], rec['source_wav'])
        setting = config['banks'][bank]
        archive = Path(setting['archive'])
        if _file_sha(archive) != _sha64(rec['source_archive_sha256']):
            raise ValueError('original voicebank ZIP SHA mismatch: ' + bank)
        root = setting['root']
        if not root or root.startswith('/') or '..' in Path(root).parts:
            raise ValueError('unsafe inside-archive root')
        with zipfile.ZipFile(archive) as z:
            raw = z.read(root + wav)
        if _pcm_sha(raw) != rec['source_pcm_sha256']:
            raise ValueError('original PCM SHA mismatch: ' + bank + '/' + wav)
        verified.append((bank, wav, raw))
    out.mkdir(parents=True)
    for bank, wav, raw in verified:
        d = out / bank
        d.mkdir(exist_ok=True)
        (d / wav).write_bytes(raw)
    return {'audio_sources_verified': len(verified), 'destination': str(out),
            'note': 'Keep source WAVs local/private; do not commit original audio to GitHub.'}


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m phonoweave.native_pilot_v8')
    sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    for s in ['v2','v3','v4','v5','output']:
        prep.add_argument('--'+s, type=Path, required=True)
    seq = sub.add_parser('verify-sequences')
    seq.add_argument('--cohort', type=Path, required=True)
    seq.add_argument('--sequences', type=Path, required=True)
    audio = sub.add_parser('extract-audio')
    audio.add_argument('--cohort', type=Path, required=True)
    audio.add_argument('--manifest', type=Path, required=True)
    audio.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if a.action == 'prepare':
        report = prepare(a.v2,a.v3,a.v4,a.v5,a.output)
    elif a.action == 'verify-sequences':
        report = verify_sequences(a.cohort,a.sequences)
    else:
        report = extract_audio(a.cohort,a.manifest,a.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())