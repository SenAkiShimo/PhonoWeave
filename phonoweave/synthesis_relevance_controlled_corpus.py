"""Pre-registered Mandarin /sh/ pilot capture plan and read-only integrity audit.

This module does not extract OTO, generate sound, estimate policy utility, or
modify captured WAVs. Session isolation is declared and audited as metadata;
it cannot establish speaker-independent inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import wave

SCHEMA = "phonoweave.synthesis_relevance.controlled_sh_corpus.v1"
PAIRS = (("a", "ua"), ("ai", "uai"), ("an", "uan"), ("ang", "uang"))
FINALS = tuple(final for pair in PAIRS for final in pair)
ROLES = (("S01", "train"), ("S02", "heldout"))
REPEATS = 3


def make_plan(*, speaker_id="pilot_01", pitch_subbank="pilot_pitch_01"):
    """48 predeclared targets: 8 finals x 3 takes x 2 independent sessions."""
    sessions = [
        {"session_id": sid, "role": role, "capture_date": None,
         "device_id": None, "room_id": None, "gain_setting": None,
         "target_pitch_hz": None, "setup_reset_confirmed": None}
        for sid, role in ROLES
    ]
    recordings = [
        {"take_id": f"{sid}_sh{final}_{repeat:02d}", "session_id": sid,
         "final": final, "repeat": repeat,
         "relative_wav": f"audio/{sid}/sh{final}_{repeat:02d}.wav",
         "wav_sha256": None, "frication_onset_ms": None,
         "cv_boundary_ms": None}
        for sid, _ in ROLES for final in FINALS
        for repeat in range(1, REPEATS + 1)
    ]
    return {
        "schema": SCHEMA, "study_id": "controlled_sh_pilot_v1",
        "speaker_id": speaker_id, "pitch_subbank": pitch_subbank,
        "base": "sh", "final_pairs": [list(p) for p in PAIRS],
        "sessions": sessions, "recordings": recordings,
        "limitations": ["One speaker and one pitch subbank are pilot-only; "
                        "no speaker-generalization or rendered-quality claim."]
    }


def write_plan(path: Path, **kwargs):
    """Fail rather than overwrite a manifest holding acquired data."""
    path = path.expanduser()
    data = make_plan(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


def _digest(path: Path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _wav_info(path: Path):
    with wave.open(str(path), "rb") as audio:
        if audio.getcomptype() != "NONE" or audio.getnchannels() != 1:
            raise ValueError("expected mono uncompressed PCM WAV")
        if audio.getsampwidth() not in (2, 3):
            raise ValueError("expected PCM16 or PCM24")
        rate = audio.getframerate()
        if rate not in (44100, 48000, 96000):
            raise ValueError("expected 44.1, 48 or 96 kHz")
        duration_ms = audio.getnframes() * 1000.0 / rate
        if not 300 <= duration_ms <= 8000:
            raise ValueError("expected 0.3 to 8.0 seconds")
        return (rate, audio.getsampwidth(), duration_ms)


def audit_manifest(data: dict, root: Path):
    """Audit an existing plan without silently imputing missing takes.

    Analysis-ready requires all 48 distinct audio files, hashes, declared
    independent sessions and onset/CV annotations. It is NOT an outcome test.
    """
    errors, missing, unverified, unannotated = [], [], [], []
    if data.get("schema") != SCHEMA or data.get("base") != "sh":
        errors.append("schema_or_base_mismatch")
    if data.get("final_pairs") != [list(p) for p in PAIRS]:
        errors.append("predeclared_pair_set_modified")
    if not data.get("speaker_id") or not data.get("pitch_subbank"):
        errors.append("missing_speaker_or_pitch_identifier")
    sessions = data.get("sessions", [])
    expected_roles = dict(ROLES)
    if not isinstance(sessions, list) or len(sessions) != 2 or {
        x.get("session_id"): x.get("role") for x in sessions if isinstance(x, dict)
    } != expected_roles:
        errors.append("session_role_design_modified")
        sessions = []
    dates = []
    for session in sessions:
        sid = session["session_id"]
        for key in ("capture_date", "device_id", "room_id",
                    "gain_setting", "target_pitch_hz"):
            value = session.get(key)
            if value is None or value == "":
                unverified.append(f"{sid}:{key}")
        if session.get("setup_reset_confirmed") is not True:
            unverified.append(f"{sid}:setup_reset_confirmed")
        try:
            dates.append(date.fromisoformat(str(session["capture_date"])))
        except (TypeError, ValueError):
            if session.get("capture_date") is not None:
                errors.append(f"{sid}:invalid_capture_date")
        pitch = session.get("target_pitch_hz")
        if pitch is not None and (not isinstance(pitch, (int, float))
                                  or not 60 <= pitch <= 1000):
            errors.append(f"{sid}:invalid_pitch")
    if len(dates) == 2 and dates[0] == dates[1]:
        errors.append("sessions_must_use_distinct_capture_dates")

    items = data.get("recordings", [])
    if not isinstance(items, list):
        errors.append("recordings_not_a_list")
        items = []
    expected = {(sid, final, i) for sid, _ in ROLES
                for final in FINALS for i in range(1, REPEATS + 1)}
    observed, ids, paths, fingerprints = set(), set(), set(), set()
    counts = Counter()
    formats = set()
    root = root.expanduser().resolve()
    for item in items:
        if not isinstance(item, dict):
            errors.append("non_object_recording")
            continue
        sid, final, repeat = (item.get("session_id"),
                              item.get("final"), item.get("repeat"))
        key = (sid, final, repeat)
        if key not in expected or key in observed:
            errors.append(f"unexpected_or_duplicate_recording:{key}")
        observed.add(key)
        take_id = item.get("take_id")
        if not isinstance(take_id, str) or not take_id or take_id in ids:
            errors.append(f"invalid_or_duplicate_take_id:{take_id}")
        ids.add(take_id)
        wav = item.get("relative_wav")
        if not isinstance(wav, str) or not wav:
            errors.append(f"{take_id}:missing_relative_wav")
            continue
        rel = Path(wav)
        if rel.is_absolute() or ".." in rel.parts or wav in paths:
            errors.append(f"{take_id}:invalid_or_duplicate_relative_wav")
            continue
        paths.add(wav)
        candidate = (root / rel).resolve()
        if not candidate.is_relative_to(root):
            errors.append(f"{take_id}:wav_outside_root")
            continue
        if not candidate.is_file():
            missing.append(take_id)
            continue
        try:
            audio_format = _wav_info(candidate)
            digest = _digest(candidate)
        except (OSError, EOFError, wave.Error, ValueError) as exc:
            errors.append(f"{take_id}:invalid_audio:{exc}")
            continue
        formats.add(audio_format[:2])
        if digest in fingerprints:
            errors.append(f"{take_id}:duplicate_pcm_file_hash")
        fingerprints.add(digest)
        recorded = item.get("wav_sha256")
        if recorded is None or recorded == "":
            unverified.append(f"{take_id}:wav_sha256")
        elif recorded != digest:
            errors.append(f"{take_id}:sha256_mismatch")
        counts[(sid, final)] += 1
        a, b = item.get("frication_onset_ms"), item.get("cv_boundary_ms")
        if a is None or b is None:
            unannotated.append(take_id)
        elif not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                     for x in (a, b)) or not 0 <= a < b <= audio_format[2]:
            errors.append(f"{take_id}:invalid_boundaries")
    if observed != expected:
        errors.append(f"recording_grid_mismatch:missing={len(expected-observed)}")
    if len(formats) > 1:
        errors.append("inconsistent_wav_format")
    for sid, _ in ROLES:
        for final in FINALS:
            if counts[(sid, final)] < REPEATS:
                missing.extend([f"{sid}:sh{final}:incomplete"
                                for _ in range(REPEATS - counts[(sid, final)])])
    missing = sorted(set(missing))
    unverified = sorted(set(unverified))
    unannotated = sorted(set(unannotated))
    capture_complete = not (errors or missing or unverified)
    annotation_complete = not unannotated and capture_complete
    return {
        "schema": "phonoweave.synthesis_relevance.controlled_sh_audit.v1",
        "planned_takes": len(expected),
        "verified_wav_files": len(fingerprints),
        "per_session_final_verified": {
            sid: {f: counts[(sid, f)] for f in FINALS} for sid, _ in ROLES
        },
        "capture_complete": capture_complete,
        "analysis_ready": annotation_complete,
        "errors": errors, "missing": missing,
        "unverified": unverified, "unannotated": unannotated,
        "outcome_status": "not_evaluated",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plan and audit independent /sh/ pilot WAVs")
    actions = parser.add_subparsers(dest="action", required=True)
    plan = actions.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--speaker-id", default="pilot_01")
    plan.add_argument("--pitch-subbank", default="pilot_pitch_01")
    audit = actions.add_parser("audit")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--root", type=Path)
    audit.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.action == "plan":
        try:
            path = write_plan(args.output, speaker_id=args.speaker_id,
                              pitch_subbank=args.pitch_subbank)
        except FileExistsError:
            parser.error("output exists; refusing to overwrite acquired-data manifest")
        print(path)
        return 0
    data = json.loads(args.manifest.expanduser().read_text(encoding="utf-8"))
    report = audit_manifest(data, args.root or args.manifest.parent)
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.expanduser().write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report["analysis_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
