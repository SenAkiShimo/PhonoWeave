"""Guided acquisition of the frozen controlled /sh/ pilot grid.

Optional recording dependency: sounddevice (PortAudio). WAVs are PCM16 mono
48 kHz, and captures are never overwritten. No audio is generated or shared.
"""
from __future__ import annotations

import argparse
from datetime import date
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import tempfile
import time
import wave

from .synthesis_relevance_capture_screen import analyze_pcm, format_screen

from .synthesis_relevance_controlled_corpus import (
    SCHEMA, FINALS, ROLES, REPEATS, audit_manifest,
)

RATE = 48000


def _record_pcm(frames: int, *, device=None) -> bytes:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "microphone capture requires: python -m pip install sounddevice"
        ) from exc
    with sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16",
                           device=device) as stream:
        pcm, overflow = stream.read(frames)
    if overflow:
        raise RuntimeError("microphone input overflow; audio not saved")
    return bytes(pcm)


def _save_manifest(path: Path, data: dict):
    # Atomic update after each successful WAV. Original manifest is never
    # re-generated; this only records acquired-file metadata.
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent,
                                         prefix=".manifest-", suffix=".tmp",
                                         encoding="utf-8", delete=False) as f:
            tmp = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _write_pcm_exclusive(path: Path, pcm: bytes, frames: int):
    if len(pcm) != frames * 2:
        raise ValueError("microphone returned the wrong number of PCM16 frames")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: refuse any existing take including failed imports.
    with path.open("xb") as f:
        with wave.open(f, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(RATE)
            out.writeframes(pcm)


def _relative_audio_path(root: Path, rel: str):
    relative = Path(rel)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid relative WAV path in capture manifest")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("WAV path escapes pilot root")
    return path


def _session_order(records, session_id):
    # Deterministic shuffled prompt order; never select by acoustic outcome.
    order = list(records)
    seed = int.from_bytes(sha256(session_id.encode("utf-8")).digest()[:8], "big")
    random.Random(seed).shuffle(order)
    return order


def _countdown(seconds=3, sleeper=time.sleep, output=print):
    for second in range(seconds, 0, -1):
        output(f"Recording in {second}...")
        sleeper(1)
    output("RECORD")


def trial_capture(root: Path, *, device=None, target_pitch_hz=233.08,
                  duration=3.0, recorder=None, ask=input, countdown=_countdown):
    """Four disposable-free but independent preflight WAVs; no manifest edits."""
    if not 60 <= target_pitch_hz <= 1000 or not 0.3 <= duration <= 8:
        raise ValueError("invalid target pitch or duration")
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    recorder = recorder or _record_pcm
    results = []
    for final in ("a", "ua", "ai", "uai"):
        name = "sh" + final
        target = root / f"{name}.wav"
        if target.exists():
            raise FileExistsError(f"preflight WAV already exists: {target}")
        answer = ask(f"Pilot {name}: Enter to record, q to stop: ")
        if answer.strip().lower() == "q":
            return results
        if answer.strip():
            raise ValueError("invalid response")
        countdown()
        frames = round(RATE * duration)
        pcm = recorder(frames, device=device)
        screen = analyze_pcm(pcm, rate=RATE, target_pitch_hz=target_pitch_hz)
        _write_pcm_exclusive(target, pcm, frames)
        results.append({"syllable": name, "wav": str(target), **screen})
        print(f"{name}: {format_screen(screen)}")
    return results


def capture_session(manifest: Path, session_id: str, *, room_id: str,
                    gain_setting: str, target_pitch_hz: float, duration: float = 3.0,
                    device=None, device_id: str = "default_input",
                    confirm_reset: bool = False, recorder=None, ask=input,
                    capture_date=None, countdown=None):
    """Record missing takes; preserve existing files and manifest hashes.

    A repeated launch is safe: a hashed WAV is checked and skipped. An
    unregistered pre-existing WAV is blocked, never overwritten. A session
    must be acquired on one calendar date, and S02 must be on a later date.
    """
    if session_id not in dict(ROLES):
        raise ValueError("session must be S01 or S02")
    if not 0.3 <= duration <= 8.0:
        raise ValueError("recording duration must be between 0.3 and 8 seconds")
    if not 60 <= target_pitch_hz <= 1000:
        raise ValueError("target pitch must be between 60 and 1000 Hz")
    if not room_id.strip() or not gain_setting.strip() or not device_id.strip():
        raise ValueError("room, gain, and device identifiers are required")
    if not confirm_reset:
        raise ValueError("confirm independent session setup reset before recording")

    manifest = manifest.expanduser().resolve(strict=True)
    root = manifest.parent
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError("unrecognized capture manifest schema")

    # Check destination conflicts *before* the full WAV-format audit. A
    # pre-existing unregistered file is a non-overwrite conflict even when
    # it is corrupt or not a WAV at all. Never change the manifest on error.
    # The full audit still validates all registered WAVs and the frozen grid.
    for row in data.get("recordings", []):
        if not isinstance(row, dict) or row.get("session_id") != session_id:
            continue
        rel = row.get("relative_wav")
        if not isinstance(rel, str) or not rel:
            continue  # Full manifest audit reports missing/invalid paths.
        target = _relative_audio_path(root, rel)
        digest = row.get("wav_sha256")
        if target.exists():
            if not digest or sha256(target.read_bytes()).hexdigest() != digest:
                raise FileExistsError(
                    f"WAV exists without matching manifest hash: {target}"
                )
        elif digest:
            raise FileNotFoundError(
                f"manifest contains hash but WAV is missing: {target}"
            )

    report = audit_manifest(data, root)
    if report["errors"]:
        raise ValueError("invalid pilot manifest: " + "; ".join(report["errors"]))
    slots = [row for row in data["recordings"] if row["session_id"] == session_id]
    if len(slots) != len(FINALS) * REPEATS:
        raise ValueError("invalid session recording grid")
    session = next(x for x in data["sessions"] if x["session_id"] == session_id)
    recorded_date = capture_date or date.today().isoformat()
    date.fromisoformat(recorded_date)
    if session_id == "S02":
        source = next(x for x in data["sessions"] if x["session_id"] == "S01")
        if not source.get("capture_date"):
            raise ValueError("S01 must be completed before heldout S02")
        if any(not row.get("wav_sha256") for row in data["recordings"]
               if row["session_id"] == "S01"):
            raise ValueError("S01 WAV hashes are incomplete")
        if date.fromisoformat(recorded_date) <= date.fromisoformat(source["capture_date"]):
            raise ValueError("heldout S02 must be captured on a later date")

    expected_meta = {
        "capture_date": recorded_date, "device_id": device_id,
        "room_id": room_id, "gain_setting": gain_setting,
        "target_pitch_hz": target_pitch_hz, "setup_reset_confirmed": True,
    }
    for key, value in expected_meta.items():
        prior = session.get(key)
        if prior is not None and prior != value:
            raise ValueError(f"session metadata changed on resume: {key}")
    session.update(expected_meta)
    _save_manifest(manifest, data)

    frames = round(RATE * duration)
    recorder = recorder or _record_pcm
    captured = 0
    order = _session_order(slots, session_id)
    for index, row in enumerate(order, 1):
        target = _relative_audio_path(root, row["relative_wav"])
        digest = row.get("wav_sha256")
        if target.exists():
            if digest and sha256(target.read_bytes()).hexdigest() == digest:
                print(f"[{index:02d}/24] existing verified take: {row['take_id']}")
                continue
            raise FileExistsError(f"WAV exists without matching manifest hash: {target}")
        if digest:
            raise FileNotFoundError(f"manifest contains hash but WAV is missing: {target}")
        syllable = "sh" + row["final"]
        answer = ask(
            f"[{index:02d}/24] {syllable} (tone 1, sustained pitch) — "
            f"{row['take_id']}\n"
            f"Press Enter to record {duration:g} s after a short pause; "
            "type q to stop safely: "
        )
        if answer.strip().lower() == "q":
            print("Recording stopped; existing WAVs and hashes preserved.")
            return captured
        if answer.strip():
            raise ValueError("unrecognized response; no recording for this take")
        if countdown is not None:
            countdown()
        pcm = recorder(frames, device=device)
        screen = analyze_pcm(pcm, rate=RATE, target_pitch_hz=target_pitch_hz)
        _write_pcm_exclusive(target, pcm, frames)
        print(f"Capture screen: {format_screen(screen)}")
        row["wav_sha256"] = sha256(target.read_bytes()).hexdigest()
        row["capture_order"] = index
        _save_manifest(manifest, data)
        captured += 1
        print(f"Saved: {row['relative_wav']}")
    print(f"Session {session_id}: {captured} new captures; existing takes preserved.")
    return captured


def main(argv=None):
    parser = argparse.ArgumentParser(description="Guided, lossless 48-take pilot recording")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--trial-output", type=Path)
    parser.add_argument("--countdown-seconds", type=int, default=3)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--session", choices=("S01", "S02"))
    parser.add_argument("--room-id")
    parser.add_argument("--gain-setting")
    parser.add_argument("--target-pitch-hz", type=float)
    parser.add_argument("--device", default=None,
                        help="sounddevice device index or name; omitted = default input")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--confirm-setup-reset", action="store_true")
    args = parser.parse_args(argv)
    if args.list_devices:
        try:
            import sounddevice as sd
        except ImportError:
            parser.error("install optional input driver: python -m pip install sounddevice")
        print(sd.query_devices())
        return 0
    if not 0 <= args.countdown_seconds <= 10:
        parser.error("countdown must be between 0 and 10 seconds")
    if args.trial:
        if not args.trial_output or args.target_pitch_hz is None:
            parser.error("--trial requires --trial-output and --target-pitch-hz")
        try:
            import sounddevice as sd
            device = int(args.device) if args.device and args.device.isdecimal() else args.device
            sd.query_devices(device, kind="input")
            trial_capture(args.trial_output, device=device, target_pitch_hz=args.target_pitch_hz,
                          duration=args.duration,
                          countdown=lambda: _countdown(args.countdown_seconds))
        except (OSError, ValueError, RuntimeError, FileExistsError) as exc:
            parser.error(str(exc))
        return 0
    if (args.manifest is None or args.session is None or
            args.room_id is None or args.gain_setting is None or
            args.target_pitch_hz is None):
        parser.error("capture requires --manifest, --session, --room-id, "
                     "--gain-setting and --target-pitch-hz")
    if not args.confirm_setup_reset:
        parser.error("capture requires an independently verified setup reset")
    device = args.device
    if device is not None and device.isdecimal():
        device = int(device)
    try:
        # This checks availability before a manifest can be mutated.
        import sounddevice as sd
        resolved = sd.query_devices(device, kind="input")
        device_id = resolved["name"]
        capture_session(
            args.manifest, args.session,
            room_id=args.room_id, gain_setting=args.gain_setting,
            target_pitch_hz=args.target_pitch_hz,
            duration=args.duration, device=device, device_id=device_id,
            confirm_reset=True,
            countdown=lambda: _countdown(args.countdown_seconds),
        )
    except (OSError, ValueError, RuntimeError, FileNotFoundError,
            FileExistsError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        print("\nRecording interrupted; completed WAVs remain saved.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
