"""Independent negative-control corpus for acoustic onset screening.

A recording with an explicit negative-category label is not automatically an
expert-verified negative. Evaluate trigger rates separately from validated FPR.
"""
from __future__ import annotations

import json
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio import read_wav

CATEGORIES = (
    "vowel_a", "vowel_i", "breath", "silence", "room_noise", "handling_noise",
)
ENVIRONMENTS = ("quiet", "indoor", "outdoor", "other")
DEVICES = ("phone", "built_in", "external_usb", "other", "unknown")
CORPUS_DIR = "negative_controls"
MAX_WAV_BYTES = 16 * 1024 * 1024


def _wav_info(wav_bytes: bytes) -> tuple[int, float]:
    import io
    if len(wav_bytes) < 44 or len(wav_bytes) > MAX_WAV_BYTES:
        raise ValueError("invalid WAV byte length")
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as source:
            if source.getcomptype() != "NONE" or source.getnchannels() != 1:
                raise ValueError("expected uncompressed mono WAV")
            if source.getsampwidth() != 2:
                raise ValueError("expected signed 16-bit PCM WAV")
            sample_rate = source.getframerate()
            if not 8000 <= sample_rate <= 192000:
                raise ValueError("unsupported audio sample rate")
            duration_s = source.getnframes() / sample_rate
    except (wave.Error, EOFError) as exc:
        raise ValueError("invalid WAV encoding") from exc
    if not 1.5 <= duration_s <= 6.0:
        raise ValueError("negative-control duration must be 1.5–6 seconds")
    return sample_rate, duration_s


def save_negative_recording(
    root: Path,
    category: str,
    wav_bytes: bytes,
    *,
    device_category: str = "unknown",
    environment: str = "other",
    actual_processing: dict[str, bool | None] | None = None,
) -> dict[str, Any]:
    """Create a labeled recording without touching calibration sessions."""
    if category not in CATEGORIES:
        raise ValueError("invalid negative-control category")
    if device_category not in DEVICES:
        raise ValueError("invalid capture device category")
    if environment not in ENVIRONMENTS:
        raise ValueError("invalid capture environment")
    actual_processing = actual_processing or {}
    allowed = ("echoCancellation", "noiseSuppression", "autoGainControl")
    if set(actual_processing) - set(allowed):
        raise ValueError("unsupported processing metadata")
    if any(v is not None and not isinstance(v, bool) for v in actual_processing.values()):
        raise ValueError("audio-processing metadata must be boolean or null")

    sample_rate, duration_s = _wav_info(wav_bytes)
    root = root.expanduser().resolve() / CORPUS_DIR
    root.mkdir(parents=True, exist_ok=True)
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:10]
    wav_name = f"{identifier}_{category}.wav"
    audio_path = root / wav_name
    metadata = {
        "schema": "negative_control_v1",
        "id": identifier,
        "category": category,
        "label_status": "self_reported",
        "environment": environment,
        "device_category": device_category,
        "audio_processing_actual": {k: actual_processing.get(k) for k in allowed},
        "sample_rate": sample_rate,
        "duration_ms": round(duration_s * 1000, 3),
        "wav": wav_name,
    }
    audio_path.write_bytes(wav_bytes)
    (root / f"{identifier}_{category}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"saved": str(audio_path), "metadata": metadata}


def evaluate_negative_corpus(root: Path) -> dict[str, Any]:
    """Trigger screening using the unchanged v1.4/v1.5 detector.

    No ground-truth timing is claimed. The nominal 1.25s cue is an
    experimentally fixed evaluation window, not a recorded beep.
    """
    from .calibration_transition_onset import acoustic_tracks
    from .calibration_frication_shoulder import detect_frication_shoulder

    root = root.expanduser().resolve() / CORPUS_DIR
    if not root.is_dir():
        raise ValueError("no negative-control recordings found")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for p in sorted(root.glob("*.json")):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            if meta.get("schema") != "negative_control_v1":
                continue
            file = root / str(meta["wav"])
            if file.parent.resolve() != root:
                raise ValueError("WAV path outside corpus")
            samples, sample_rate = read_wav(file)
            if len(samples) / sample_rate < 1.5:
                raise ValueError("recording is too short")
            result = detect_frication_shoulder(acoustic_tracks(samples, sample_rate), 1250.0)
            rows.append({
                "id": meta["id"],
                "category": meta["category"],
                "environment": meta["environment"],
                "device_category": meta["device_category"],
                "label_status": meta["label_status"],
                "core_trigger": result.core_onset_ms is not None,
                "shoulder_trigger": result.shoulder_onset_ms is not None,
                "core_onset_ms": result.core_onset_ms,
                "shoulder_onset_ms": result.shoulder_onset_ms,
                "status": result.status,
                "reason": result.reason,
            })
        except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            failures.append({"metadata": p.name, "reason": str(exc)})
    from collections import defaultdict
    by_category: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "recordings": 0, "core_triggers": 0, "shoulder_triggers": 0,
    })
    for row in rows:
        entry = by_category[row["category"]]
        entry["recordings"] += 1
        entry["core_triggers"] += int(row["core_trigger"])
        entry["shoulder_triggers"] += int(row["shoulder_trigger"])
    summary = {
        name: {**counts,
            "shoulder_trigger_fraction": round(counts["shoulder_triggers"] / counts["recordings"], 4)}
        for name, counts in sorted(by_category.items())
    }
    return {
        "analysis": "independent_negative_control_screening",
        "version": "1.9-development",
        "summary": summary,
        "total_recordings": len(rows),
        "failures": failures,
        "rows": rows,
        "limitations": (
            "Negative category labels are self-reported, not independently adjudicated. "
            "Trigger fractions do not establish validated specificity. "
            "Evaluation uses a fixed time window around 1.25 s; voice outside "
            "the window is not evaluated."
        ),
    }


def write_negative_corpus_report(root: Path) -> dict[str, Any]:
    report = evaluate_negative_corpus(root)
    dest = root.expanduser().resolve() / CORPUS_DIR / "negative_control_screening_v1.9.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["output"] = str(dest)
    return report
