from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .calibration_beep_alignment import detect_beep_sequence
from .audio import read_wav
from .calibration_io import load_calibration_session


SELECTION_VERSION = "dev16_v0.2"

DEV_ANCHOR_TARGETS: tuple[tuple[str, str], ...] = (
    ("zh", "plain"),
    ("ch", "rounded"),
    ("j", "plain"),
    ("q", "rounded"),
    ("x", "plain"),
    ("sh", "rounded"),
    ("f", "rounded"),
    ("h", "other"),
    ("r", "front"),
    ("m", "i_series"),
    ("n", "u_series"),
    ("n", "other"),
    ("b", "i_series"),
    ("d", "i_series"),
    ("g", "u_series"),
    ("k", "other"),
)

_ONSET_CLASS = {
    **{item: "stop" for item in ("b", "p", "d", "t", "g", "k")},
    **{item: "affricate" for item in ("zh", "ch", "z", "c", "j", "q")},
    **{item: "fricative" for item in ("f", "h", "x", "sh", "s")},
    **{item: "sonorant" for item in ("m", "n", "l", "r")},
}

ANCHOR_TYPES = {
    "stop": "manual_release_anchor",
    "affricate": "manual_frication_release_anchor",
    "fricative": "manual_frication_onset_anchor",
    "sonorant": "manual_sonorant_onset_anchor",
    "other": "manual_acoustic_anchor",
}


@dataclass(frozen=True)
class ManualPrompt:
    prompt_index: int
    base_unit: str
    context_family: str
    syllable: str
    class_name: str
    wav: str
    cues_ms: tuple[float, float]
    prev_cues_ms: tuple[float, float]
    next_cues_ms: tuple[float, float]
    token_indices: tuple[int, int]

    def label_min_ms_after_cue(self, occurrence: int) -> float:
        cue = self.cues_ms[occurrence - 1]
        prev_cue = self.prev_cues_ms[occurrence - 1]
        return prev_cue - cue + 20.0

    def label_max_ms_after_cue(self, occurrence: int) -> float:
        cue = self.cues_ms[occurrence - 1]
        next_cue = self.next_cues_ms[occurrence - 1]
        return next_cue - cue - 20.0


def label_path(session_dir: Path) -> Path:
    return session_dir.expanduser().resolve() / "analysis" / "calibration_manual_anchor_labels_v0.2.json"


@lru_cache(maxsize=8)
def _resolve_cached(session_dir_text: str) -> tuple[ManualPrompt, ...]:
    session_dir = Path(session_dir_text)
    session = load_calibration_session(session_dir)
    protocol = session["protocol"]
    recordings = session["recordings"]
    if not isinstance(protocol, dict) or not isinstance(recordings, dict):
        raise ValueError("invalid calibration session")
    prompts = protocol.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("calibration protocol is missing prompts")

    by_key: dict[tuple[str, str], tuple[int, dict[str, object]]] = {}
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            continue
        key = (str(prompt.get("base_unit", "")), str(prompt.get("context_family", "")))
        by_key[key] = (index, prompt)

    selected: list[ManualPrompt] = []
    missing: list[str] = []
    for key in DEV_ANCHOR_TARGETS:
        found = by_key.get(key)
        if found is None:
            missing.append(f"{key[0]}:{key[1]}")
            continue
        index, prompt = found
        recording = recordings.get(index)
        if not isinstance(recording, dict):
            missing.append(f"{key[0]}:{key[1]} (recording)")
            continue

        tokens = prompt.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            tokens = str(prompt.get("spoken_pattern", "")).split()
        syllable = str(prompt.get("syllable", ""))
        target_indices = [i for i, token in enumerate(tokens) if str(token) == syllable]
        if len(target_indices) != 2:
            missing.append(f"{key[0]}:{key[1]} (targets)")
            continue

        wav_name = str(recording["wav"])
        samples, sample_rate = read_wav(session_dir / "recordings" / wav_name)
        sequence = detect_beep_sequence(samples, sample_rate, len(tokens))

        current_events = [
            sequence.events[target_indices[0] + 1],
            sequence.events[target_indices[1] + 1],
        ]
        previous_events = [
            sequence.events[target_indices[0]],
            sequence.events[target_indices[1]],
        ]
        next_events = [
            sequence.events[target_indices[0] + 2],
            sequence.events[target_indices[1] + 2],
        ]

        selected.append(
            ManualPrompt(
                prompt_index=index,
                base_unit=key[0],
                context_family=key[1],
                syllable=syllable,
                class_name=_ONSET_CLASS.get(key[0], "other"),
                wav=wav_name,
                cues_ms=tuple(float(event.time_ms) for event in current_events),
                prev_cues_ms=tuple(float(event.time_ms) for event in previous_events),
                next_cues_ms=tuple(float(event.time_ms) for event in next_events),
                token_indices=(target_indices[0], target_indices[1]),
            )
        )

    if missing:
        raise ValueError("development anchor selection is incomplete: " + ", ".join(missing))
    return tuple(selected)


def resolve_dev_selection(session_dir: Path) -> tuple[ManualPrompt, ...]:
    resolved = session_dir.expanduser().resolve()
    return _resolve_cached(str(resolved))


def load_manual_labels(session_dir: Path) -> dict[str, object]:
    path = label_path(session_dir)
    if not path.is_file():
        return {
            "analysis": "calibration_manual_anchor_labels",
            "version": "0.2",
            "selection_version": SELECTION_VERSION,
            "session_id": session_dir.expanduser().resolve().name,
            "labels": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manual anchor label file is invalid")
    if payload.get("selection_version") != SELECTION_VERSION:
        raise ValueError("manual anchor label selection version does not match")
    labels = payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("manual anchor label file has invalid labels")
    return payload


def save_manual_label(
    session_dir: Path,
    prompt_index: int,
    occurrence: int,
    anchor_ms_after_cue: float | None,
    status: str,
) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    selected = {item.prompt_index: item for item in resolve_dev_selection(session_dir)}
    prompt = selected.get(prompt_index)
    if prompt is None:
        raise ValueError("prompt is not part of the fixed development selection")
    if occurrence not in (1, 2):
        raise ValueError("occurrence must be 1 or 2")
    if status not in {"ok", "uncertain", "unset"}:
        raise ValueError("invalid manual label status")

    if status != "unset":
        if anchor_ms_after_cue is None:
            raise ValueError("anchor time is required")
        value = float(anchor_ms_after_cue)
        minimum = prompt.label_min_ms_after_cue(occurrence)
        maximum = prompt.label_max_ms_after_cue(occurrence)
        if value < minimum or value > maximum:
            raise ValueError(
                f"anchor must be between {minimum:.1f} and {maximum:.1f} ms relative to cue"
            )
    else:
        value = None

    payload = load_manual_labels(session_dir)
    labels = payload.setdefault("labels", {})
    if not isinstance(labels, dict):
        raise ValueError("manual anchor label file has invalid labels")

    key = str(prompt_index)
    row = labels.setdefault(
        key,
        {
            "prompt_index": prompt_index,
            "base_unit": prompt.base_unit,
            "context_family": prompt.context_family,
            "syllable": prompt.syllable,
            "class_name": prompt.class_name,
            "anchor_type": ANCHOR_TYPES.get(prompt.class_name, ANCHOR_TYPES["other"]),
            "occurrences": {},
        },
    )
    if not isinstance(row, dict):
        raise ValueError("manual anchor row is invalid")
    occurrences = row.setdefault("occurrences", {})
    if not isinstance(occurrences, dict):
        raise ValueError("manual anchor occurrences are invalid")

    if status == "unset":
        occurrences.pop(str(occurrence), None)
    else:
        cue_ms = prompt.cues_ms[occurrence - 1]
        occurrences[str(occurrence)] = {
            "occurrence": occurrence,
            "cue_ms": round(cue_ms, 3),
            "anchor_ms_after_cue": round(value, 3),
            "absolute_anchor_ms": round(cue_ms + value, 3),
            "status": status,
        }

    analysis_dir = session_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    path = label_path(session_dir)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def setup_payload(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    selected = resolve_dev_selection(session_dir)
    labels = load_manual_labels(session_dir)
    return {
        "selection_version": SELECTION_VERSION,
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "prompts": [
            {
                "prompt_index": item.prompt_index,
                "base_unit": item.base_unit,
                "context_family": item.context_family,
                "syllable": item.syllable,
                "class_name": item.class_name,
                "wav": item.wav,
                "cues_ms": [round(value, 3) for value in item.cues_ms],
                "prev_cues_ms": [round(value, 3) for value in item.prev_cues_ms],
                "next_cues_ms": [round(value, 3) for value in item.next_cues_ms],
                "token_indices": list(item.token_indices),
            }
            for item in selected
        ],
        "labels": labels.get("labels", {}),
    }
