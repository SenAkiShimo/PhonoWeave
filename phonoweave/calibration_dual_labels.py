"""Independent dual acoustic boundaries without overwriting legacy anchors."""
from __future__ import annotations
import json
import math
import wave
from pathlib import Path
from .calibration_manual_labels import resolve_dev_selection, load_manual_labels

VERSION = "0.3"
ANCHORS = ("consonant_onset", "cv_boundary")
FILENAME = "calibration_dual_anchor_labels_v0.3.json"


def dual_path(session_dir: Path) -> Path:
    return session_dir.expanduser().resolve() / "analysis" / FILENAME


def load_dual_labels(session_dir: Path) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    p = dual_path(session_dir)
    if not p.exists():
        return {"analysis": "calibration_dual_anchor_labels", "version": VERSION,
                "session_id": session_dir.name, "labels": {}}
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != VERSION or not isinstance(payload.get("labels"), dict):
        raise ValueError("incompatible dual anchor file")
    return payload


def save_dual_label(session_dir: Path, prompt_index: int, occurrence: int,
                    anchor_type: str, anchor_ms_after_cue: float | None,
                    status: str, *, from_legacy: bool = False) -> dict[str, object]:
    if anchor_type not in ANCHORS:
        raise ValueError("invalid anchor type")
    if occurrence not in (1, 2):
        raise ValueError("invalid occurrence")
    if status not in ("ok", "uncertain", "unset"):
        raise ValueError("invalid status")
    session_dir = session_dir.expanduser().resolve()
    prompts = {p.prompt_index: p for p in resolve_dev_selection(session_dir)}
    prompt = prompts.get(prompt_index)
    if prompt is None:
        raise ValueError("prompt outside fixed selection")
    value = None
    out_of_slot = False
    if status != "unset":
        if anchor_ms_after_cue is None:
            raise ValueError("missing anchor timestamp")
        value = float(anchor_ms_after_cue)
        if not math.isfinite(value):
            raise ValueError("anchor timestamp must be finite")
        with wave.open(str(session_dir / "recordings" / prompt.wav), "rb") as handle:
            duration = 1000.0 * handle.getnframes() / handle.getframerate()
        cue = prompt.cues_ms[occurrence - 1]
        if not 0 <= cue + value <= duration:
            raise ValueError("anchor outside recording duration")
        out_of_slot = not prompt.label_min_ms_after_cue(occurrence) <= value <= prompt.label_max_ms_after_cue(occurrence)

    result = load_dual_labels(session_dir)
    labels = result["labels"]
    item = labels.setdefault(str(prompt_index), {
        "prompt_index": prompt_index, "base_unit": prompt.base_unit,
        "context_family": prompt.context_family, "syllable": prompt.syllable,
        "class_name": prompt.class_name, "occurrences": {}
    })
    occ = item["occurrences"].setdefault(str(occurrence), {})
    if status == "unset":
        occ.pop(anchor_type, None)
    else:
        occ[anchor_type] = {
            "anchor_ms_after_cue": round(value, 3),
            "absolute_anchor_ms": round(prompt.cues_ms[occurrence - 1] + value, 3),
            "status": status,
            "outside_expected_slot": out_of_slot,
            "source": "legacy_reviewed_copy" if from_legacy else "manual_v0.3",
        }
    destination = dual_path(session_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def copy_legacy_label(session_dir: Path, prompt_index: int, occurrence: int,
                      anchor_type: str) -> dict[str, object]:
    """Explicit user action only; preserve legacy uncertainty and provenance."""
    old = load_manual_labels(session_dir)["labels"]
    entry = old.get(str(prompt_index), {}).get("occurrences", {}).get(str(occurrence))
    if not isinstance(entry, dict):
        raise ValueError("no legacy anchor to copy")
    result = load_dual_labels(session_dir)
    existing = result["labels"].get(str(prompt_index), {}).get("occurrences", {}).get(str(occurrence), {})
    if anchor_type in existing:
        raise ValueError("destination anchor already exists; clear it first")
    return save_dual_label(session_dir, prompt_index, occurrence, anchor_type,
                           float(entry["anchor_ms_after_cue"]), str(entry["status"]),
                           from_legacy=True)
