"""Read-only calibration evidence adapter for downstream research review.

Consumes versioned acoustic predictions and separately authored dual-anchor
annotations. Produces an auditable report; does not mutate SpeakerProfile,
SynthesisInventory, OTO, source recordings or manual labels.

Temporal agreement is NOT evidence of acoustic context separability,
synthesis relevance or speaker identity. All profile decisions remain blocked.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import wave

from .calibration_dual_labels import load_dual_labels
from .calibration_io import load_calibration_session
from .calibration_results import OUTPUT_FILENAME as PREDICTION_NAME
from .calibration_results import SCHEMA as PREDICTION_SCHEMA
from .calibration_results import SCHEMA_VERSION as PREDICTION_VERSION

SCHEMA = "phonoweave.calibration_evidence"
VERSION = "1.0.0"
OUTPUT_FILENAME = "calibration_evidence_v1.json"
ANCHOR_NAMES = ("consonant_onset", "cv_boundary")


def _nonnegative_time(value: object, name: str, duration: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= duration:
        raise ValueError(f"{name} outside recording duration")
    return round(result, 3)


def _manual_anchor(
    raw: object, name: str, duration: float
) -> tuple[dict | None, str | None]:
    """Return accepted human annotation or an explicit review reason.

    Here 'accepted' means the annotator submitted an in-slot, manually placed
    label, not that the phonetic boundary is independently adjudicated.
    """
    if raw is None:
        return None, f"{name}_missing"
    if not isinstance(raw, dict):
        raise ValueError(f"{name} label must be an object")
    if raw.get("status") != "ok":
        return None, f"{name}_not_ok"
    if raw.get("source") != "manual_v0.3":
        return None, f"{name}_not_independent_manual_source"
    if raw.get("outside_expected_slot") is not False:
        return None, f"{name}_slot_review"
    absolute = _nonnegative_time(
        raw.get("absolute_anchor_ms"), f"{name} manual anchor", duration
    )
    return {
        "absolute_ms": absolute,
        "source": "manual_v0.3",
        "reviewed_independently": False,
    }, None


def _checked_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid JSON object: {path.name}")
    return payload


def adapt_calibration_evidence(
    session_dir: Path,
    *, prediction_path: Path | None = None,
    output: Path | None = None,
) -> dict:
    """Build an immutable measurement-review view of an existing session.

    Validate file provenance before joining predictions to manual labels.
    Do not synthesize missing anchors and never infer any inventory decision.
    """
    session_dir = session_dir.expanduser().resolve()
    session = load_calibration_session(session_dir)
    prediction_path = (
        prediction_path.expanduser().resolve() if prediction_path is not None
        else session_dir / "analysis" / PREDICTION_NAME
    )
    prediction = _checked_json(prediction_path)
    if (prediction.get("schema") != PREDICTION_SCHEMA
            or prediction.get("schema_version") != PREDICTION_VERSION):
        raise ValueError("unsupported calibration prediction schema")
    if prediction.get("session_id") != session["session_id"]:
        raise ValueError("prediction session ID mismatch")
    model = prediction.get("model")
    if (not isinstance(model, dict)
            or model.get("maturity") != "research_only"
            or model.get("time_unit") != "ms_from_recording_start"
            or model.get("default_voicebank_decisions_modified") is not False):
        raise ValueError("incompatible or untrusted calibration model provenance")

    source_rows = prediction.get("recording_sources")
    candidate_rows = prediction.get("predictions")
    if not isinstance(source_rows, list) or not isinstance(candidate_rows, list):
        raise ValueError("missing prediction sources or candidates")
    recorded = session["recordings"]
    protocol = session["protocol"]["prompts"]
    recording_dir = (session_dir / "recordings").resolve()
    sources: dict[int, dict] = {}
    for row in source_rows:
        if not isinstance(row, dict) or type(row.get("prompt_index")) is not int:
            raise ValueError("invalid recording source")
        index = row["prompt_index"]
        if index in sources or index not in recorded:
            raise ValueError("duplicate or unknown recording source")
        prompt = protocol[index]
        if prompt["base_unit"] not in {"sh", "x"}:
            raise ValueError("unsupported source class")
        expected_relpath = "recordings/" + str(recorded[index]["wav"])
        if row.get("recording") != expected_relpath:
            raise ValueError("source recording metadata mismatch")
        wav = (recording_dir / str(recorded[index]["wav"])).resolve()
        if wav.parent != recording_dir or not wav.is_file():
            raise ValueError("recording must reside within session recordings")
        digest = sha256(wav.read_bytes()).hexdigest()
        if row.get("sha256") != digest:
            raise ValueError("recording SHA-256 mismatch")
        with wave.open(str(wav), "rb") as w:
            rate = w.getframerate()
            duration = 1000.0 * w.getnframes() / rate
        if type(row.get("sample_rate")) is not int or row["sample_rate"] != rate:
            raise ValueError("source sample rate mismatch")
        sources[index] = {
            "recording": expected_relpath,
            "sha256": digest,
            "duration_ms": round(duration, 3),
        }

    labels_payload = load_dual_labels(session_dir)
    if labels_payload.get("session_id") != session["session_id"]:
        raise ValueError("manual annotation session ID mismatch")
    labels = labels_payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("invalid dual-anchor annotation payload")
    observations = []
    seen: set[tuple[int, int]] = set()
    for row in candidate_rows:
        if not isinstance(row, dict):
            raise ValueError("invalid candidate entry")
        index = row.get("prompt_index")
        occurrence = row.get("occurrence")
        if type(index) is not int or type(occurrence) is not int:
            raise ValueError("invalid candidate key")
        key = (index, occurrence)
        if key in seen or index not in sources:
            raise ValueError("duplicate or unproven candidate")
        seen.add(key)
        prompt = protocol[index]
        if (row.get("base_unit") != prompt["base_unit"]
                or row.get("context_family") != prompt["context_family"]
                or row.get("syllable") != prompt["syllable"]):
            raise ValueError("candidate prompt metadata mismatch")
        expected = int(prompt["target_occurrences"])
        if not 1 <= occurrence <= expected:
            raise ValueError("candidate occurrence outside protocol")
        detector = row.get("detector")
        refinement = row.get("experimental_refinement")
        if not isinstance(detector, dict) or not isinstance(refinement, dict):
            raise ValueError("malformed candidate evidence")
        status = detector.get("status")
        if status not in {"paired_candidate", "review", "unresolved"}:
            raise ValueError("unsupported candidate status")
        duration = sources[index]["duration_ms"]
        onset = detector.get("consonant_onset_ms")
        cv = detector.get("cv_boundary_ms")
        if status == "paired_candidate":
            onset = _nonnegative_time(onset, "predicted onset", duration)
            cv = _nonnegative_time(cv, "predicted CV boundary", duration)
            if onset >= cv:
                raise ValueError("predicted onset/CV temporal contradiction")
        else:
            onset = cv = None
        selected = refinement.get("selected_onset_ms")
        if status == "paired_candidate":
            selected = _nonnegative_time(selected, "experimental onset", duration)
            if selected > onset or selected >= cv:
                raise ValueError("invalid experimental onset")
        elif selected is not None:
            raise ValueError("unpaired candidate cannot have experimental onset")

        annotation = (
            labels.get(str(index), {}).get("occurrences", {})
            .get(str(occurrence), {})
        )
        if not isinstance(annotation, dict):
            raise ValueError("invalid annotation occurrence")
        accepted = {}
        reasons = []
        for anchor_name in ANCHOR_NAMES:
            value, reason = _manual_anchor(
                annotation.get(anchor_name), anchor_name, duration
            )
            accepted[anchor_name] = value
            if reason:
                reasons.append(reason)
        c = accepted["consonant_onset"]
        v = accepted["cv_boundary"]
        if c is not None and v is not None and c["absolute_ms"] >= v["absolute_ms"]:
            reasons.append("manual_anchor_order_conflict")
        coherent = (not reasons and c is not None and v is not None)
        if coherent and status == "paired_candidate":
            evidence_state = "manual_pair_available"
        elif status != "paired_candidate":
            evidence_state = "detector_not_paired"
        else:
            evidence_state = "manual_review_required"
        observations.append({
            "prompt_index": index,
            "occurrence": occurrence,
            "base_unit": prompt["base_unit"],
            "context_family": prompt["context_family"],
            "syllable": prompt["syllable"],
            "recording_sha256": sources[index]["sha256"],
            "evidence_state": evidence_state,
            "review_reasons": reasons,
            "detector": {
                "status": status,
                "method": detector.get("method"),
                "consonant_onset_ms": onset,
                "cv_boundary_ms": cv,
            },
            "manual": {
                "consonant_onset": c,
                "cv_boundary": v,
                "label_status": "complete" if coherent else "requires_review",
            },
            "experimental": {
                "selected_onset_ms": selected,
                "selected_minus_manual_ms": (
                    round(selected - c["absolute_ms"], 3)
                    if status == "paired_candidate" and coherent else None
                ),
                "detector_minus_manual_ms": (
                    round(onset - c["absolute_ms"], 3)
                    if status == "paired_candidate" and coherent else None
                ),
                "cv_minus_manual_ms": (
                    round(cv - v["absolute_ms"], 3)
                    if status == "paired_candidate" and coherent else None
                ),
            },
            "speaker_profile_admissibility": "blocked",
            "inventory_decision_admissibility": "blocked",
        })

    for index in sources:
        n = int(protocol[index]["target_occurrences"])
        for occurrence in range(1, n + 1):
            if (index, occurrence) not in seen:
                raise ValueError("missing candidate for recorded supported target")

    counts = Counter(row["evidence_state"] for row in observations)
    report = {
        "schema": SCHEMA,
        "schema_version": VERSION,
        "session_id": session["session_id"],
        "prediction_contract": {
            "schema": prediction["schema"],
            "schema_version": prediction["schema_version"],
            "sha256": sha256(prediction_path.read_bytes()).hexdigest(),
        },
        "manual_annotation_file": "analysis/calibration_dual_anchor_labels_v0.3.json",
        "provenance_verified": True,
        "identity_binding": "not_established",
        "profile_decisions_modified": False,
        "synthesis_inventory_modified": False,
        "summary": {
            "observations": len(observations),
            "manual_pairs_available": counts["manual_pair_available"],
            "manual_review_required": counts["manual_review_required"],
            "detector_not_paired": counts["detector_not_paired"],
        },
        "observations": observations,
        "interpretation": {
            "manual_pair_available": "Two manually placed timing marks are available, not independently adjudicated.",
            "profile_admissibility": "No automatic split/merge or confidence decisions are allowed.",
            "speaker_identity": "Calibration and voicebank speaker identity has not been linked.",
        },
        "limitations": [
            "Only recorded elicited Mandarin x/sh candidates are represented.",
            "Manual_v0.3 annotations are not independent inter-annotator verification.",
            "Timing alignment agreement does not establish contrast distinctiveness or synthesis utility.",
            "Cue synchronization is nominal; predictions and labels can share systematic error.",
            "No automatic speaker-profile or synthesis-inventory modifications.",
        ],
    }
    destination = (
        output.expanduser().resolve() if output is not None
        else session_dir / "analysis" / OUTPUT_FILENAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
