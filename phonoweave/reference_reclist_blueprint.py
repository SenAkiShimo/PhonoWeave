"""Compile an existing bank's recording sequences into a reusable reclist draft.

The reference is intentionally *source-derived*: it does not optimize split
decisions, invent samples, or author an OTO file. Sequence strings are read
only from supported WAV basenames with matching pitch subbank provenance.
"""
from __future__ import annotations

from collections import defaultdict, Counter
import json
from pathlib import Path, PurePosixPath
import re

SCHEMA = "phonoweave.reference_reclist_blueprint.v0.1"
RECORD_SCHEMA = "phonoweave.lossless_alias_record.v0.1"
_PITCH = re.compile(r"^[A-Ga-g][#b]?-?\d+$")
_SYLLABLE = re.compile(r"^[a-züv]+$")


def _sequence(wav: str, pitch: str) -> tuple[str, ...] | None:
    """Only interpret explicit underscore-delimited note-suffixed filenames."""
    path = PurePosixPath(wav)
    if (not wav or wav == "[external_path]" or path.is_absolute()
            or ".." in path.parts or not _PITCH.fullmatch(pitch)
            or path.suffix.lower() != ".wav"
            or path.parts[0] != pitch):
        return None
    stem = path.stem
    if not (stem.startswith("_") and stem.endswith(pitch)):
        return None
    body = stem[1:-len(pitch)]
    tokens = body.split("_")
    if not tokens or any(not _SYLLABLE.fullmatch(t) for t in tokens):
        return None
    return tuple(tokens)


def compile_reference_reclist(records: list[dict]) -> dict:
    """Generate a draft from physical WAV references, not counts of OTO rows."""
    wavs = {}
    seen_entry = set()
    declared_pitches = set()
    for row in records:
        if row.get("schema") != RECORD_SCHEMA:
            raise ValueError("unsupported lossless alias schema")
        src, role = row["source"], row["role"]
        identity = tuple(src.get(k) for k in (
            "source_oto", "line_number", "original_alias"))
        if any(src.get(k) != role.get(k) for k in
               ("source_oto", "line_number", "original_alias")):
            raise ValueError("source and role provenance mismatch")
        if identity in seen_entry:
            raise ValueError("duplicate OTO source identity")
        seen_entry.add(identity)
        pitch = role.get("pitch_suffix") or ""
        if role.get("pitch_suffix_status") != "matched" or not _PITCH.fullmatch(pitch):
            continue
        declared_pitches.add(pitch)
        wav = src.get("source_wav") or ""
        if wav not in wavs:
            wavs[wav] = {
                "pitch": pitch,
                "sequence": _sequence(wav, pitch),
                "roles": Counter(),
                "aliases": set(),
                "oto_files": set(),
            }
        item = wavs[wav]
        if item["pitch"] != pitch:
            raise ValueError("inconsistent pitch suffix on one source WAV")
        item["roles"][role.get("role", "unknown")] += 1
        item["aliases"].add(src["original_alias"])
        item["oto_files"].add(src["source_oto"])

    pitches = sorted(declared_pitches)
    by_sequence = defaultdict(dict)
    excluded = Counter()
    for wav, item in sorted(wavs.items()):
        seq = item["sequence"]
        if seq is None:
            excluded["unsupported_or_nonsequence_filename"] += 1
            continue
        pitch = item["pitch"]
        if pitch in by_sequence[seq]:
            # Multiple physical WAVs under the same logical sequence are
            # ambiguous, not evidence for additional independent takes.
            excluded["duplicate_sequence_pitch"] += 1
            by_sequence[seq][pitch].append((wav, item))
        else:
            by_sequence[seq][pitch] = [(wav, item)]

    rows, ready = [], []
    for seq, groups in sorted(by_sequence.items()):
        present = sorted(groups)
        ambiguous = any(len(items) != 1 for items in groups.values())
        status = ("ambiguous_duplicate_source" if ambiguous else
                  "all_pitch_groups_observed" if present == pitches else
                  "some_pitch_groups_not_observed")
        text_line = "_".join(seq)
        if status == "all_pitch_groups_observed":
            ready.append(text_line)
        rows.append({
            "sequence": text_line,
            "syllable_count": len(seq),
            "pitch_groups_observed": present,
            "pitch_groups_not_observed": sorted(set(pitches) - set(present)),
            "status": status,
            "source_wavs": [
                {"pitch_group": pitch, "source_wav": wav,
                 "oto_files": sorted(item["oto_files"]),
                 "existing_alias_count": len(item["aliases"]),
                 "role_counts": dict(sorted(item["roles"].items()))}
                for pitch in present for wav, item in groups[pitch]
            ],
        })

    return {
        "schema": SCHEMA,
        "source": "existing_WAV_basenames_and_lossless_OTO_provenance",
        "pitch_groups": pitches,
        "unique_source_wavs": len(wavs),
        "interpretable_sequences": len(rows),
        "all_pitch_reference_sequences": len(ready),
        "sequence_status_counts": dict(sorted(Counter(
            row["status"] for row in rows).items())),
        "excluded_wav_groups": dict(sorted(excluded.items())),
        "sequences": rows,
        "reference_reclist": ready,
        "new_recordings_authorized": 0,
        "production_oto_export": "disabled",
        "limitation": ("A reference reclist describes an existing bank's filename "
                       "convention; it is not a personalized optimized plan or a "
                       "validated new OpenUtau phonemizer configuration."),
    }


def write_reference_reclist(report: dict, output: Path) -> Path:
    """Write to a newly created folder without modifying the original bank."""
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "reference_reclist_blueprint.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "reference_reclist.txt").write_text(
        "".join(line + "\n" for line in report["reference_reclist"]),
        encoding="utf-8",
    )
    (output / "README.txt").write_text(
        "SOURCE-DERIVED RECLIST REFERENCE — not a new recording order.\n"
        "Only sequences with one source WAV per observed pitch subbank are\n"
        "included in reference_reclist.txt. Review all rows marked partial\n"
        "or ambiguous in reference_reclist_blueprint.json. The program\n"
        "neither edits the voicebank nor creates aliases, OTO, or WAVs.\n",
        encoding="utf-8",
    )
    return output
