"""User-native, template-based WAV/OTO coverage for PhonoWeave.

This is an opaque author-name inventory check, not phoneme discovery.
No Mandarin tables, IPA mapping or automatic new syllables are involved.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "phonoweave.user_native_recording_plan.v2.1"
TARGET_SCHEMA = "phonoweave.user_native_recording_targets.v1"


def _name(value: Any, *, wav: bool = False) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("invalid original recording name")
    if len(value) > 240 or value in (".", ".."):
        raise ValueError("invalid recording name")
    if any(c in value for c in ("/", "\\", "\x00", "\r", "\n")) or any(ord(c) < 32 for c in value):
        raise ValueError("recording name must not contain a path or control character")
    value = unicodedata.normalize("NFC", value)
    if wav and not value.lower().endswith(".wav"):
        raise ValueError("expected source WAV must end in .wav")
    return value


def _bank(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid subbank")
    if value == ".":
        return "."
    if value.startswith(("/", "\\")) or "\\" in value:
        raise ValueError("invalid subbank path")
    value = unicodedata.normalize("NFC", value.strip("/"))
    if not value or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError("invalid subbank path")
    return value


def read_targets(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(obj, dict) or obj.get("schema") != TARGET_SCHEMA or not isinstance(obj.get("targets"), list):
            raise ValueError("invalid user-native target JSON schema")
        rows = obj["targets"]
    else:
        rows = [{"line": line.strip()} for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip() and not line.lstrip().startswith(("#", ";"))]
    if not rows:
        raise ValueError("existing recording template is empty; cannot infer intended sounds")
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("target must be an object")
        name = _name(row.get("line"))
        wav = _name(row.get("wav", name + ".wav"), wav=True)
        subbanks = row.get("subbanks")
        if subbanks is not None:
            if not isinstance(subbanks, list) or not subbanks:
                raise ValueError("target subbanks must be a nonempty list")
            subbanks = [_bank(b) for b in subbanks]
            if len(subbanks) != len(set(subbanks)):
                raise ValueError("duplicate target subbank")
        if name in seen:
            raise ValueError("duplicate original recording line; ambiguous WAV mapping")
        seen.add(name)
        result.append({"line": name, "wav": wav, "subbanks": subbanks})
    return result


def _subbanks(root: Path) -> dict[str, Path]:
    result = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.lower() != "oto.ini":
            continue
        folder = p.parent.resolve()
        if folder.is_relative_to(root):
            result[folder.relative_to(root).as_posix() or "."] = folder
    return result


def build_native_recording_workflow(
    voicebank: Path, *, expected_reclist: Path, max_lines: int = 8,
    bank_names: tuple[str, ...] | None = None,
    existing_reclist: Path | None = None,
) -> dict[str, Any]:
    """Check only known recording templates across requested pitch subbanks.

    An existing WAV without an OTO association is a registration review, not
    a request to record the same waveform a second time.
    """
    if type(max_lines) is not int or not 0 <= max_lines <= 10000:
        raise ValueError("max_lines must be an integer in [0,10000]")
    root = Path(voicebank).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("voicebank must be an existing directory")
    targets = read_targets(expected_reclist)
    banks = _subbanks(root)
    # A valid, author-declared subbank may contain WAVs but no oto.ini at all.
    # Do not silently mark it complete merely because OTO-based discovery skipped it.
    declared = {_bank(b) for b in (bank_names or ())}
    declared.update(bank for row in targets for bank in (row["subbanks"] or ()))
    for bank in sorted(declared):
        directory = (root if bank == "." else root / bank).resolve()
        if not directory.is_dir() or not directory.is_relative_to(root):
            raise ValueError("declared subbank directory is missing or outside the voicebank: " + bank)
        banks.setdefault(bank, directory)
    if not banks:
        raise ValueError("cannot infer target subbanks; supply explicit existing subbank names")
    wanted = sorted(banks) if bank_names is None else list(dict.fromkeys(_bank(b) for b in bank_names))
    if not wanted or any(name not in banks for name in wanted):
        raise ValueError("requested subbank does not exist in this voicebank")
    banks_missing_oto = sorted(
        bank for bank in wanted
        if not any(p.is_file() and p.name.lower() == "oto.ini" for p in banks[bank].iterdir())
    )

    from .oto import load_voicebank
    from .source_aware_recording_v17 import pcm_identity

    entries, warnings = load_voicebank(root)
    aliases: dict[tuple[str, str], set[Path]] = defaultdict(set)
    registered: set[tuple[str, Path]] = set()
    for entry in entries:
        folder = entry.oto_path.parent.resolve()
        path = entry.wav_path.resolve()
        if not folder.is_relative_to(root) or not path.is_relative_to(root):
            continue
        group = folder.relative_to(root).as_posix() or "."
        if group in wanted:
            aliases[(group, unicodedata.normalize("NFC", entry.alias))].add(path)
            registered.add((group, path))

    indexed: dict[str, dict[str, list[Path]]] = {}
    for group in wanted:
        files: dict[str, list[Path]] = defaultdict(list)
        for p in banks[group].iterdir():
            if p.is_file() and p.suffix.lower() == ".wav" and p.resolve().is_relative_to(root):
                files[unicodedata.normalize("NFC", p.name)].append(p.resolve())
        indexed[group] = files
    hashes: dict[Path, str | None] = {}

    def pcm(path: Path) -> str | None:
        if path not in hashes:
            try:
                hashes[path] = pcm_identity(path)
            except (OSError, EOFError, ValueError, wave.Error):
                hashes[path] = None
        return hashes[path]

    states, candidates, review = [], [], []
    for target in targets:
        scope = target["subbanks"] if target["subbanks"] is not None else wanted
        if any(group not in wanted for group in scope):
            raise ValueError("target requests subbank outside selected groups")
        rows = []
        for group in scope:
            matches = indexed[group].get(target["wav"], [])
            alias_paths = sorted(aliases.get((group, target["line"]), set()))
            if len(matches) > 1:
                rows.append({"bank": group, "status": "ambiguous_file_names", "pcm_sha256": None})
                continue
            if matches:
                path, how = matches[0], "exact_wav_filename"
            elif len(alias_paths) == 1:
                path, how = alias_paths[0], "exact_oto_alias"
            elif len(alias_paths) > 1:
                rows.append({"bank": group, "status": "ambiguous_oto_alias", "pcm_sha256": None})
                continue
            else:
                rows.append({"bank": group, "status": "missing_audio", "pcm_sha256": None})
                continue
            digest = pcm(path)
            if digest is None:
                status = "invalid_audio"
            elif (group, path) not in registered or path not in aliases.get((group, target["line"]), set()):
                # A WAV registered under another alias does not prove that the
                # author's requested opaque recording label is covered.
                status = "audio_present_needs_oto_review"
            else:
                status = "audio_present_registered"
            rows.append({"bank": group, "status": status, "pcm_sha256": digest, "match": how})

        missing = [r["bank"] for r in rows if r["status"] in ("missing_audio", "invalid_audio")]
        needs_review = [r["bank"] for r in rows if r["status"] in (
            "audio_present_needs_oto_review", "ambiguous_oto_alias", "ambiguous_file_names")]
        digests = [r["pcm_sha256"] for r in rows if r["pcm_sha256"] is not None]
        item = {
            "line": target["line"], "expected_wav": target["wav"], "banks": rows,
            "missing_audio_banks": missing, "registration_review_banks": needs_review,
            "existing_distinct_pcm_sources": len(set(digests)),
            "identical_pcm_in_multiple_subbanks": len(digests) > len(set(digests)),
        }
        states.append(item)
        if missing:
            candidates.append(item)
        elif needs_review:
            review.append(item)

    # Identical recorded text is one scheduling line, but may require many
    # pitch-subbank takes. Preserve original author order on equal priority.
    candidates.sort(key=lambda item: -len(item["missing_audio_banks"]))
    scheduled = set()
    if existing_reclist is not None:
        scheduled = {_name(s.strip()) for s in Path(existing_reclist).read_text(encoding="utf-8-sig").splitlines()
                     if s.strip() and not s.lstrip().startswith(("#", ";"))}
    selected, deferred = [], []
    for item in candidates:
        if item["line"] in scheduled:
            reason = "already_scheduled_not_verified_audio"
        elif len(selected) >= max_lines:
            reason = "recording_budget_reached"
        else:
            selected.append({
                **item,
                "gap_type": "source_template_audio_missing",
                "context_family": ", ".join(item["missing_audio_banks"]),
                "base_unit": "user_native_opaque_template",
            })
            continue
        deferred.append({"line": item["line"], "missing_audio_banks": item["missing_audio_banks"], "reason": reason})

    warnings_text = [
        "Only exact author-supplied WAV/OTO template coverage is checked; not phonetic correctness.",
        "Original user labels are opaque; no IPA, legal syllables or new phonemes are inferred.",
    ]
    if any(x["identical_pcm_in_multiple_subbanks"] for x in states):
        warnings_text.append("Identical PCM in multiple subbanks is not evidence of independent recordings.")
    if warnings:
        warnings_text.append(f"{len(warnings)} oto.ini parse warnings may affect source coverage.")
    if banks_missing_oto:
        warnings_text.append("Some explicitly selected subbanks lack oto.ini; WAV coverage and OTO registration must be checked separately.")
    if review or any(x["registration_review_banks"] for x in states):
        warnings_text.append("Some files need OTO/ambiguous-source review, not automatic rerecording.")
    if not selected:
        warnings_text.append("No new recording line selected does not imply complete phonetic coverage.")

    return {
        "schema": SCHEMA,
        "scope": "explicit_user_native_recording_templates_no_language_inference",
        "voicebank": {"root": str(root), "subbank_count": len(wanted), "subbanks": wanted,
                      "oto_parse_warnings": len(warnings),
                      "subbanks_missing_oto": banks_missing_oto},
        "coverage": {"expected_template_count": len(targets),
                     "subbank_template_checks": sum(len(x["banks"]) for x in states),
                     "missing_audio_checks": sum(len(x["missing_audio_banks"]) for x in states),
                     "registration_or_ambiguity_review_checks": sum(len(x["registration_review_banks"]) for x in states),
                     "template_states": states},
        "evidence": {"gaps": [], "diagnostic_recording_gaps": len(candidates),
                     "needs_other_validation": sum(bool(x["registration_review_banks"]) for x in states),
                     "interpretation": "author-provided template file coverage only"},
        "recording_plan": {"max_lines": max_lines, "selected_count": len(selected),
                           "recording_lines": [x["line"] for x in selected],
                           "selected": selected, "deferred_or_skipped": deferred,
                           "already_scheduled_count": sum(x["reason"] == "already_scheduled_not_verified_audio" for x in deferred),
                           "oto_review_without_rerecording": [
                               {"line": x["line"], "banks": x["registration_review_banks"]}
                               for x in states if x["registration_review_banks"]]},
        "warnings": warnings_text,
        "restrictions": {"no_new_phoneme_or_alias_generated": True, "no_automatic_split_merge": True,
                         "voicebank_unchanged": True, "filename_evidence_not_audio_realization_validation": True},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="phonoweave native-recording-plan")
    p.add_argument("voicebank", type=Path)
    p.add_argument("--expected-reclist", type=Path, required=True)
    p.add_argument("--max-lines", type=int, default=8)
    p.add_argument("--bank", action="append", dest="banks")
    p.add_argument("--existing-reclist", type=Path)
    p.add_argument("--export", type=Path)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    try:
        data = build_native_recording_workflow(
            a.voicebank, expected_reclist=a.expected_reclist,
            max_lines=a.max_lines, bank_names=tuple(a.banks) if a.banks else None,
            existing_reclist=a.existing_reclist)
        if a.export:
            from .recording_workflow_v18 import export_recording_workflow
            export_recording_workflow(data, a.export, voicebank=a.voicebank)
    except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    if a.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print("PhonoWeave native recording coverage (not phoneme discovery)")
        print(f"Templates: {data['coverage']['expected_template_count']} | suggested: {data['recording_plan']['selected_count']}")
        for x in data["recording_plan"]["selected"]:
            print(f"  {x['line']}: missing in {', '.join(x['missing_audio_banks'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
