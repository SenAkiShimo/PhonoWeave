"""v3.4: same-rime, same-source context pairs for local voicebank listening.

This narrows the large context confound in v3.3: e.g. sha/shua vs
qia/que, not a new phoneme recognizer or approved split algorithm.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import wave

import numpy as np

from .accent_context_comparison_v33 import scan_voicebank

SCHEMA = "phonoweave.matched_rime_accent_audit.v3.4"
# Explicitly conservative Mandarin orthographic pairs: the same spelled
# final after removing an intervening /u/ medial. Neither segment identity
# nor phonetic nucleus equivalence is assumed from spelling alone.
RIME_PAIRS = {"a": "ua", "ai": "uai", "an": "uan",
              "ang": "uang", "o": "uo"}


def matched_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pair like sha/shua only if both OTO tokens occur in the same PCM WAV.

    Never treat matched source storage as independent repetitions. Features
    are three OTO-relative spectral snapshots, *not* isolated consonants.
    """
    grouped = defaultdict(list)
    rejected = defaultdict(int)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid source observation")
        base = row["base_unit"]
        token = row["stimulus"].lower()
        context = row["context"]
        if not token.startswith(base.lower()):
            rejected["unparseable_surface_token"] += 1
            continue
        final = token[len(base):]
        if context == "plain" and final in RIME_PAIRS:
            rime = final
        elif context == "rounded" and final in RIME_PAIRS.values():
            rime = next(k for k, v in RIME_PAIRS.items() if v == final)
        else:
            rejected["not_strict_mandarin_rime_pair"] += 1
            continue
        vector = np.asarray(row["feature"], dtype=float)
        if vector.shape != (36,) or not np.isfinite(vector).all():
            raise ValueError("invalid source feature shape")
        for k in ("bank", "original_alias", "original_wav", "source_pcm_sha256"):
            if not isinstance(row.get(k), str) or not row[k]:
                raise ValueError("invalid source provenance")
        if not isinstance(row.get("oto_line"), int) or row["oto_line"] < 1:
            raise ValueError("invalid source OTO row")
        key = (base, rime, row["bank"], row["source_pcm_sha256"], context)
        grouped[key].append(row)

    pairs = []
    quartet = defaultdict(set)
    for base, rime, bank, pcm, context in sorted(grouped):
        if context != "plain":
            continue
        plain = sorted(grouped[(base, rime, bank, pcm, "plain")],
                       key=lambda x: (x["original_alias"], x["oto_line"]))
        rounded = sorted(grouped.get((base, rime, bank, pcm, "rounded"), []),
                         key=lambda x: (x["original_alias"], x["oto_line"]))
        if not plain or not rounded:
            continue
        # Many aliases from one WAV are NOT independent samples; mean
        # once within the matched WAV/condition for descriptive comparison.
        a = np.mean([np.asarray(r["feature"]) for r in plain], axis=0)
        b = np.mean([np.asarray(r["feature"]) for r in rounded], axis=0)
        dif = (a - b) ** 2
        first, last = float(np.mean(dif[:12])), float(np.mean(dif[24:36]))
        anchor_a, anchor_b = plain[0], rounded[0]
        family = Path(anchor_a["original_wav"]).name
        quartet[(base, rime)].add(family)
        pairs.append({
            "base_unit": base, "plain_final": rime,
            "rounded_final": RIME_PAIRS[rime],
            "bank": bank, "same_source_pcm_sha256": pcm,
            "same_source_file": anchor_a["original_wav"] == anchor_b["original_wav"],
            "physical_filename_family": family,
            "plain_original_alias": anchor_a["original_alias"],
            "rounded_original_alias": anchor_b["original_alias"],
            "plain_original_wav": anchor_a["original_wav"],
            "rounded_original_wav": anchor_b["original_wav"],
            "plain_oto_line": anchor_a["oto_line"],
            "rounded_oto_line": anchor_b["oto_line"],
            "plain_aliases_collapsed": len(plain),
            "rounded_aliases_collapsed": len(rounded),
            "early_spectral_mse": round(first, 9),
            "late_spectral_mse": round(last, 9),
            "early_minus_late_mse": round(first-last, 9),
            "not_consonant_isolation": True,
        })
    # Each source-file family is reported once in the summary even if
    # the same score/recording template recurs at several pitches.
    by_base = defaultdict(list)
    for pair in pairs:
        by_base[pair["base_unit"]].append(pair)
    summaries = []
    for base in sorted(by_base):
        these = by_base[base]
        rimes = sorted({p["plain_final"] for p in these})
        families = sorted({p["physical_filename_family"] for p in these})
        summaries.append({
            "base_unit": base, "matched_orthographic_rimes": rimes,
            "distinct_source_filename_families": len(families),
            "observed_pitch_bank_comparisons": len(these),
            "mean_early_minus_late_mse": round(
                float(np.mean([p["early_minus_late_mse"] for p in these])), 9),
            "consonant_specific_or_accent_effect_proven": False,
            "recording_split_or_share": "UNRESOLVED",
        })
    return {
        "schema": SCHEMA,
        "scope": "conservative_same_rime_same_PCM_source_counterpart_audit",
        "pairs": pairs, "base_summaries": summaries,
        "matched_pair_count": len(pairs),
        "excluded_observation_reasons": dict(sorted(rejected.items())),
        "new_phonemes": [], "new_recording_lines": [],
        "automatic_alias_or_oto_changes": False,
        "limitations": [
            "Paired orthographic endings control some vowel identity but added u-medial itself changes coarticulation.",
            "OTO-relative early/late spectral snapshots are not reviewed phoneme boundaries.",
            "The same long WAV controls recording container, not independent human pronunciation repetitions.",
            "Same filename across pitch banks is not an independent singer/source recording session.",
            "Early-vs-late feature differences are descriptive, not split-benefit or singing-quality scores.",
            "Only attested exact Mandarin rime pairs are considered; missing pairs mean no conclusion.",
        ],
    }


def _clip(root: Path, bank: str, filename: str, alias: str, row_number: int,
          expected_pcm: str) -> tuple[np.ndarray, int]:
    """Find EXACT original OTO row and return a short labeled source excerpt."""
    from .audio import read_wav
    from .oto import load_voicebank
    from .source_aware_recording_v17 import pcm_identity

    root = root.resolve(strict=True)
    matches = []
    for entry in load_voicebank(root)[0]:
        folder = entry.oto_path.parent.resolve()
        path = entry.wav_path.resolve()
        if not folder.is_relative_to(root) or not path.is_relative_to(root):
            continue
        if ((folder.relative_to(root).as_posix() or ".") == bank
                and path.relative_to(root).as_posix() == filename
                and entry.alias == alias and entry.line_number == row_number):
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError("original OTO row missing or ambiguous during clip export")
    entry = matches[0]
    if pcm_identity(entry.wav_path) != expected_pcm:
        raise ValueError("original PCM source changed since analysis")
    audio, rate = read_wav(entry.wav_path)
    left = max(0, round((entry.offset - 60) * rate / 1000))
    right = min(len(audio), round(
        (entry.offset + max(entry.preutterance, 0) + 250) * rate / 1000))
    if right - left < round(.05 * rate):
        raise ValueError("original OTO excerpt too short for listening")
    return np.asarray(audio[left:right], dtype=np.float64), rate


def _write_pcm16(path: Path, audio: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "xb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def export(report: dict, output: Path, root: Path,
           *, max_listening_rimes_per_base: int = 2) -> Path:
    """Create NEW private report + original source excerpts outside voicebank."""
    root = Path(root).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("output must be outside original voicebank")
    if output.exists():
        raise FileExistsError("output directory already exists")
    if not 0 <= max_listening_rimes_per_base <= 30:
        raise ValueError("invalid listening-pair budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phonoweave-matched-", dir=output.parent) as temp:
        dest = Path(temp) / "result"
        dest.mkdir()
        with (dest / "matched_rime_pairs.csv").open("x", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            columns = ("base_unit", "plain_final", "rounded_final", "bank",
                       "plain_original_alias", "rounded_original_alias",
                       "plain_original_wav", "rounded_original_wav",
                       "early_spectral_mse", "late_spectral_mse",
                       "early_minus_late_mse", "same_source_file")
            writer.writerow(columns)
            for row in report["pairs"]:
                # csv injection guard for author text and file labels
                fields = [row[x] for x in columns]
                writer.writerow([("'" + v) if isinstance(v, str) and
                                 v.lstrip().startswith(("=", "+", "-", "@")) else v
                                 for v in fields])
        sampled: dict[str, set[str]] = defaultdict(set)
        audio_report = []
        for row in report["pairs"]:
            base, rime = row["base_unit"], row["plain_final"]
            if rime in sampled[base] or len(sampled[base]) >= max_listening_rimes_per_base:
                continue
            if not row["same_source_file"]:
                continue
            a, rate_a = _clip(root,row["bank"],row["plain_original_wav"],
                              row["plain_original_alias"],row["plain_oto_line"],
                              row["same_source_pcm_sha256"])
            b, rate_b = _clip(root,row["bank"],row["rounded_original_wav"],
                              row["rounded_original_alias"],row["rounded_oto_line"],
                              row["same_source_pcm_sha256"])
            sampled[base].add(rime)
            folder = dest / "original_source_excerpts"
            folder.mkdir(exist_ok=True)
            stem = re.sub(r"[^A-Za-z0-9_-]", "_", f"{base}_{rime}_{row['bank']}")
            prefix = f"{stem}_{sha256((row['same_source_pcm_sha256'] + rime).encode()).hexdigest()[:8]}"
            left = folder / f"{prefix}_A_plain.wav"
            right = folder / f"{prefix}_B_u_medial.wav"
            _write_pcm16(left, a, rate_a)
            _write_pcm16(right, b, rate_b)
            audio_report.append({
                "base_unit": base, "rime": rime, "bank": row["bank"],
                "a_path": left.relative_to(dest).as_posix(),
                "b_path": right.relative_to(dest).as_posix(),
                "source_excerpt_only_not_resynthesized": True,
            })
        report_to_write = {**report, "listening_source_excerpts": audio_report}
        (dest / "matched_rime_audit.json").write_text(
            json.dumps(report_to_write, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        os.rename(dest, output)
    return output


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Strict same-rime source pairing; no automatic phoneme splits")
    p.add_argument("voicebank", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base", action="append", dest="bases")
    p.add_argument("--listening-rimes-per-base", type=int, default=2)
    args = p.parse_args(argv)
    try:
        observations, scan = scan_voicebank(args.voicebank,
                                          only_bases=tuple(args.bases or ()))
        result = {**matched_pairs(observations), "source_scan": scan}
        if scan["source_mode"] != "mandarin_provisional_context_adapter":
            raise ValueError("v3.4 built-in rime rules require Mandarin adapter")
        export(result, args.output, args.voicebank,
               max_listening_rimes_per_base=args.listening_rimes_per_base)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, wave.Error) as exc:
        p.error(str(exc))
    print(json.dumps({
        "same_source_matched_pairs": result["matched_pair_count"],
        "base_units": [r["base_unit"] for r in result["base_summaries"]],
        "production_split_decisions": 0,
        "report": str(args.output)
    }, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
