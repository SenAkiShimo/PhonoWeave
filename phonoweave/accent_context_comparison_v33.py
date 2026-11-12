"""Source-disjoint, context-conditioned acoustic preview for a singer's voicebank.

Candidate contexts are hypotheses, not phonemes or approved recording units.
The first-party Mandarin adapter only proposes comparisons; an exact author
context map can provide cases for any language or notation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from hashlib import sha256
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

SCHEMA = "phonoweave.accent_context_comparison.v3.3"
MAP_SCHEMA = "phonoweave.author_context_map.v1"
BANDS = np.geomspace(220, 8000, 13)
POSITIONS = (0.20, 0.50, 0.80)


def _checked_text(value: Any) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(ord(ch) < 32 for ch in value)):
        raise ValueError("invalid original label or context")
    return value


def load_context_map(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Exact bank/OTO alias mapping, never transliteration or phoneme guessing."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema") != MAP_SCHEMA:
        raise ValueError("invalid author context map schema")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("context map requires cases")
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in cases:
        if not isinstance(row, dict):
            raise ValueError("malformed author context case")
        bank = _checked_text(row.get("bank"))
        alias = _checked_text(row.get("original_alias"))
        unit = _checked_text(row.get("base_unit"))
        context = _checked_text(row.get("context"))
        stimulus = _checked_text(row.get("stimulus", alias))
        if bank != "." and (bank.startswith("/") or "\\" in bank
                            or any(part in ("", ".", "..") for part in bank.split("/"))):
            raise ValueError("invalid author subbank")
        key = bank, alias
        if key in result:
            raise ValueError("duplicate author context alias")
        result[key] = {"base_unit": unit, "context": context,
                       "stimulus": stimulus}
    return result


def _profile(samples: np.ndarray, rate: int, start_ms: float, end_ms: float) -> list[float]:
    """Three spectral snapshots from a provisional OTO consonant-region crop.

    The crop is NOT an independently verified phoneme boundary. All features
    are sample-normalized, with no heldout/test-bank fitted normalization.
    """
    if (rate < 16000 or not all(math.isfinite(float(x)) for x in (start_ms, end_ms))
            or end_ms - start_ms < 75):
        raise ValueError("insufficient provisional consonant region")
    n = max(64, round(rate * .020))
    freq = np.fft.rfftfreq(n, 1.0 / rate)
    win = np.hanning(n)
    out = []
    for pos in POSITIONS:
        center = start_ms + (end_ms - start_ms) * pos
        begin = round(center * rate / 1000) - n // 2
        if begin < 0 or begin + n > len(samples):
            raise ValueError("OTO crop extends outside WAV")
        frame = samples[begin:begin+n].astype(np.float64, copy=False)
        if np.sqrt(np.mean(frame * frame)) < 0.0001:
            raise ValueError("silent candidate spectrum")
        power = np.abs(np.fft.rfft((frame - frame.mean()) * win)) ** 2
        bands = []
        for lo, hi in zip(BANDS[:-1], BANDS[1:]):
            mask = (freq >= lo) & (freq < hi)
            if not np.any(mask):
                raise ValueError("sample rate cannot resolve spectral bands")
            bands.append(math.log10(float(np.mean(power[mask])) + 1e-10))
        vec = np.asarray(bands)
        vec -= vec.mean()
        out.extend(vec.tolist())
    if len(out) != 36 or not all(math.isfinite(v) for v in out):
        raise ValueError("bad candidate spectrum")
    return out


def scan_voicebank(root: Path, *, context_map: Path | None = None,
                   only_bases: tuple[str, ...] = ()) -> tuple[list[dict], dict]:
    """Read original OTO/WAV only; reject out-of-bank paths and unreadable audio."""
    from .audio import read_wav
    from .oto import load_voicebank
    from .source_aware_recording_v17 import pcm_identity

    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("voicebank must be a directory")
    entries, warnings = load_voicebank(root)
    user_cases = load_context_map(context_map) if context_map else None
    if user_cases is None:
        from .mandarin import collect_observations, context_for
        from .prefixmap import affix_pairs, load_prefix_maps
        observations = collect_observations(entries, affix_pairs(load_prefix_maps(root)))
        assignments = [
            (o.entry, o.base_unit, context_for(o.base_unit, o.final), o.alias_token)
            for o in observations
        ]
    else:
        seen_cases = set()
        assignments = []
        for entry in entries:
            folder = entry.oto_path.parent.resolve()
            if not folder.is_relative_to(root):
                continue
            bank = folder.relative_to(root).as_posix() or "."
            key = (bank, entry.alias)
            if key in user_cases:
                case = user_cases[key]
                seen_cases.add(key)
                assignments.append((entry, case["base_unit"], case["context"], case["stimulus"]))
        if seen_cases != set(user_cases):
            raise ValueError("author context map references missing bank/OTO aliases")

    @lru_cache(maxsize=4)
    def cached_wav(path: Path) -> tuple[np.ndarray, int]:
        return read_wav(path)

    pcm_cache: dict[Path, str] = {}
    rows: list[dict] = []
    skipped = defaultdict(int)
    for entry, base, context, stimulus in assignments:
        if context is None or (only_bases and base not in only_bases):
            continue
        folder = entry.oto_path.parent.resolve()
        source = entry.wav_path.resolve()
        if not folder.is_relative_to(root) or not source.is_relative_to(root):
            skipped["outside_voicebank"] += 1
            continue
        if not source.is_file():
            skipped["missing_source_wav"] += 1
            continue
        try:
            samples, rate = cached_wav(source)
            if source not in pcm_cache:
                pcm_cache[source] = pcm_identity(source)
            start = max(0.0, float(entry.offset))
            end = float(entry.offset + entry.preutterance)
            if end - start < 75:
                raise ValueError("short provisional crop")
            trim = min(10.0, 0.08 * (end - start))
            feature = _profile(samples, rate, start + trim, end - trim)
        except (OSError, ValueError, RuntimeError, EOFError):
            skipped["unusable_acoustic_region"] += 1
            continue
        rows.append({
            "bank": folder.relative_to(root).as_posix() or ".",
            "base_unit": base,
            "context": context,
            "original_alias": entry.alias,
            "stimulus": stimulus,
            "original_wav": source.relative_to(root).as_posix(),
            "oto_line": entry.line_number,
            "source_pcm_sha256": pcm_cache[source],
            "feature": feature,
        })
    return rows, {
        "voicebank_root": str(root),
        "source_mode": "author_exact_context_map" if context_map else "mandarin_provisional_context_adapter",
        "original_oto_entries": len(entries),
        "oto_parse_warnings": len(warnings),
        "usable_candidate_regions": len(rows),
        "abstentions": dict(sorted(skipped.items())),
    }


def _aggregate(rows: list[dict]) -> dict[tuple[str, str, str, str], dict]:
    """One independent source-group weight per bank/context/PCM, not per OTO row."""
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        for key in ("bank", "base_unit", "context", "source_pcm_sha256",
                    "original_alias", "original_wav", "stimulus"):
            _checked_text(row.get(key))
        vector = np.asarray(row["feature"], dtype=np.float64)
        if vector.shape != (36,) or not np.isfinite(vector).all():
            raise ValueError("expected finite 36-dimensional spectrum")
        grouped[(row["bank"], row["base_unit"], row["context"],
                 row["source_pcm_sha256"])].append(row)
    result = {}
    for key, observed in grouped.items():
        result[key] = {
            "feature": np.mean([np.asarray(r["feature"], dtype=np.float64)
                                for r in observed], axis=0),
            "source": observed[0]["original_wav"],
            "alias": observed[0]["original_alias"],
            "stimulus": observed[0]["stimulus"],
            "oto_rows_collapsed": len(observed),
        }
    return result


def _evaluate_bank(sources: dict[str, dict[str, dict]]) -> dict:
    """Leave one entire decoded PCM group out; both arms get identical donors."""
    contexts = sorted(sources)
    assert len(contexts) == 2
    pcm_ids = sorted(set(sources[contexts[0]]) | set(sources[contexts[1]]))
    heldout = []
    for test_pcm in pcm_ids:
        training = {
            c: [row["feature"] for pcm, row in sources[c].items() if pcm != test_pcm]
            for c in contexts
        }
        if any(len(training[c]) < 2 for c in contexts):
            continue
        # One source = one vote. Shared estimator uses balanced context means.
        centers = {c: np.mean(training[c], axis=0) for c in contexts}
        pooled = (centers[contexts[0]] + centers[contexts[1]]) * 0.5
        for c in contexts:
            if test_pcm not in sources[c]:
                continue
            target = sources[c][test_pcm]["feature"]
            separate = float(np.mean((target - centers[c]) ** 2))
            shared = float(np.mean((target - pooled) ** 2))
            heldout.append((test_pcm, c, shared - separate))
    by_context = {c: {x[0] for x in heldout if x[1] == c} for c in contexts}
    eligible = all(len(by_context[c]) >= 2 for c in contexts)
    return {
        "source_groups": {c: len(sources[c]) for c in contexts},
        "eligible_heldout_source_groups": {c: len(by_context[c]) for c in contexts},
        "heldout_cases": len(heldout) if eligible else 0,
        "mean_shared_minus_separate_mse": (
            round(float(np.mean([x[2] for x in heldout])), 9) if eligible else None
        ),
        "positive_heldout_cases": sum(x[2] > 0 for x in heldout) if eligible else None,
        "status": "exploratory_acoustic_comparison" if eligible else "insufficient_independent_PCM_groups",
    }


def compare_contexts(rows: list[dict]) -> dict:
    """Generate every attested pair within an authored base; never invent units."""
    units = _aggregate(rows)
    by_unit: dict[str, set[str]] = defaultdict(set)
    for bank, base, context, pcm in units:
        by_unit[base].add(context)

    candidates = []
    for base in sorted(by_unit):
        for a, b in combinations(sorted(by_unit[base]), 2):
            bank_names = sorted({key[0] for key in units
                                 if key[1] == base and key[2] in (a, b)})
            per_bank = {}
            for bank in bank_names:
                pair = {
                    c: {pcm: data for (bank_id, unit, context, pcm), data in units.items()
                        if bank_id == bank and unit == base and context == c}
                    for c in (a, b)
                }
                if all(pair.values()):
                    per_bank[bank] = _evaluate_bank(pair)
            validated = [r["mean_shared_minus_separate_mse"] for r in per_bank.values()
                         if r["mean_shared_minus_separate_mse"] is not None]
            delta = round(float(np.mean(validated)), 9) if validated else None
            sources = {}
            for c in (a, b):
                options = sorted(
                    ((bank, pcm, data) for (bank, unit, context, pcm), data in units.items()
                     if unit == base and context == c),
                    key=lambda x: (x[0], x[2]["source"], x[2]["alias"]),
                )
                sources[c] = [{
                    "bank": bank, "original_alias": data["alias"],
                    "original_wav": data["source"], "stimulus": data["stimulus"],
                } for bank, _, data in options[:3]]
            candidates.append({
                "base_unit": base,
                "contexts": [a, b],
                "bank_results": per_bank,
                "eligible_bank_count": len(validated),
                "mean_shared_minus_separate_mse": delta,
                "acoustic_direction": (
                    "context_conditioned_profile_closer" if delta is not None and delta > 0 else
                    "shared_profile_closer" if delta is not None and delta < 0 else
                    "indeterminate_or_insufficient_evidence"),
                "example_original_recordings": sources,
                "production_split_or_merge_decision": "UNRESOLVED",
                "new_alias_authorized": False,
                "new_recording_authorized": False,
            })
    candidates.sort(key=lambda x: (
        x["mean_shared_minus_separate_mse"] is None,
        -(x["mean_shared_minus_separate_mse"] or 0),
        x["base_unit"], x["contexts"],
    ))
    return {
        "schema": SCHEMA,
        "scope": "speaker_specific_context_candidate_screen_not_phonetic_or_synthesis_validation",
        "candidate_count": len(candidates),
        "candidate_pairs": candidates,
        "new_phonemes": [],
        "new_recording_lines": [],
        "source_voicebank_modified": False,
        "limitations": [
            "Mandarin context labels are hypotheses, not accents, phone classes or split approvals.",
            "OTO-derived comparison regions are not independently verified consonant boundaries.",
            "Source-disjoint PCM hashes do not prove independent human takes or distinct singers.",
            "Context-dependent vowel coarticulation can cause differences without useful synthesis splits.",
            "Spectral reconstruction loss is not actual concatenative/renderer quality or listening quality.",
            "No automatic renaming, unit splitting/merging or new recording decisions.",
        ],
    }


def _csv_safe(value: object) -> str:
    text = str(value)
    if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")):
        return "'" + text
    return text


def export(report: dict, dest: Path, root: Path) -> Path:
    """One newly-created, private output folder, outside the input soundbank."""
    root = Path(root).expanduser().resolve(strict=True)
    dest = Path(dest).expanduser().resolve()
    if dest.is_relative_to(root):
        raise ValueError("output must be outside the source voicebank")
    if dest.exists():
        raise FileExistsError("output directory already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phonoweave-accent-", dir=dest.parent) as temp:
        stage = Path(temp) / "report"
        stage.mkdir()
        (stage / "context_candidates.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        columns = ("base_unit", "context_a", "context_b", "eligible_banks",
                   "mean_shared_minus_separate_mse", "acoustic_direction",
                   "example_a", "example_b", "production_decision")
        with (stage / "context_review_sheet.csv").open("x", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(columns)
            for row in report["candidate_pairs"]:
                a, b = row["contexts"]
                examples = row["example_original_recordings"]
                writer.writerow([_csv_safe(x) for x in (
                    row["base_unit"], a, b, row["eligible_bank_count"],
                    row["mean_shared_minus_separate_mse"], row["acoustic_direction"],
                    " | ".join(x["stimulus"] + " [" + x["original_alias"] + "]"
                               for x in examples[a]),
                    " | ".join(x["stimulus"] + " [" + x["original_alias"] + "]"
                               for x in examples[b]),
                    "UNRESOLVED",
                )])
        os.rename(stage, dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Speaker/context-conditioned acoustic hypothesis preview (NOT automatic phone splitting)"
    )
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--author-context-map", type=Path)
    parser.add_argument("--base", action="append", dest="bases")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows, metadata = scan_voicebank(
            args.voicebank, context_map=args.author_context_map,
            only_bases=tuple(args.bases or ()),
        )
        report = {**compare_contexts(rows), "scan": metadata}
        export(report, args.output, args.voicebank)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "candidate_pairs": report["candidate_count"],
        "with_source_disjoint_acoustic_evidence": sum(
            r["eligible_bank_count"] > 0 for r in report["candidate_pairs"]),
        "output": str(args.output), "production_decision": "UNRESOLVED",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
