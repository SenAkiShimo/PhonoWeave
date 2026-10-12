"""Read-only, OTO-aligned Mandarin fricative→vowel substitution pilot.

The OTO preutterance mark is an alignment proxy, NOT a validated CV
boundary. This is a spectral discontinuity assay on spliced acoustic windows,
NOT a rendered singing-voice test or a perceptual evaluation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .audio import AudioReadError, AudioSegment, read_wav
from .mandarin import collect_observations, context_for
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps
from .splice import boundary_penalty

WINDOW_MS = 25.0
MAX_TARGETS_PER_CONTEXT = 16
MAX_DONORS_PER_TARGET = 12
SCHEMA = "phonoweave.synthesis_relevance.oto_cv_proxy.v0.1"


@dataclass(frozen=True)
class WindowSample:
    identity: str
    source_wav: str
    subbank: str
    alias: str
    base: str
    context: str
    sample_rate: int
    preutterance_ms: float
    left: AudioSegment
    right: AudioSegment


def _window(data: np.ndarray, rate: int, begin: int, end: int) -> AudioSegment:
    if begin < 0 or end > len(data) or end <= begin:
        raise AudioReadError("CV proxy window outside WAV")
    return AudioSegment(
        samples=data[begin:end], sample_rate=rate,
        start_ms=begin * 1000.0 / rate, end_ms=end * 1000.0 / rate,
    )


def _extract(root: Path, base: str) -> tuple[list[WindowSample], dict]:
    entries, warnings = load_voicebank(root)
    observations = collect_observations(
        (e for e in entries if e.wav_path.is_file()),
        affix_pairs(load_prefix_maps(root)),
    )
    cache: dict[Path, tuple[np.ndarray, int]] = {}
    grouped: dict[tuple, WindowSample] = {}
    reasons: Counter[str] = Counter()
    for obs in observations:
        if obs.base_unit != base:
            continue
        context = context_for(base, obs.final)
        if context not in ("plain", "rounded"):
            continue
        entry = obs.entry
        subbank = str(entry.oto_path.parent.relative_to(root))
        physical = entry.wav_path.resolve()
        try:
            if physical not in cache:
                cache[physical] = read_wav(physical)
            wav, rate = cache[physical]
            center_ms = entry.offset + entry.preutterance
            if entry.preutterance <= 0 or not np.isfinite(center_ms):
                raise ValueError("invalid OTO preutterance")
            mid = int(round(center_ms * rate / 1000.0))
            width = max(64, int(round(WINDOW_MS * rate / 1000.0)))
            left = _window(wav, rate, mid-width, mid)
            right = _window(wav, rate, mid, mid+width)
            # Multiple OTO aliases may point to exactly the same acoustic event.
            key = (str(physical), mid, base, context, subbank)
            if key in grouped:
                reasons["duplicate_oto_alignment"] += 1
                continue
            relative = str(physical.relative_to(root))
            identifier = sha256(f"{relative}|{mid}|{base}|{context}".encode()).hexdigest()[:16]
            grouped[key] = WindowSample(
                identity=identifier, source_wav=relative,
                subbank=subbank, alias=entry.alias, base=base,
                context=context, sample_rate=rate,
                preutterance_ms=center_ms, left=left, right=right,
            )
        except (AudioReadError, OSError, ValueError, OverflowError):
            reasons["invalid_oto_window_or_audio"] += 1
    return list(grouped.values()), {
        "oto_entries": len(entries), "oto_warnings": len(warnings),
        "candidate_events": len(grouped),
        "skipped_reasons": dict(reasons),
    }


def _donors(target: WindowSample, samples: list[WindowSample], context: str) -> list[WindowSample]:
    valid = [
        s for s in samples
        if s.context == context
        and s.subbank == target.subbank
        and s.source_wav != target.source_wav
        and s.sample_rate == target.sample_rate
    ]
    # Deterministic matching on OTO preutterance duration only; no acoustic
    # score, donor vowel or boundary penalty enters donor selection.
    valid.sort(key=lambda s: (
        abs(s.preutterance_ms - target.preutterance_ms),
        s.source_wav, s.identity,
    ))
    # Equal number of donors is used in both comparison arms per target.
    return valid[:MAX_DONORS_PER_TARGET]


def evaluate_voicebank(root: Path, *, base: str = "sh") -> dict:
    if base not in {"sh","s","x"}:
        raise ValueError("pilot only supports sh, s, x")
    root = root.expanduser().resolve()
    samples, extraction = _extract(root, base)
    rows = []
    exclusions = Counter()
    for context in ("plain", "rounded"):
        targets = sorted(
            [s for s in samples if s.context == context],
            key=lambda s:(s.subbank,s.source_wav,s.identity),
        )
        # Do not overweight voicebanks with many repeated OTO aliases.
        unique_targets = []
        used = set()
        for target in targets:
            if (target.subbank,target.source_wav) in used:
                continue
            used.add((target.subbank,target.source_wav))
            unique_targets.append(target)
        for target in unique_targets[:MAX_TARGETS_PER_CONTEXT]:
            same = _donors(target,samples,context)
            cross = _donors(target,samples,"rounded" if context=="plain" else "plain")
            count = min(len(same),len(cross))
            if count < 2:
                exclusions["insufficient_independent_donors"] += 1
                continue
            same = same[:count]
            cross = cross[:count]
            try:
                natural = boundary_penalty(target.left,target.right)
                same_scores = [boundary_penalty(d.left,target.right) for d in same]
                cross_scores = [boundary_penalty(d.left,target.right) for d in cross]
            except (AudioReadError,ValueError,FloatingPointError):
                exclusions["invalid_boundary_spectrum"] += 1
                continue
            rows.append({
                "target_event_id":target.identity,
                "target_context":context,
                "target_wav":target.source_wav,
                "subbank":target.subbank,
                "oto_alias":target.alias,
                "oto_alignment_ms":round(target.preutterance_ms,3),
                "donors_each_arm":count,
                "same_context_donor_events":[d.identity for d in same],
                "cross_context_donor_events":[d.identity for d in cross],
                "natural_proxy":round(float(natural),6),
                "matched_proxy":round(float(np.median(same_scores)),6),
                "substituted_proxy":round(float(np.median(cross_scores)),6),
                "cross_minus_matched":round(float(np.median(cross_scores)-np.median(same_scores)),6),
            })
    groups = []
    for context in ("plain","rounded"):
        group = [row for row in rows if row["target_context"]==context]
        if not group:
            groups.append({"context":context,"targets":0,"mean_cross_minus_matched":None})
        else:
            groups.append({
                "context":context,"targets":len(group),
                "mean_cross_minus_matched":round(float(np.mean([r["cross_minus_matched"] for r in group])),6),
                "median_cross_minus_matched":round(float(np.median([r["cross_minus_matched"] for r in group])),6),
                "cross_worse_target_count":sum(r["cross_minus_matched"]>0 for r in group),
            })
    return {
        "schema":SCHEMA,"base_unit":base,
        "alignment":"OTO offset+preutterance proxy, not manually labeled CV boundary",
        "window_ms":WINDOW_MS,
        "donor_selection":"same subbank, same sample rate, distinct source WAV, nearest OTO preutterance, fixed count",
        "metric":"existing splice.boundary_penalty (spectral + RMS jump)",
        "extraction":extraction,"excluded_targets":dict(exclusions),
        "context_summary":groups,"targets":rows,
        "maturity":"exploratory_diagnostic_only",
        "profile_or_inventory_modified":False,
        "limitations":[
            "No singing synthesis or crossfaded OpenUtau render is generated.",
            "OTO preutterance is a synthesis alignment parameter, not an independently validated acoustic CV boundary.",
            "Natural concatenation is taken from contiguous source audio; substituted windows are virtual hard boundaries.",
            "Matched/cross groups differ in phonetic content and recording conditions despite same subbank controls.",
            "No pitch, loudness or speaker-identity matching beyond sample rate and subbank.",
            "Targets from the same speaker and source WAV are dependent; do not treat all alias rows as independent speakers.",
            "Only window-edge discontinuity is tested, not the following vowel trajectory or perceived singing quality.",
            "No automatic split/merge decision can be inferred from this diagnostic.",
        ],
    }


def run_voicebank(root: Path, output: Path, *, base: str = "sh") -> dict:
    result=evaluate_voicebank(root,base=base)
    path=output.expanduser().resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return result
