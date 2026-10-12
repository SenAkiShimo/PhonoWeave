"""Balanced, source-WAV-independent CV boundary proxy pilot v0.2.

This supersedes the biased v0.1 target cap and OTO-time donor matching.
The v0.1 implementation remains available as a frozen historical control.

No OpenUtau synthesis is performed. The analyzed boundary is provisional
OTO offset + preutterance, not an adjudicated consonant-vowel transition.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
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
from .synthesis_relevance_cv_pilot import _window, WINDOW_MS, MAX_DONORS_PER_TARGET, MAX_TARGETS_PER_CONTEXT

SCHEMA = "phonoweave.synthesis_relevance.oto_cv_proxy.v0.2"


@dataclass(frozen=True)
class CVEvent:
    identity: str
    source_wav: str
    subbank: str
    alias: str
    base: str
    context: str
    sample_rate: int
    oto_preutterance_ms: float  # OTO-relative duration, used for matching
    oto_alignment_ms: float  # WAV-relative offset + preutterance, used for slicing
    left: AudioSegment
    right: AudioSegment


def _extract(root: Path, base: str) -> tuple[list[CVEvent], dict]:
    entries, warnings = load_voicebank(root)
    observations = collect_observations(
        (entry for entry in entries if entry.wav_path.is_file()),
        affix_pairs(load_prefix_maps(root)),
    )
    cache: dict[Path, tuple[np.ndarray, int]] = {}
    events: dict[tuple[str, int, str], CVEvent] = {}
    excluded = Counter()
    for obs in observations:
        if obs.base_unit != base:
            continue
        context = context_for(base, obs.final)
        if context not in ("plain","rounded"):
            continue
        entry = obs.entry
        physical = entry.wav_path.resolve()
        try:
            if not np.isfinite(entry.offset) or not np.isfinite(entry.preutterance) or entry.preutterance <= 0:
                raise ValueError("invalid OTO timing")
            if physical not in cache:
                cache[physical] = read_wav(physical)
            wave, rate = cache[physical]
            position = float(entry.offset + entry.preutterance)
            mid = int(round(position * rate / 1000.0))
            width = max(64,int(round(WINDOW_MS * rate / 1000.0)))
            left = _window(wave,rate,mid-width,mid)
            right = _window(wave,rate,mid,mid+width)
            rel = str(physical.relative_to(root))
            key = (rel,mid,context)
            if key in events:
                excluded["duplicate_event"] += 1
                continue
            event_id = sha256(f"{rel}|{mid}|{base}|{context}".encode()).hexdigest()[:16]
            events[key] = CVEvent(
                identity=event_id,source_wav=rel,
                subbank=str(entry.oto_path.parent.relative_to(root)),
                alias=entry.alias,base=base,context=context,
                sample_rate=rate,oto_preutterance_ms=float(entry.preutterance),
                oto_alignment_ms=position,left=left,right=right,
            )
        except (AudioReadError, ValueError, OSError, OverflowError):
            excluded["invalid_window_or_audio"] += 1
    return list(events.values()), {
        "oto_entries":len(entries),"oto_warnings":len(warnings),
        "candidate_events":len(events),
        "skipped_reasons":dict(excluded),
    }


def _donors(target: CVEvent, events: list[CVEvent], context: str) -> list[CVEvent]:
    """One representative per physical WAV, matching on *relative* preutterance."""
    candidates = [
        event for event in events
        if event.context == context
        and event.subbank == target.subbank
        and event.sample_rate == target.sample_rate
        and event.source_wav != target.source_wav
    ]
    candidates.sort(key=lambda item:(
        abs(item.oto_preutterance_ms-target.oto_preutterance_ms),
        item.source_wav,item.identity,
    ))
    out=[]
    seen=set()
    for event in candidates:
        if event.source_wav in seen:
            continue
        out.append(event)
        seen.add(event.source_wav)
        if len(out)==MAX_DONORS_PER_TARGET:
            break
    return out


def _balanced_targets(events: list[CVEvent], context: str, limit: int) -> list[CVEvent]:
    """Deterministic subbank round-robin with one event per WAV."""
    strata: dict[str,dict[str,CVEvent]] = defaultdict(dict)
    for event in events:
        if event.context != context:
            continue
        key=event.source_wav
        prior=strata[event.subbank].get(key)
        if prior is None or event.identity < prior.identity:
            strata[event.subbank][key]=event
    # Hash order prevents alphabetical final names from monopolizing the cap.
    queue={
        subbank:deque(sorted(by_wav.values(),key=lambda item:item.identity))
        for subbank,by_wav in strata.items()
    }
    out=[]
    while len(out)<limit and any(queue.values()):
        for subbank in sorted(queue):
            if queue[subbank] and len(out)<limit:
                out.append(queue[subbank].popleft())
    return out


def evaluate_voicebank(root:Path, *, base:str="sh") -> dict:
    if base not in {"sh","s","x"}:
        raise ValueError("pilot supports sh, s, and x")
    root=root.expanduser().resolve()
    events, extraction=_extract(root,base)
    targets=[]
    excluded=Counter()
    for context in ("plain","rounded"):
        for target in _balanced_targets(events,context,MAX_TARGETS_PER_CONTEXT):
            same=_donors(target,events,context)
            other=_donors(target,events,"rounded" if context=="plain" else "plain")
            count=min(len(same),len(other))
            if count < 2:
                excluded["insufficient_independent_WAV_donors"] += 1
                continue
            same=same[:count]
            other=other[:count]
            try:
                natural=boundary_penalty(target.left,target.right)
                a=[boundary_penalty(d.left,target.right) for d in same]
                b=[boundary_penalty(d.left,target.right) for d in other]
            except (AudioReadError,ValueError,FloatingPointError):
                excluded["invalid_boundary_spectrum"] += 1
                continue
            targets.append({
                "event_id":target.identity,
                "target_context":target.context,"subbank":target.subbank,
                "target_wav":target.source_wav,"target_alias":target.alias,
                "oto_preutterance_ms":round(target.oto_preutterance_ms,3),
                "oto_alignment_ms":round(target.oto_alignment_ms,3),
                "donors_each_arm":count,
                "same_donor_wavs":[d.source_wav for d in same],
                "cross_donor_wavs":[d.source_wav for d in other],
                "same_donor_event_ids":[d.identity for d in same],
                "cross_donor_event_ids":[d.identity for d in other],
                "natural_proxy":round(float(natural),6),
                "same_context_proxy":round(float(np.median(a)),6),
                "cross_context_proxy":round(float(np.median(b)),6),
                "cross_minus_same":round(float(np.median(b)-np.median(a)),6),
            })
    def summarize(group):
        values=[v["cross_minus_same"] for v in group]
        return {
            "targets":len(group),
            "mean_cross_minus_same":round(float(np.mean(values)),6) if values else None,
            "median_cross_minus_same":round(float(np.median(values)),6) if values else None,
            "cross_worse_targets":sum(v>0 for v in values),
        }
    contexts=[
        {"context":c,**summarize([t for t in targets if t["target_context"]==c])}
        for c in ("plain","rounded")
    ]
    strata=[
        {"subbank":subbank,"context":context,
         **summarize([t for t in targets if t["subbank"]==subbank and t["target_context"]==context])}
        for subbank in sorted({t["subbank"] for t in targets})
        for context in ("plain","rounded")
    ]
    return {
        "schema":SCHEMA,"base_unit":base,
        "protocol":"balanced_target_subbanks_true_OTO_preutterance_distinct_donor_WAVs",
        "alignment":"OTO offset+preutterance proxy, not manually labeled CV transition",
        "window_ms":WINDOW_MS,
        "donor_matching":"same subbank and sample rate; unique physical WAV per arm; nearest actual OTO preutterance duration",
        "target_sampling":"round-robin subbanks; hashed ordering within subbank; max 16 per context, one target per physical WAV",
        "extraction":extraction,
        "excluded_targets":dict(excluded),
        "context_summary":contexts,
        "subbank_summary":strata,
        "targets":targets,
        "maturity":"research_only_not_perceptually_validated",
        "speaker_profile_modified":False,
        "synthesis_inventory_modified":False,
        "limitations":[
            "OTO offset+preutterance is not a manually verified phonetic CV boundary.",
            "This compares hard-window spectral discontinuity, not an OpenUtau render or listening quality.",
            "Cross-context donors also differ in vowel phonetic content and possible timing/acoustic conditions.",
            "Target and donor events are clustered by singer, subbank, and source WAV; counts are not independent speakers.",
            "No pitch or F0 matching, time stretching, crossfading, or explicit articulatory trajectory evaluation.",
            "Reported contrast cannot by itself justify synthesis-unit split or merge decisions.",
        ],
    }


def run_voicebank(root:Path,output:Path,*,base:str="sh") -> dict:
    report=evaluate_voicebank(root,base=base)
    path=output.expanduser().resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report
