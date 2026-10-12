"""Construct an isolated temporary UTAU singer for controlled OpenUtau A/B rendering.

Both experimental aliases contain the same target post-boundary vowel audio.
Only a same-context versus cross-context pre-boundary donor is exchanged.
A third natural-control alias retains the target's original pre-boundary audio.

OTO offset+preutterance is a provisional alignment, NOT a verified CV boundary.
This builder never writes to a source singer or production inventory.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import wave

import numpy as np

from .audio import AudioReadError, read_wav
from .synthesis_relevance_cv_pilot_v02 import _extract

VERSION = "phonoweave.synthesis_relevance.rendered_ab_prep.v0.3"
ALIASES = {"natural": "PW_N", "matched": "PW_A", "cross": "PW_B"}
RIGHT_MS = 600.
MAX_LEFT_MS = 120.
CROSSFADE_MS = 8.


def _slice(samples: np.ndarray, rate: int, center_ms: float,
           before_ms: float, after_ms: float) -> np.ndarray:
    c = int(round(center_ms * rate / 1000))
    before = int(round(before_ms * rate / 1000))
    after = int(round(after_ms * rate / 1000))
    if before < 0 or after < 0 or c-before < 0 or c+after > len(samples):
        raise AudioReadError("insufficient audio around OTO coordinate")
    return np.asarray(samples[c-before:c+after], dtype=np.float64)


def _assemble(left: np.ndarray, right: np.ndarray, rate: int) -> tuple[np.ndarray, float]:
    fade = max(2, int(round(CROSSFADE_MS * rate / 1000)))
    if len(left) <= fade or len(right) <= fade:
        raise ValueError("insufficient samples for controlled crossfade")
    t = np.linspace(0., 1., fade, endpoint=False)
    middle = left[-fade:] * np.cos(0.5 * np.pi * t) + right[:fade] * np.sin(0.5 * np.pi * t)
    joined = np.concatenate([left[:-fade],middle,right[fade:]])
    # Perceptual midpoint of the overlapping junction, in joined-sample time.
    preutterance_ms = (len(left) - fade / 2) * 1000 / rate
    return joined, preutterance_ms


def _write_mono_pcm16(path: Path, samples: np.ndarray, rate: int) -> None:
    if not np.all(np.isfinite(samples)) or np.max(np.abs(samples)) > 1.0:
        raise ValueError("audio outside normalized PCM range")
    pcm = np.round(samples * 32767.).astype("<i2")
    with wave.open(str(path),"wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def choose_trial(rows: list[dict], *, context: str = "plain",
                 target_id: str | None = None,
                 subbank: str | None = None) -> dict:
    if context not in ("plain", "rounded"):
        raise ValueError("context must be plain or rounded")
    if target_id is not None:
        options = [r for r in rows if r["event_id"] == target_id]
    else:
        options = [r for r in rows if r["target_context"] == context
                   and (subbank is None or r["subbank"] == subbank)]
        options.sort(key=lambda r: (r["subbank"],r["event_id"]))
    if not options:
        raise ValueError("no eligible v0.2 target event")
    row = options[0]
    if target_id is not None and subbank is not None and row["subbank"] != subbank:
        raise ValueError("target event and subbank mismatch")
    for field in ("same_donor_event_ids","cross_donor_event_ids"):
        if not row.get(field):
            raise ValueError("trial has no donor candidates")
    return row


def stage_trial(
    source_voicebank: Path,
    pilot_json: Path,
    output_singer: Path,
    *,
    context: str = "plain",
    subbank: str | None = None,
    target_id: str | None = None,
) -> dict:
    """Create one new singer in a nonexistent output folder; never overwrite."""
    source_voicebank = source_voicebank.expanduser().resolve(strict=True)
    pilot_json = pilot_json.expanduser().resolve(strict=True)
    output_singer = output_singer.expanduser().resolve()
    if output_singer == source_voicebank or source_voicebank in output_singer.parents:
        raise ValueError("experiment output cannot be inside original voicebank")
    if output_singer.exists():
        raise FileExistsError(f"refusing to overwrite: {output_singer}")
    data = json.loads(pilot_json.read_text("utf-8"))
    if data.get("schema") != "phonoweave.synthesis_relevance.oto_cv_proxy.v0.2":
        raise ValueError("requires the corrected v0.2 research JSON")
    if data.get("base_unit") not in ("sh","s","x"):
        raise ValueError("unsupported original consonant")
    row = choose_trial(data["targets"],context=context,
                       target_id=target_id,subbank=subbank)
    events, _ = _extract(source_voicebank,data["base_unit"])
    by_id = {event.identity: event for event in events}
    try:
        target = by_id[row["event_id"]]
    except KeyError as exc:
        raise ValueError("selected target is not present in source OTO") from exc
    if target.source_wav != row["target_wav"] or target.subbank != row["subbank"]:
        raise ValueError("source OTO differs from pilot provenance")
    def choose_distinct_donors():
        for a_id in row["same_donor_event_ids"]:
            for b_id in row["cross_donor_event_ids"]:
                a,b = by_id.get(a_id),by_id.get(b_id)
                if (a is not None and b is not None
                    and len({a.source_wav,b.source_wav,target.source_wav}) == 3
                    and a.sample_rate == b.sample_rate == target.sample_rate
                    and a.subbank == b.subbank == target.subbank
                    and a.context == target.context and b.context != target.context):
                    return a,b
        raise ValueError("no distinct-WAV matched/cross donor pair")
    same,cross = choose_distinct_donors()
    rate = target.sample_rate
    left_ms = min(
        MAX_LEFT_MS,target.oto_preutterance_ms,
        same.oto_preutterance_ms,cross.oto_preutterance_ms
    )
    if left_ms < 50.:
        raise ValueError("provisional donor consonant interval < 50ms")
    expected = [target,same,cross]
    waveforms = {}
    for item in expected:
        path = (source_voicebank / item.source_wav).resolve(strict=True)
        if source_voicebank not in path.parents:
            raise ValueError("source WAV path escapes singer root")
        x,r = read_wav(path)
        if r != rate:
            raise ValueError("mismatched source sample rate")
        waveforms[item.identity] = x
    target_right = _slice(
        waveforms[target.identity],rate,target.oto_alignment_ms,0,RIGHT_MS
    )
    made = {}
    for condition, donor in (("natural",target),("matched",same),("cross",cross)):
        donor_left = _slice(
            waveforms[donor.identity],rate,donor.oto_alignment_ms,left_ms,0
        )
        audio,preutterance = _assemble(donor_left,target_right,rate)
        made[condition] = audio
    # One common safety gain: the two treatment arms are not independently
    # loudness-normalized and the exact same target-right samples are retained.
    highest = max(np.max(np.abs(w)) for w in made.values())
    common_gain = min(1.0,0.95 / highest) if highest > 0 else 1.0
    output_singer.mkdir(parents=True,exist_ok=False)
    rows=[]
    hashes={}
    for condition,samples in made.items():
        filename=f"{condition}.wav"
        _write_mono_pcm16(output_singer/filename,samples*common_gain,rate)
        hashes[condition] = _file_digest(output_singer/filename)
        # Same timing on all aliases. No prefix-map or original oto file altered.
        rows.append(
            f"{filename}={ALIASES[condition]},0,{preutterance:.3f},0,"
            f"{preutterance:.3f},{preutterance/3:.3f}"
        )
    (output_singer/"oto.ini").write_text("\n".join(rows)+"\n",encoding="utf-8")
    singer_name="PW CV substitution pilot " + row["event_id"][:8]
    (output_singer/"character.txt").write_text(
        "name="+singer_name+"\n",encoding="utf-8"
    )
    result = {
        "schema":VERSION,"singer_name":singer_name,
        "pilot_event_id":row["event_id"],"target_context":target.context,
        "pitch_subbank":target.subbank,
        "target_alias":target.alias,
        "target_relative_wav":target.source_wav,
        "donors":{
            "matched":{"event_id":same.identity,"alias":same.alias,"wav":same.source_wav},
            "cross":{"event_id":cross.identity,"alias":cross.alias,"wav":cross.source_wav}
        },
        "render_aliases":ALIASES,
        "sample_rate":rate,
        "left_ms":round(left_ms,3),
        "right_ms":RIGHT_MS,
        "crossfade_ms":CROSSFADE_MS,
        "oto_preutterance_ms":round(preutterance,3),
        "common_gain":round(common_gain,8),
        "pcm_sha256":hashes,
        "source_audio_unchanged":True,
        "pilot_only":True,
        "requires_manual_CV_boundary_review":True,
        "limitations":[
            "Target right segment is identical in A and B; only the donor left segment differs.",
            "All samples are artificial splice constructs; N is a splice-processed target-derived control.",
            "OTO offset+preutterance is provisional and may not be the phonetic CV boundary.",
            "Donors differ in phonetic vowel context and may also differ in articulation and pitch.",
            "No automatic OpenUtau rendering occurs; the renderer and sample mapping must be manually verified.",
            "Results cannot establish natural CV coarticulation or perceptual equivalence without further validation.",
        ]
    }
    (output_singer/"experiment.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    return result
