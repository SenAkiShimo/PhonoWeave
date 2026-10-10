"""Development-only robustness probes for v1.4 core and v1.5 shoulder.

Uses deterministic amplitude changes and additive noise, with original
time coordinates preserved. Results are *sensitivity*, not held-out accuracy.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .audio import read_wav
from .calibration_io import load_calibration_session
from .calibration_dual_labels import load_dual_labels
from .calibration_sibilant_onset import SIBILANTS
from .calibration_transition_onset import acoustic_tracks, nominal_target_cue_ms
from .calibration_frication_shoulder import detect_frication_shoulder

SCENARIOS = ("original", "gain_minus_12db", "gain_plus_6db", "noise_35db_snr", "noise_20db_snr")


def perturb(x: np.ndarray, scenario: str, rng: np.random.Generator) -> np.ndarray:
    if scenario == "original":
        return x.copy()
    if scenario == "gain_minus_12db":
        return x * 10 ** (-12 / 20)
    if scenario == "gain_plus_6db":
        return x * 10 ** (6 / 20)
    if scenario.startswith("noise_"):
        snr = float(scenario.split("_")[1].replace("db", ""))
        active = np.asarray(x[np.abs(x) >= 0.01], dtype=np.float64)
        reference = float(np.sqrt(np.mean(active ** 2))) if len(active) else float(np.sqrt(np.mean(x*x)))
        noise_std = reference / (10 ** (snr / 20))
        return x + rng.normal(0.0, noise_std, len(x))
    raise ValueError("unknown perturbation")


def robustness_session(session_dir: Path) -> dict:
    session_dir = session_dir.expanduser().resolve()
    sess = load_calibration_session(session_dir)
    labels = load_dual_labels(session_dir)["labels"]
    rows = []
    for index, rec in sorted(sess["recordings"].items()):
        prompt = sess["protocol"]["prompts"][index]
        if prompt.get("base_unit") not in SIBILANTS:
            continue
        tokens = prompt.get("tokens") or str(prompt.get("spoken_pattern", "")).split()
        places = [i for i,t in enumerate(tokens) if t == prompt.get("syllable")]
        if len(places) != 2: continue
        x, sr = read_wav(session_dir / "recordings" / rec["wav"])
        per_scenario = {}
        for scenario_idx, scenario in enumerate(SCENARIOS):
            rng = np.random.default_rng(6000 + index * 100 + scenario_idx)
            y = perturb(np.asarray(x,dtype=np.float64), scenario, rng)
            tr = acoustic_tracks(y,sr)
            per_scenario[scenario] = [
                detect_frication_shoulder(tr, nominal_target_cue_ms(token_index))
                for token_index in places
            ]
        for occurrence in (1,2):
            mark = labels.get(str(index),{}).get("occurrences",{}).get(str(occurrence),{}).get("consonant_onset")
            eligible = isinstance(mark,dict) and mark.get("status")=="ok" and not mark.get("outside_expected_slot",False) and mark.get("source")=="manual_v0.3"
            gold = float(mark["absolute_anchor_ms"]) if eligible else None
            base = per_scenario["original"][occurrence-1]
            for scenario in SCENARIOS:
                pred = per_scenario[scenario][occurrence-1]
                rows.append({
                    "base_unit":prompt["base_unit"],"context":prompt["context_family"],
                    "prompt_index":index,"occurrence":occurrence,"scenario":scenario,
                    "core_ms":pred.core_onset_ms,"shoulder_ms":pred.shoulder_onset_ms,
                    "core_shift_from_original_ms":None if pred.core_onset_ms is None or base.core_onset_ms is None else round(pred.core_onset_ms-base.core_onset_ms,3),
                    "shoulder_shift_from_original_ms":None if pred.shoulder_onset_ms is None or base.shoulder_onset_ms is None else round(pred.shoulder_onset_ms-base.shoulder_onset_ms,3),
                    "eligible_gold_ms":gold,
                    "core_error_ms":None if pred.core_onset_ms is None or gold is None else round(pred.core_onset_ms-gold,3),
                    "shoulder_error_ms":None if pred.shoulder_onset_ms is None or gold is None else round(pred.shoulder_onset_ms-gold,3),
                    "status":pred.status,
                })
    summaries = {}
    for scenario in SCENARIOS:
        matches = [r for r in rows if r["scenario"]==scenario]
        valid = [r for r in matches if r["eligible_gold_ms"] is not None]
        summaries[scenario] = {"total":len(matches),"eligible":len(valid)}
        for kind in ("core","shoulder"):
            detected = [r for r in valid if r[f"{kind}_ms"] is not None]
            shifts = [abs(r[f"{kind}_shift_from_original_ms"]) for r in matches if r[f"{kind}_shift_from_original_ms"] is not None]
            summaries[scenario][kind] = {
                "matched":len(detected),
                "mae_ms":round(float(np.mean([abs(r[f"{kind}_error_ms"]) for r in detected])),3) if detected else None,
                "median_abs_shift_from_original_ms":round(float(np.median(shifts)),3) if shifts else None,
                "max_abs_shift_from_original_ms":round(float(max(shifts)),3) if shifts else None,
            }
    report={"analysis":"frication_stress_test","version":"1.6-dev",
        "session_id":session_dir.name,
        "scenarios":list(SCENARIOS),"summary":summaries,"rows":rows,
        "caution":"Perturbation tests reuse development utterances; neither held-out validation nor real background recordings. Noise SNR is referenced to detected active amplitude and should be interpreted approximately."}
    out=session_dir/"analysis"/"calibration_robustness_v1.6.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report
