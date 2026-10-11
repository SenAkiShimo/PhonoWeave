# Calibration v2.3: noisy onset gate attribution

Development-session v2.2: 146/146 local tests passed after correcting an erroneously collected pytest entrypoint. In the unchanged paired detector, 20 dB SNR white-noise perturbations reduced complete positive predictions to 4/8, with only 1/4 manually referenced instances matched. Negative paired detections remained 0/9 in a fixed nominal window. The matched-only MAE of 60.384 ms is not a coverage-adjusted measure.

This branch **does not change acoustic decision thresholds**. It records independent Boolean conditions underlying the v1.4 sustained frication core:

- high-band energy rise at least 12 dB above a nominal-window baseline
- high-band fraction at least -12 dB relative to overall RMS
- absolute high-band level at least -90 dBFS
- 40 ms continuous joint persistence

For each target and scenario, it measures the number of passing frames, longest contiguous joint run, and *diagnostic-only* counterfactual persistence after excluding each gate separately. It also records the unchanged v2.1 paired CV outcome and optional feature statistics near independent manual onset marks. A single-gate ablation satisfying persistence is an attribution clue, **not a valid onset estimate or proposed replacement threshold**.

Run:
```bash
python -m pytest -q
python -m phonoweave.calibration_noise_gate_audit_cli \
  ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

Report: `analysis/calibration_noise_gate_attribution_v2.3.json`.

Limitations: the five deterministic perturbation scenarios reuse one development session, browser nominal cue times remain unverified, and acoustic gate ablation alone cannot establish positive-label correctness or false-positive risk. Prioritize follow-up changes based on observed failure categories, then validate on independent recordings.
