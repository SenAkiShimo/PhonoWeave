# Calibration v1.6: robustness probes (not algorithm retuning)

User validation on 2026-10-10: **131 passed**. On four same-session
manually annotated sh/x samples, v1.4 core MAE=11.746ms, v1.5 shoulder
MAE=6.997ms, P90 26.728 → 11.903ms. One sh sample improved, another
worsened. No independent generalization is established.

This branch **does not alter detector logic or fitted thresholds**.
Instead, run the exact same recordings through:
- original
- -12dB amplitude scaling
- +6dB amplitude scaling (processed samples not clipped)
- additive white noise ~35dB SNR relative to active signal
- additive white noise ~20dB SNR relative to active signal

The deterministic-noise run measures sensitivity to particular artificial
perturbations; it is **not** a real environment robustness guarantee.
Report coverage, matched-only MAE (4 eligible records), and timing shifts
relative to original detections. Four additional sh/x occurrences remain
unlabeled and are included for stability, not for timing accuracy.

```bash
python -m pytest -q
python -m phonoweave.calibration_robustness_cli \
  ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

Output: `analysis/calibration_robustness_v1.6.json`.

Before promoting any of v1.4/v1.5 to general use, obtain a distinct
recording session (preferably multiple speakers/microphones), annotate
both events, and check vowel-only / negative utterances. Keep calibration
probes separate from actual vowel-only false positive evaluation.
