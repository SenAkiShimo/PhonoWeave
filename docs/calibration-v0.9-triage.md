# Calibration v0.9 failure triage

The first real assessment of session `20261009_225541_655813` after manual labeling reported:

- 32 labels; 26 primary eligible, 6 uncertain
- Paired v0.3.3 all: 26/26 predictions, MAE 201.832 ms, p90 absolute error 391.89 ms
- Paired usable: 6/26, MAE 169.37 ms, p90 263.935 ms
- Sibilant v0.6: 0/4 eligible primary matches; all eight targets unresolved
- All eight unresolved reason: `no_sustained_aperiodic_frication`

These are observed *development-session* results, not a generalization estimate. The usable subset still has unacceptable errors. Preserve the six uncertain labels for manual review.

Before adjusting thresholds, run the unmodified v0.6 gate diagnostic report:

```bash
python -m phonoweave.calibration_sibilant_diagnostics_cli \
  ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

It separates high-frequency rise, RMS rise, spectral flatness, ZCR, and periodicity gates and counts frames satisfying all five. If each gate passes individually but not jointly, inspect temporal mismatches; if most gates fail, examine acoustic feature extraction, mic filtering and tone subtraction. Also inspect predicted-anchor signed errors versus manual anchors to distinguish early detections, vowel-onset detections and cue misalignment.

This branch does not change any detector thresholds, recordings, or manual labels. The next candidate algorithm should be based on this gate report and held-out recordings, **not** tuned until performance appears good on the same manual labels.
