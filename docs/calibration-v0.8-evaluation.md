# PhonoWeave calibration v0.8

## Spectrogram upgrades

Manual anchor view adds switchable high-band (0–8 kHz or Nyquist) / low-band (0–2 kHz or Nyquist) spectrum views with 2 kHz-scale labels, a frequency grid, and a display magnitude guide. The visual intensity is based on the existing uncalibrated FFT magnitude mapping; **it is not an absolute sound pressure level in dB SPL**. The same pan/zoom applies to waveform and spectrogram. Labels and WAVs are unchanged by display changes.

## Evaluation: browser JSON -> acoustic model comparison

```bash
python -m phonoweave.calibration_manual_compare_cli \
  ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

Reads `analysis/calibration_manual_anchor_labels_v0.2.json` from the
browser marker, runs the established v0.3.3 paired anchor and v0.6 sibilant
detectors, then writes
`analysis/calibration_manual_comparison_v0.8.json`.

Reports:
- Eligible manual anchors (status ok, within expected beat), uncertain, out-of-slot
- Coverage: accepted detections / eligible gold anchors
- MAE, p90 absolute error, median signed error **only on matched samples**
- Per-class and per-base metrics, ten highest-error matched cases
- Rejection statuses and unresolved reasons

`paired_v0.3.3_all` is a diagnostic baseline that counts estimates even with weak QC;
`paired_v0.3.3_usable` counts only estimates with usable QC.
`sibilant_v0.6_all` counts candidate/review estimates;
`sibilant_v0.6_candidate` counts candidates only.
Because the new v0.6 detector previously produced 8 unresolved cases, its
MAE may remain None while a paired detector's MAE exists.

Development-set metrics **do not establish** cross-speaker generalization,
phonemic identity classification, or valid split/merge inventory decisions.
Do not treat a small MAE on low-coverage subsets as strong performance.

## Development

```bash
python -m pytest -q
```

The new functions have targeted test fixtures, but the full suite and actual
real-recording operation must be verified on the local Mac before merge.
