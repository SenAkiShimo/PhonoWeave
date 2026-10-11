# Calibration v2.6: frozen multi-seed noise stability

The v2.5 local result passed 155 tests and recovered 8/8 paired x/sh candidates under one deterministic 20 dB SNR additive-white-noise realization, compared with 4/8 under the strict detector. All four annotated onsets were matched (conditional MAE 25.292 ms); strict matched only one onset (conditional MAE 60.384 ms). Across nine category-labeled negative controls, neither detector produced paired CV candidates in the tested fixed or overlapping windows. This finding is development-only.

This branch repeats the **unchanged v2.5 algorithm and thresholds** using six independent random seeds per noise type and SNR, with white and pink noise at 20 and 35 dB SNR. Pink noise uses a 1/f power spectrum above 30 Hz, normalized to the same full-band RMS as white noise. The assay records paired positive coverage, manual-reference localization error, count of errors above 40 ms, and independent-recording-level negative CV triggers. Overlapping analysis windows are never interpreted as independent false-positive examples.

Run:

```bash
python -m pytest -q
python -m phonoweave.calibration_multiseed_noise_cli ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

Output: `analysis/calibration_multiseed_noise_stability_v2.6.json`.

**Limitations:** all six seeds repeat perturbations of the same original utterances, not independent biological or device acquisitions; the nominal 20 dB SNR is active-speech-referenced, not a measured real-environment SNR; pink and white share full-band RMS but have different high-band SNR. Classification statistics are descriptive, not confidence bounds on field performance. Original audio and dual-anchor labels are unchanged.
