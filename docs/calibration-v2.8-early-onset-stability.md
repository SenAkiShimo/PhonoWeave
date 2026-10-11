# Calibration v2.8: frozen early-frication backtrace audit

Eight independent consonant-onset reference positions were added to the existing x/sh development recording session. In v2.7, a bounded 100 ms early-frication backtrace gave onset MAE 10.497 → 12.120 ms on clean recordings, 27.196 → 16.541 ms on a 20 dB white-noise realization, and 34.573 → 16.418 ms on a 20 dB pink-noise realization. The clean condition deteriorated despite improvements under noise.

This audit holds all detector and backtrace parameters fixed and repeats white/pink additive-noise perturbations at nominal 20/35 dB SNR across six reproducible seeds. It records matched coverage, per-trial MAE and P90, individual greater-than-40-ms errors, and paired improvements/deteriorations; it also tests whether nine independently collected category-labeled negative recordings yield CV-paired candidates in overlapping windows. Backtrace only runs on an already paired baseline candidate and is never substituted into the standard prediction.

Run `python -m pytest -q` and `python -m phonoweave.calibration_early_frication_multiseed_cli ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813`.

Output: `analysis/calibration_multiseed_early_frication_v2.8.json`.

Caveats: noise seeds are correlated transformations of four existing WAV files rather than new speaker/device data; the eight target onsets are all from one development recording session; negative labels are self-reported; nominal SNR is active-speech referenced and differs in spectral distribution between pink and white noise. No production detection threshold or UI behavior changes.
