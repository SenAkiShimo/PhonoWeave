# Calibration v2.5: noise-excess fallback evaluation

The v2.4 local evaluation passed 152 tests. For the eight target occurrences, strict paired detection is 8/8 on original recordings and 4/8 under deterministic 20 dB SNR additive noise. The median high-band peak rise is 16.951 dB for detected and 16.005 dB for undetected utterances at 20 dB SNR, suggesting that peak rise alone has limited discrimination. Power-domain excess-energy P90 differs by under approximately 0.28 dB between the original and contaminated versions of each occurrence, but it spans the entire nominal window and does not establish an onset.

This development-only alternative retains the original strict candidate whenever it exists. If strict frication fails, it estimates a local high-band background, computes nonnegative high-band excess power, and tests a sustained fraction-and-excess condition. Every alternative onset requires subsequent CV evidence with a limited plausible onset-to-CV interval. No threshold is changed in the existing detector.

The comparison reports recovered positive candidates, conditional onset MAE and matched coverage, and independently recorded category-labeled negative recordings. Negatives are checked in both the original fixed 1.25s window and overlapping 200ms windows. Overlapping-window detections are summarized at the recording level, not as independent false-positive examples.

Run:

```bash
python -m pytest -q
python -m phonoweave.calibration_excess_fallback_cli ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

Output: `analysis/calibration_excess_fallback_comparison_v2.5.json`.

All numeric parameters in the fallback are **exploratory** and have not been validated on a separate dataset. The development recordings are not a held-out generalization set. Zero negative paired triggers is not a reliable specificity estimate with nine clips. The algorithm remains a shadow candidate pending broader validation.
