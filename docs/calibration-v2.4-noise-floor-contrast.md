# High-band noise-floor contrast audit (v2.4)

The 20 dB SNR additive-noise condition rejects four of eight positive target occurrences before CV verification. Every rejection arises at the baseline-relative high-band-rise condition; omitting this gate restores Boolean persistence in all four rejected records, but this does not establish correct onset estimates.

This diagnostic computes local high-band background power and the nonnegative band-power excess over that estimate. It compares noisy and original feature distributions while retaining all onset and CV inference parameters and outputs unchanged. A counterfactual noise-floor subtraction cannot be promoted directly into a calibrated onset detector.

```bash
python -m pytest -q
python -m phonoweave.calibration_noise_floor_contrast_cli ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

Output: `analysis/calibration_noise_floor_contrast_v2.4.json`.

The nominal-window baseline may be contaminated by target speech. White-noise perturbations and single-session data cannot demonstrate real-environment generalization; independent recordings and category-labeled negative controls remain necessary before threshold selection.
