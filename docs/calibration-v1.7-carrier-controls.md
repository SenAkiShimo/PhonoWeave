# v1.7: Carrier-window frication trigger audit

v1.6 actual session results: 134 tests passed. At synthetic 20dB
noise, matched anchor coverage drops to 1/4 and matched-only MAE
rises to 92.384ms (core) / 60.384ms (shoulder). The model cannot
be called robust at this level.

This probe reuses the full calibration session and applies the existing
v1.4 core/v1.5 shoulder to positions of `a` carrier tokens in the protocol.
It reports how often frication candidates appear near those tokens
under original, 35dB SNR and 20dB SNR artificial-noise conditions.

**IMPORTANT**: These are *adversarial carrier windows*, not isolated
pure-vowel negative examples. They can contain a preceding target's
frication and nominal browser clock alignment may drift. Their trigger
fraction is not a valid false-positive rate. Do not tune the detector to
minimize it without reviewing sampled recordings and true vowel-only
negative examples.

```bash
python -m pytest -q
python -m phonoweave.calibration_carrier_controls_cli \
  ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

Outputs `analysis/calibration_carrier_negative_controls_v1.7.json`,
leaves the WAV and manual JSON untouched. Standalone /a/, breathing,
silence and environmental-noise negative examples still need separate
recordings for reliable specificity claims.
