# Calibration v2.1: frication and CV consistency

A 48kHz breath control WAV (2.987s) triggered the v1.4 frication detector at 1344ms. It contains a sustained breath noise burst beginning at that location, with typical 1.4–1.6s median full-band RMS approximately -43.1dBFS and 2.5–10kHz energy approximately -51.4dBFS. The onset detector is detecting the acoustic burst but cannot distinguish it from target frication by the initial event alone.

The existing CV detector returns no sustained vowel transition on this breath control. This branch evaluates a **two-stage candidate**: sustained high-band frication onset (strict core + optional shoulder) followed by a plausible vowel transition. Missing CV evidence is classified as `review` (onset candidate retained), not an automatic negative or a reason to discard a possible isolated fricative.

Run:
```bash
python -m pytest -q
python -m phonoweave.calibration_cv_consistency_cli ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

A common report compares the same v1.5 positive x/sh occurrences with the nine independently recorded, self-labeled negative controls. Separate positive coverage and onset MAE from negative paired-trigger counts. **This is a single-speaker development experiment, not held-out validation or a production classifier.** The negative evaluation uses a fixed 1.25s reference window for controlled comparison; do not present it as full-file specificity.

The original audio, dual labels, strict onset detector, shoulder detector and CV detector remain unchanged.
