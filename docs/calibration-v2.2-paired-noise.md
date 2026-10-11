# Paired CV noise validation

The v2.1 local test passed 144 unit tests. Eight target x/sh occurrences all produced paired candidates. All four eligible onset anchors were matched with 6.997ms MAE; zero of nine independent category-labeled negative controls produced complete onset-plus-CV candidates. The breath recording produced onset-only evidence and entered review.

This experiment runs the **unchanged paired detector** across original, ± amplitude and deterministic additive white-noise cases on the same positive recordings and negative controls. For each noise condition the output includes paired-positive coverage, matched-only onset MAE, and negative paired-trigger counts. No detector threshold is retuned.

Run:

```bash
python -m pytest -q
python -m phonoweave.calibration_paired_noise_cli ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

Output: `analysis/calibration_paired_noise_v2.2.json`.

Limitations: the positive recordings are the original development session, not held out. Negative annotations are self-reported and evaluated at the fixed 1250ms reference. Artificial gain/noise perturbations cannot establish performance in independent real environments or on external microphones. Report detection coverage jointly with matched-only MAE, especially when noise causes abstention.
