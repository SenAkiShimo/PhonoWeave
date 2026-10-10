# Calibration v1.4: separate onset / CV-boundary evaluation

The dual-anchor v0.3 JSON has 32 consonant-onset marks and 32 CV-boundary
marks. The source metadata is retained: many CV marks were copied from
legacy v0.2 data and should not be described as newly independent gold.

This branch provides a provisional fricative-specific CV boundary detector.
Its features are low-band (150–1600 Hz) and high-band (2.5–8 kHz) energies,
computed with 20 ms Hann FFT and 4 ms hop. Within a detected fricative,
the algorithm uses the early-frication local baseline and searches for a
sustained low-frequency vowel-energy increase and change in spectral balance.
It is *not* a generic CV detector for stop/affricate/nasal classes.

Run from current branch:

```bash
python -m pytest -q
python -m phonoweave.calibration_dual_eval_cli \
  ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

Output: `analysis/calibration_dual_anchor_evaluation_v1.4.json`.

Reports onset MAE, CV MAE for provisional pooled sources, and CV metrics
separately for hand-placed v0.3 vs legacy-imported anchors. Also emits
per-utterance signed errors, coverage, statuses, source and manual durations.
Uncertain marks are excluded from summary metrics but kept in rows.

Development-only: limited 2 user-uploaded WAVs yielded provisional CV
candidate offsets of approximately 0–12 ms relative to old CV marks, which
were imported and are **not independent external validation**. The full
local 46-prompt session must be tested, and another session recorded to
assess generalization. No change to existing WAVs or label files.

Official releases must hide/disable the development-only legacy import
UI and endpoint (tracked in GitHub issue #13).
