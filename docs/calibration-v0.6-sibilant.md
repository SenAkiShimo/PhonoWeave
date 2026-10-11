# Calibration v0.6: exploratory Mandarin sibilant onset detector

Branch: `calibration/class-specific-sibilant-onset-v0.6`.
Builds on `calibration/anchor-failure-analysis-v0.5` (not the generic recording-list importer).

## Scope

- Detect provisional sustained aperiodic frication onset for Mandarin `sh`, `x`, and (when recorded) `s`.
- Distinguish candidates, review-required candidates and unresolved cases.
- Reuse existing beep-sequence detection and subtraction, as well as the v0.3.2 repeat-alignment estimate as supporting evidence.
- Compare candidates to existing manual anchors without editing the manual-label file.
- Save a new report only: `analysis/calibration_sibilant_onset_v0.6.json`.

## Run

```bash
python -m phonoweave.calibration_sibilant_onset_cli \
  ~/Downloads/PhonoWeaveCalibration/20260829_133902_433402 --json
```

The original WAV files and session directory are necessary to report any actual accuracy. Current thresholds are exploratory; successful synthetic tests are *not* validation on the 46 recorded prompts. Do not claim to identify all Mandarin phonemes or to estimate phonological boundaries precisely.

## Evaluation and next steps

1. Run on the 2026-08-29 development session; inspect per-token errors and genuine missed cases.
2. Record exact metrics and compare with frozen manual onset labels. The old 46-prompt QC counts are **not** equivalent to phoneme recognition accuracy.
3. Revise sibilant decisions only after inspecting error classes, while retaining a held-out test recording set.
4. Continue with stops and affricates using different acoustic events; sonorants need voicing-onset logic, not this detector.
