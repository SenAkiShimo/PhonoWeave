# Calibration v1.0 — overlap-safe periodicity experiment

Observed October 9 v0.9 real-data gate diagnostics: 8 of 8 Mandarin
sibilant occurrences unresolved. Periodicity veto passes 0–4 short frames
per occurrence, despite many frames with high-frequency frication energy.

## Confirmed implementation defect

Legacy `_periodicity` estimates autocorrelation on 14 ms frames but allows
lags leaving only **two samples** of overlap. With such a small effective
sample, normalized correlation can be spuriously close to 1 even for noise.

This experiment measures periodicity on a 30 ms frame and demands at least
half-window overlap for every tested lag. It retains old v0.6 measurements
for comparison and leaves legacy detector code untouched.

## Run

```bash
python -m pytest -q
python -m phonoweave.calibration_periodicity_repair_cli \
  ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

The new report includes per-utterance old vs repaired median periodicity,
the strict original five-gate result with repaired periodicity, and a
**diagnostic only** no-RMS ablation. It records coverage and matched-only
error versus existing manually labeled x/sh cases.

**Do not interpret increased detections as improved accuracy without
checking signed timing errors and false positives.** The original real
data and all annotations remain unchanged. Evaluate on a separate recording
session before treating improvements as validated.
