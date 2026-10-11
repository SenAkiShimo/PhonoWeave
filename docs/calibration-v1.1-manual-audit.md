# Calibration v1.1: manual anchor window and gate audit

Observed v1.0 development recording: 116 passing code tests, 0/8
sibilant onset detections even after the overlap-safe periodicity repair.
No-RMS ablation had runs of ~31–35 frames for four occurrences; this
does *not* prove that skipping RMS would locate the correct onset.

This branch does not change the detector, threshold, or manual marks.
Instead it uses the existing manually labeled JSON to answer:

1. Is the human mark inside the detector's 660 ms window?
2. Which of the five original gates pass in the ±60 ms region of that mark?
3. Does the diagnostic-only four-gate (no RMS) candidate land near the mark?
4. Does the local signal have unusually high 700/1000 Hz energy concentration
   (possible tone leak, not definitive evidence)?

```bash
python -m pytest -q
python -m phonoweave.calibration_manual_window_audit_cli \
  ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

Writes `analysis/calibration_manual_window_audit_v1.1.json` without editing
the recording or annotation files. Only labels with status `ok` and not
flagged `outside_expected_slot` serve as gold anchors. The local signal
comparison is exploratory; use fresh held-out recordings for claims of
accuracy. Never treat no-RMS candidates as accepted detections.
