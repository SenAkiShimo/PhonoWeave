# Calibration v3.0 — independent-session evaluation

The v2.9 selective rule accepts a backward onset only when an existing frication backtrace spans at least 20 ms and preserves CV temporal consistency. In the existing development session and correlated synthetic noise realizations, 42/200 candidate timestamps were adjusted without an observed increase in absolute error. This is not an independent test: the 20 ms operating point was chosen after examining the development recordings.

This branch freezes the existing detector and selective timing decision, evaluates **new sessions** using separate manual consonant-onset and CV-boundary labels, and rejects exact SHA-256 WAV matches against the development four-recording set. It records capture-device and environment categories supplied through CLI arguments. Missing labels are explicitly excluded from conditional MAE and remain reflected in coverage.

Run only after collecting a different session with the same x/sh prompts (indices 13–16 in the 46-prompt protocol) and independently marking both boundaries:

```bash
python -m phonoweave.calibration_independent_session_cli /path/to/NEW_SESSION \
  --development-session ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813 \
  --device phone --environment quiet
```

Change device/environment only to match the acquisition conditions. Evaluate subsequent external-microphone and noise sessions independently instead of pooling results. Report the 8 expected onsets, manually marked coverage, conditional onset MAE/P90, >40 ms errors, signed bias, CV-boundary coverage and error, and relative changes for all three frozen methods.

**Limitations:** SHA-256 does not detect re-encoded duplicates; the category metadata is manually declared; nominal cue-time synchronization is not sample-accurate; a different session by the same speaker does not establish cross-speaker generalization. This test must not be used to tune the frozen operating point unless the evaluation set is redesignated as development data and another held-out set is reserved.
