# Calibration v2.9: selective early-frication replay

## Rationale

The v2.8 report evaluates eight manually anchored x/sh occurrences on a single development session with 24 synthetic noise realizations. Unconditional early-frication backtrace improved median per-seed MAE at 20 dB SNR (white 26.446 → 16.291 ms; pink 30.885 → 15.041 ms) but worsened clean MAE (10.497 → 12.120 ms). At 35 dB SNR, aggregate gains concealed many individual regressions.

## Experimental comparison

This branch uses the **existing frozen v2.8 JSON**. It compares the unchanged paired baseline, unconditional v2.7 shadow backtrace, and a provisional selection rule: accept an earlier onset only when the existing backtrace represents at least one 20 ms frame, remains within 100 ms of baseline onset, and preserves 45–320 ms onset-to-CV timing. The rule consumes no manual anchors, class/SNR tags, or error values in its decision. Manual labels are evaluation-only.

The 20 ms criterion is an explanatory engineering hypothesis chosen after inspecting this development session, **not an independently validated or deployable threshold**. No recording, annotation, default onset, or GUI is modified.

## Run

```bash
python -m pytest -q
python -m phonoweave.calibration_selective_early_onset_cli ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

Input: `analysis/calibration_multiseed_early_frication_v2.8.json`.

Output: `analysis/calibration_selective_early_onset_v2.9.json` with SHA-256 of input JSON, per-token decisions, per-condition MAE, >40 ms errors, and paired improvement/deterioration counts. The stored v2.8 negative-control outcomes are inherited as context only; selective timestamp shifts do not change CV acceptance.

## Independent validation gate

Freeze the rule before a new session is evaluated. Obtain independently marked consonant onset and CV boundary on a new recording session, ideally including external microphone and phone capture, preserving true recording provenance. Analyze clean and actual environmental noise separately from synthetic perturbations. Score detection coverage and MAE/P90, signed bias, errors over 40 ms, fraction of harmed onsets, and incorrect cross-token onset ownership. Do not tune on the held-out results without designating a separate development split.
