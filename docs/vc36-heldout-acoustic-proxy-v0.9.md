# VC36 v0.9: first genuine train/held-out acoustic comparison

## What this delivers

This milestone advances the frozen VC36 acquisition protocol from `plan|cues|ingest|annotate|audit` to a **real held-out numerical comparison**. When the singer has recorded and reviewed all 36 independent two-syllable takes, run:

```bash
python -m phonoweave.vc36_heldout_probe_v09 \
  --manifest /experiment/vc36/manifest.json \
  --output /outside/vc36_heldout_probe_01
```

The output folder must be newly named and outside the capture/voicebank directory. It contains `heldout_acoustic_proxy.json` and `heldout_per_take.csv`; original audio, OTO and the VC36 manifest are never modified. Keep reports private: they contain speaker and source-hash details.

If `vc36 audit` is not `analysis_ready`, the comparator **fails**, rather than treating missing WAVs, unreviewed QC, reused audio, or unannotated transitions as evidence. A manual transition shorter than 40 ms is ineligible for the 5×16ms feature window.

## Exact experimental comparison

The six frozen stimuli are `li ma`, `shi ma`, `zi ma`, `li sha`, `shi sha`, `zi sha`, using the author's original `i/ir/i0` surface realization labels and two next-context labels `m/sh`.

The first recording session **S01** (18 independent takes) fits BOTH estimators:

- **Separate**: one 5×12 time-normalized spectral centroid per `(left, right)` condition from the 3 S01 repeats.
- **Pooled**: one 5×12 centroid per `right` condition using all 9 S01 takes in that right-context, ignoring the left label.

Each predictor is evaluated once, without tuning, against all 18 separate-day S02 WAVs (3 independent held-out takes for each `(left, right)`). The input region is the **human-reviewed vowel-to-following-consonant interval** only, not the full utterance, to reduce the trivial first-consonant confound. Five 16ms Hann-window normalized 12-band log power profiles are extracted at fixed relative positions across the verified interval. The report presents each S02 take's squared spectral-shape error under both estimators and `pooled_minus_separate_mse` (positive means the separate acoustic prototype was closer to this recording), along with each condition's average and the 18-take overall direction.

**Important design limit:** The separate estimator has more degrees of freedom and only 3 examples per condition, while pooling has 9; holding out S02 reduces overfitting but does not eliminate capture, prompt, alignment, vowel-identity or within-singer variability confounds. `i/ir/i0` are already author-defined distinct surface realizations: this does not authorize their merger. It does not compare newly discovered abstract phonemes or estimate how many seconds a new recording list saves.

## What counts as progress and what does not

**This is genuine train/held-out acoustic evidence** once real independent singer takes and manual reviews have been supplied. It moves beyond filenames, alias counts and same-file diagnostics. It is an **exploratory signal about whether separate spectral prototypes generalize better to later recordings**, not a sufficient `split` decision.

The final decision requires a truly matched source-budget **rendered singing** baseline (same singer, pitch, score, renderer, input recordings and acquisition time), frozen before S02, plus blinded listening/perceptual assessment where possible. This module contains no renderer, no resampler output, no automatic alias/OTO change, and never calls any split or merge approved. A positive acoustic delta alone must not be published as demonstrated synthesis improvement.

## Tests

```bash
python -m pytest -q tests/test_vc36_capture.py tests/test_vc36_heldout_probe_v09.py
```

Synthetic WAVs exercise 36-take ingest, complete manually attached boundaries, S01-only fit, S02-only loss, output reproducibility and fail-closed cases. The synthetic case with distinct per-class sine tones is an intentionally obvious *software test*, **not** human experimental evidence.
