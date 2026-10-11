# Experiment 1 protocol v0.2: disjoint-WAV held-out folds

## Motivation

The original v0.1 five-seed evaluation of a 68-WAV Mandarin fricative voicebank reused much of its held-out material. Its mean pairwise held-out Jaccard overlap was **0.79246**; across the five nominal 25% holdouts only 20 distinct WAVs appeared in evaluation and each of those appeared more than once. Adaptive acquisition exceeded deterministic balancing in all five folds at 50% and 75% acquisition budgets, but these correlated repeated-fold outcomes cannot be interpreted as five independent results.

## Prespecified protocol revision

This is a **new exploratory protocol**, not a reanalysis of the same five splits. It allocates each SHA-256-unique WAV to **exactly one of five held-out test folds**. Fold assignment is determined by a fixed seed and phoneme/context metadata without using spectral features. Expected test-fold sizes for 68 WAVs are 13/13/14/14/14, subject to the actual eligible corpus size. Class/context singleton gaps are retained and explicitly reported rather than repeatedly assigning certain WAVs to the training side.

Each iteration trains/chooses from all other folds and tests on one fold. The same three frozen policies (`fixed_balanced`, `stratified_random`, `adaptive_observed`), budgets (25/50/75/100% of available training WAVs), feature normalization, acquisition semantics, and held-out distance are used from v0.1. Each policy sees the same candidate metadata, and the adaptive policy only sees acoustic features after choosing a WAV. No threshold or policy coefficient is updated.

## Interpretation

Disjoint test folds remove **within-experiment held-out reuse**, but the training sets overlap extensively, so fold outcomes are still correlated. This evaluates a single voicebank, not speaker-independent generalization. Fold values are descriptive; 5/5 wins do not constitute statistical significance. Moreover, the test proportion is now approximately 20% rather than v0.1's 25%, so absolute means across versions are not directly comparable.

Report mean distance, per-fold paired adaptive-minus-baseline differences, coarse context coverage, absent-from-training cells, and actual WAV test assignment. The metric is normalized fricative-feature nearest-neighbor distance, not perceptual or synthesized singing quality. It remains inappropriate to claim validated recording-list efficiency from this test alone.

## Run

```bash
python -m pytest -q
python -m phonoweave.recording_budget_disjoint_cv_cli \
  "/path/to/voicebank" \
  --output ~/Downloads/phonoweave_budget_disjoint_v0.2.json
```

Both the original v0.1 report and its audit are preserved unchanged.
