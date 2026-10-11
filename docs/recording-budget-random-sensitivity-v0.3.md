# Random baseline sensitivity v0.3

## Question

Does the fixed v0.2 `adaptive_observed` strategy outperform a typical stratified-random acquisition, rather than only one random seed? This analysis is **exploratory**, designed after observing v0.2 results. It does not alter the policy or make cross-speaker claims.

## Frozen comparison

Exactly the v0.2 metadata-stratified disjoint five-fold partition is used (seed 20261011). There are 100 seeded `stratified_random` sequences **per fold**, including the original v0.2 seed as repetition zero. The existing `fixed_balanced` and `adaptive_observed` sequences are unchanged. All methods share the same 25/50/75/100% WAV acquisition budgets, candidate metadata, acquired-only features, fixed spectral normalization and held-out acoustic-distance metric.

A summary compares the adaptive macro-average over the five frozen held-out folds with the distribution of 100 macro-averaged random outcomes, each assembled from one random acquisition per fold. Report the random mean, 5th/95th percentiles, min/max and counts of draws above, equal to, or below the adaptive score; lower distance is better. Fold-specific random distributions and seeds remain in the JSON for reproducibility.

**Interpretation:** these percentiles describe selection randomness conditional on a single dataset and partition, not a confidence interval for future singers. Reusing five training pools and one singer is not a sample of 500 independent vocalists. The 100 random sequence replicates measure baseline sensitivity and must not be turned into an inferential p-value. No synthesis quality, duration-adjusted burden or external speaker validation is evaluated.

## Run

```bash
python -m pytest -q
python -m phonoweave.recording_budget_random_sensitivity_cli \
  "/path/to/voicebank" \
  --repeats 100 \
  --output ~/Downloads/phonoweave_budget_random_sensitivity_v0.3.json
```
