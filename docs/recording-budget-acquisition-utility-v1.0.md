# Experiment 1 v1.0: acquired-evidence marginal utility prototype

## Aim

v0.8 established that metadata-only context-coverage-first random acquisition closes the large low-budget advantage of the original acoustic-adaptive rule. v0.9 found no robust cross-speaker benefit from correctly associated acoustic feedback relative to within-cell sham feedback. This **exploratory prototype** proposes an alternative selection objective. It must not be reported as an independently validated improvement because the eight development voicebanks were already inspected.

## Selection rule

First prioritize candidates adding the greatest number of yet-unrepresented consonant/context cells. Among ties and once all cells are represented, sum candidate cell marginal utilities. Per-cell count-dependent uncertainty reduction is the difference `(1+n)^-1/2 - (2+n)^-1/2` where `n` counts previously acquired **distinct WAVs containing the cell**, not aliases. Acquired WAVs contribute one centroid of the seven frozen normalized fricative feature coordinates per cell, regardless of OTO segment multiplicity. For cells with at least two acquired WAVs, multiply the uncertainty reduction by `1 + 0.5 * d/(0.15+d)` where `d` is the median nearest-neighbor distance between acquired WAV-level centroids in the same cell. Candidate finals contribute a fixed small diminishing-returns metadata term `0.02/(1+n_final)` per planned cell/final. Sum the scores across candidate cells, breaking all ties lexically. **All constants are fixed for the prototype and not fitted**.

This is a bounded **empirical uncertainty proxy**. It does not calculate an actual expected reduction in unknown test error; doing so would require a predictive model or held-out assumptions. Planned phonetic metadata is available before acquisition. Candidate WAV spectra and candidate realized duration are never seen before selection, and actual acoustic feedback is supplied only once the chosen WAV has been acquired.

## Development and evaluation

Use original v0.2 disjoint WAV test folds, the v0.7 unaltered strict duration ceiling, and the frozen held-out OTO fricative distance metric, independently for each speaker. Compare the new true-feedback utility, an otherwise identical metadata-only utility ablation, old adaptive, fixed-balanced and one coverage-first seeded random draw. This is not equivalent to the 100-repeat random-null distribution in v0.8, which remains the stronger statistical baseline.

Do not tune coefficients against these eight prior-inspected banks and call the result held-out external replication. Record per-speaker direction and the fraction of duration used. Before any confirmatory study, freeze all choices and obtain untouched independent singers, ideally measuring downstream synthesis and recording burden.

## Run

```bash
python -m pytest -q
python -m phonoweave.recording_budget_utility_eval_cli \
  ~/Downloads/phonoweave_voicebanks.json \
  --output ~/Downloads/phonoweave_acquisition_utility_v1.0.json
```
