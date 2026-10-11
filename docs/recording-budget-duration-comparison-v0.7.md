# Experiment 1 v0.7: equal-duration-ceiling acquisition comparison

## Research question

Does the observed-feature adaptive policy improve held-out acoustic representativeness over deterministic metadata balancing and stratified random selection when all three are constrained by the **same cumulative recorded-audio-duration ceiling** rather than the same number of physical WAV files?

This is an **exploratory retrospective re-evaluation**, motivated by prior v0.5–v0.6 results. There is no prospective or perceptual-quality claim.

## Frozen protocol

- Eight declared-independent Mandarin voicebank groups, each evaluated separately.
- Exactly the v0.2 disjoint WAV-grouped five-fold partition (seed 20261011), frozen /s, sh, x/ feature extraction, fixed normalization, metadata categories, and held-out nearest-neighbor acoustic-distance metric.
- Exactly the frozen `fixed_balanced`, `adaptive_observed` and `stratified_random` acquisition ranking algorithms, with 100 seeded random repetitions per fold and seed-zero replay consistency checks.
- Budget ceilings of 25%, 50%, 75%, or 100% of the **training fold's total full-WAV audio seconds**. Every policy in a fold receives the same cap.
- WAVs are indivisible. For each algorithm's complete frozen sequence, walk in rank order until the next WAV would cross the ceiling. Stop there and **do not skip or search later candidates** to fill the gap. The current and remaining candidates are not acquired. This is a strict consecutive-prefix rule, chosen to avoid using future recording durations in a lookahead knapsack.
- Report the actual audio seconds acquired, unused seconds, eligible OTO s/sh/x acoustic observations obtained, context coverage, matched-only and penalized held-out acoustic distances, and per-bank random-null distribution.

Selection rankings do not see candidate WAV durations. The budget is applied **post-hoc**, requiring complete reference WAV lengths. The budget is a comparable observed audio-duration *ceiling*, not equal spent time, and the evaluation may systematically penalize algorithms that happen to rank a very long WAV next. This is a meaningful sequential stopping sensitivity analysis, not a validated real-time recorder.

## Interpretation limitations

A physical recorded WAV may contain several OTO-labeled phonetic segments; all are acquired together. Total WAV duration includes rests and unmodeled phones. OTO target counts are not total voicebank inventory, and no time-weighted selection is attempted. A 25% limit that cannot accommodate the next long recorded sentence may yield very little acoustic material; those outcomes must be reported, not concealed or backfilled. No algorithm weights or features are retuned to improve the result. The held-out reference remains OTO-dependent and is not synthesized-audio fidelity or phonetic interchangeability.

Preliminary results should be reported speaker-by-speaker with cost utilization as well as quality, rather than only a mean across singers. A future prospective study would require a declared duration-prediction model and a metric tied to actual synthesis.
