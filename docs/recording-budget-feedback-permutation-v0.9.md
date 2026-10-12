# Experiment 1 v0.9 — within-context acoustic-feedback permutation

## Research question

After v0.8 established that a metadata-only coverage-first random strategy often matches or outperforms the current acoustic-adaptive policy, does *correctly associated acquired acoustic feedback* improve selection compared with sham feedback that retains the same phonetic-class conditional feature distribution?

## Prespecified intervention and replay

The original v0.8 eight-bank manifest, v0.2 five disjoint WAV folds, v0.7 full-WAV duration ceiling, original seven-component fricative acoustic distance, and original adaptive selection formula are unchanged. Authentic-feedback acquisition orders are replayed against v0.8 reference observations. A replay mismatch causes a hard failure.

For each **training fold**, permute entire existing seven-dimensional feature vectors among OTO-extracted segments within the same consonant/context cell. Preserve WAV identifiers and grouping, candidate cells/finals, segments per WAV, the cell-conditional empirical feature multisets, and each WAV's realized duration. Only the mapping between acoustic vectors and recorded segments is disrupted. Do not use held-out feature vectors to construct permutations. The sham adaptive selector receives these reassigned vectors *only after selecting the corresponding WAV*. Scoring always references true acquired WAV vectors and untouched held-out examples.

Repeat the sham assignment 100 times per fold with fixed seeds. Compare five-fold mean acoustic distances between authentic adaptive acquisition and the sham distribution at 25/50/75/100% cumulative full-WAV audio ceiling. Include 5th/95th acquisition-permutation quantiles, missing-category contribution, actual time spent and proportion of identical selection orders. Per-bank conclusions remain distinct; average only once per declared speaker. Keep v0.8 coverage-first random control as the stronger metadata-only baseline, not as a second source of acoustic feedback.

The updated scorer caches candidate-WAV distances to fixed held-out examples, and the adaptive selector incrementally retains already observed pair distances. Dedicated regression tests check that selection order is identical to the frozen original for true and sham feedback and that 100% budget outcomes coincide.

## Limitations

This is a **retrospective mechanism control**, not a deployable oracle or a prospective online sampling policy. Conditioning within phonetic cells preserves the distribution but not physical correspondence; acoustic dispersion is only a per-context scalar and may be insensitive to many shuffles. The same WAV may contain many segments, and permutation disrupts within-WAV correlations despite preserving the number of segments. Samples and randomized acquisitions are not independent singers. A stronger true-than-sham result would support, not prove, effective exploitation of correct acoustic feedback and would still require generalization to unseen speakers and synthesis-quality experiments.

## Run

```bash
python -m pytest -q
python -m phonoweave.recording_budget_feedback_permutation_cli \
  ~/Downloads/phonoweave_voicebanks.json \
  --reference ~/Downloads/phonoweave_context_control_v0.8.json \
  --repeats 100 \
  --output ~/Downloads/phonoweave_feedback_permutation_v0.9.json
```
