# Context-coverage-constrained random control (v0.8)

## Motivation

In the eight-voicebank retrospective experiment at a 25% full-WAV-duration ceiling, the apparent adaptive advantage over unconstrained stratified random acquisition was concentrated in two small voicebanks where a substantial fraction of random selections omitted at least one test phonetic context. The frozen held-out metric assigns a 2.0 normalized-distance penalty to missing WAV/context pairs. The experiment asks whether an adaptive advantage persists against a **metadata-only random control that actively prioritizes context coverage**.

## Protocol

- Eight declared-independent voicebanks; v0.2 disjoint WAV-grouped folds, OTO-derived s/sh/x observations and fixed evaluation metric.
- v0.7 fixed-balanced, adaptive-observed and original stratified-random selections and 25/50/75/100% full-WAV-duration ceilings are **unchanged**, and reproduced from the supplied existing v0.7 JSON; deviations are treated as errors.
- A new random control, on each step, restricts candidates to those that cover the greatest number of training metadata cells not yet selected. Ties use the **original frozen stratified-random inverse-cell-count weights**. Once the known candidate cell inventory is represented, acquisition continues by the same original weighted random rule. Only known OTO metadata is inspected; selection does not inspect unselected WAV acoustic features or duration.
- Draw 100 independent seeded *acquisition sequences* per fold (not independent vocalists), stopping before the first WAV whose realized duration would exceed the cap, with no backfill. Full coverage is therefore prioritized but **not guaranteed under tight duration ceilings**.
- Score each selected prefix using the exact original `evaluate_heldout` definition. Separately report the missing-cell penalty contribution (missing WAV/cell pairs × 2.0 divided by all held-out WAV/cell pairs) and matched-cell distance contribution with the **same denominator**. Ordinary matched-only means are conditional and should not be compared between different coverage subsets.
- Report selected eligible segments, full WAV duration utilized, coverage completeness, and stratified random distributions across five folds, with speaker-equal aggregated comparisons.
- The JSON stores per-fold and per-draw results to enable auditing. Recording audio and local file paths are not committed.

## Interpretation

If the new control closes the 25% gap, the earlier result likely reflected phonetic metadata coverage more than acoustic adaptation. If adaptive retains a gap, inspect remaining differences in actual spent time, missed cells and class-conditional distances before attributing an acoustic mechanism. These findings are exploratory, conditional on existing voicebanks and OTO quality. The new control is not a pure fixed-reclist policy, a prospective duration predictor, or a perceptual singing-synthesis evaluation.

## Usage

```bash
python -m pytest -q
python -m phonoweave.recording_budget_context_control_cli \
  ~/Downloads/phonoweave_voicebanks.json \
  --reference ~/Downloads/phonoweave_duration_budget_v0.7.json \
  --repeats 100 \
  --output ~/Downloads/phonoweave_context_control_v0.8.json
```
