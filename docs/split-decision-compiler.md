# Split decision directives v0.1

## Purpose

Convert a frozen **line-aligned handoff**, the same budget plan, and the neutral
synthesis-unit inventory into a reviewable, per-syllable split/merge work plan.
No alias, timing, phonemizer routing, or source WAV is modified. This is a
candidate compiler, not an automatic acoustic evaluation.

```bash
python -m phonoweave.split_decision_compiler \
  --handoff /path/to/recording_line_handoff.json \
  --budget /path/to/speaker_context_budget_plan.json \
  --units /path/to/recording_units_review.csv \
  --output /path/to/new_split_directives_directory
```

### Outputs

- `split_decision_directives.json`: lossless per-syllable/pitch instructions,
  candidate realization IDs and traceable source OTO references where present.
- `ANNOTATOR_SPLIT_DIRECTIVES.csv`: spreadsheet-safe review queue, **not** a
  canonical alias export.
- `ANNOTATOR_SPLIT_GUIDE.md`: separate instructions for locating CV onsets,
  checking internal transitions, waveform evidence and OTO segmentation.
- `TUNER_CONTEXT_CHECKS.md`: source-aware playback/transition checks; no
  approved alias routing or invented phonemizer fallback.

## Decision policy

A selected internal position receives a candidate realization only when the
upstream sequence-aligned handoff includes an OTO-linked internal context.
The candidate's consonant, declared context and inventory status must agree.
An initial syllable may receive a *lexical context cue* from the Mandarin
language module, but this cue has **no original alias or OTO assignment**.

The statuses `provisional_split_requires_review`,
`unresolved_keep_separate_pending_review`, and `not_optimized_requires_review`
reflect the frozen budget/inventory decisions. Conflicting candidates are
held for manual resolution. Unlinked syllables remain explicitly unassigned;
this is not evidence that a recording is absent.

## Production boundary

An acoustic-proxy split proposal does not establish a perceptual or synthesis
quality benefit. No new `oto.ini` numeric values, source alias names, automatic
merges, or production routing are permitted. All decisions require a separate
human-approved and engine-specific export stage. Source-bank names, local paths
and third-party corpus identity are never embedded in repository fixtures.
