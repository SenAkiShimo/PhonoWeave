# Matched-line-count literal vs speaker-context audit (v0.1)

This is an **in-sample, read-only policy audit**. It reuses the existing
Mandarin-specific speaker-context planning heuristic, and introduces an
independent language-neutral *literal token/pair* recording-list baseline.
It makes no claim that either policy improves rendered singing.

## Run

```bash
# First, compile an existing voicebank to a NEW review pack:
python -m phonoweave.review_pack_pipeline --voicebank /path/to/voicebank --output /tmp/review_pack_a --budget 60

# Then, compute matched-line-count audit in a separate new output folder:
phonoweave compare-literal-budget --pack /tmp/review_pack_a --output /tmp/compare_a

# Optional baseline weight ablation:
phonoweave compare-literal-budget --pack /tmp/review_pack_a --output /tmp/compare_b --pair-weight 1
```

The compiler checks SHA-256 of the four required review-pack source artifacts
against its manifest and refuses symlinks escaping the pack. It recalculates the
speaker plan from the frozen blueprint, unit CSV, and internal context links,
rejecting stale/forged selections.

## What is compared?

Both algorithms see **exactly the same pitch-complete reference sequences**.
They are allocated the same **maximum line count per pitch**. Because greedy
policies may stop when they exhaust their feature universe, the final number
of selected lines can differ. Thus:

- `matched_curve` pairs the two **exact prefixes** at each common feasible
  line count `k = 0..min(selected counts)`, with equal physical WAV references
  per pitch group at each k.
- `max_budget_result` reports each policy's separate end state at the common
  *budget ceiling*. If the policies select different counts, this is **not**
  an equal-cost final comparison.
- Surface token, first-token and immediately adjacent-token type coverage is
  counted from shared reference names.
- Protected unit/pitch candidate depth is scored using the existing
  provisional speaker-context evidence and capped by the existing
  repeat-target policy. These references are **not** independent phonation
  repeats, nor automatically validated CV boundaries.

## Files

- `matched_budget_comparison.json`: selected sequences, provenance hashes,
  per-policy metrics, exactly paired prefix curve and limitations.
- `matched_curve.csv`: paired k, surface coverage and context-depth proxy.
- `literal_baseline_reclist.txt`: read-only baseline source sequence names.

No voicebank WAV, source OTO, aliases, or previous review pack is modified.
Output is non-overwriting and atomic at the report-directory level.

## Research limitations

The speaker strategy is already constructed from the existing sample bank;
there is no held-out test here, no independent cross-speaker generalization,
and no measured singing quality/perceptual endpoint. Equal line counts do
**not** guarantee equal actual recording duration. Later results should
explicitly use held-out recording sessions and matched recording **durations**.

Run `pytest -q tests/test_reclist_budget_comparison.py` and the full suite.
