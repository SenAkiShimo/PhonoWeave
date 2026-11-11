# PhonoWeave v7 — preregistered, IPA-independent validation of v6 candidate intervals

**Research-only status:** This implements a protocol and synthetic/unit tests. As of this version **no independent manual phone-boundary dataset has been scored**. No phone recognition claim, synthesis-unit split/merge, recording-list recommendation, or generalization claim is authorized by passing the tests.

## Research question

Does v6's **ordered, user-native candidate alignment** localize independently annotated onset/offsets more accurately than simple time and region-grid baselines *on the same recordings and unit subset*? Does its candidate coverage decrease when the independent pronunciation order is shuffled? A candidate label may still be wrong even when the temporal endpoints are close.

We keep existing opaque symbols (`i`, `i0`, `i2`, `ir`, `a1`, `自定义`) exactly as provided by the user. Nothing in v7 maps to IPA, infers unit sequences from OTO aliases, automatically creates user labels, or edits original WAVs/voicebanks.

## Preregister a test set **before** annotation

1. Freeze a set of WAV sources and an **independent recording/transcription script** containing the complete ordered unit list per WAV. Do not derive that sequence from `oto.ini` or from v5/v6 predictions. If the source script is unavailable or ambiguous, exclude the source from the experiment **before** inspecting outcomes and document why. Preserve PCM SHA-256.
2. Decide the evaluation unit definition before annotation. For these data, annotate each intended user-native unit's **onset and offset in milliseconds**, including a fixed convention for coarticulation and overlap. A clear ambiguous interval should be documented, not silently adjusted to an automatic boundary.
3. Freeze speaker/bank/template assignments. Report which data developed the v3/v4/v5 model and v6 alignment heuristics, and keep a separate *never-tuned* held-out recording template. **Bank separation is not speaker independence** if multiple banks belong to LynneQing. Cross-speaker claims require different, independently acquired speakers.
4. Freeze v6 `alignments.json` and the v5 `candidate_segment_unit_hints.csv` before manual-gold evaluation. **Do not tune costs after looking at gold.** Existing OTO anchor diagnostics are not manual gold.
5. Publish/store `freeze_receipt.json`'s SHA-256 outside the working output directory before scoring. The co-located receipt detects accidental edits but alone does not cryptographically prove the bundle existed before gold.

## Commands

Prepare a **blind, blank** annotation file. This command reads only the independent script, not the v6 outputs:

```bash
python -m phonoweave.native_validation_v7 prepare-gold \
  --sequences /path/to/sequences.json \
  --output /path/to/manual_gold_template.json
```

Every annotation starts with null boundaries and `PENDING_MANUAL_ANNOTATION`. Annotators must fill each unit's actual `start_ms`/`end_ms` and change the provenance field to `independent_manual_annotation` when complete; a missing/incomplete annotation must never be treated as zero error. Keep human annotations blinded to all v5/v6 predictions.

**Before looking at annotated gold** freeze all baselines and negative controls:

```bash
python -m phonoweave.native_validation_v7 freeze \
  --v5 /path/to/v5-results-or-v5.zip \
  --sequences /path/to/sequences.json \
  --predictions /path/to/NEW_v6_output \
  --shuffles 5 \
  --output /path/to/NEW_v7_frozen
```

Freeze records a receipt, SHA-256 of the exact v5 CSV, independent script and v6 prediction JSON, the original v6 candidate spans/status, two baselines, and reproducible shuffled-script diagnostics. It rejects missing/extra v6 sources, inconsistent symbols/order, PCM hash mismatch, and reused output destinations. It **never reads the gold file**.

Once manual annotations have been independently finalized and the freeze receipt has been stored, score:

```bash
python -m phonoweave.native_validation_v7 score \
  --frozen /path/to/NEW_v7_frozen \
  --gold /path/to/independent_manual_gold.json \
  --output /path/to/NEW_v7_scoring
```

This phase refuses an altered freeze bundle, changed symbols, duplicate/missing gold sources, invalid endpoints, and nonmanual provenance. For the initial confirmatory evaluation, **every preregistered WAV must have an independently annotated gold entry**; this prevents retrospective cherry-picking of a favorable subset. The output is `validation_metrics.json`.

## Baseline comparators

- **Uniform duration:** divide the overall covered extent of v5 candidate regions into equal-duration bins, one per independently scripted unit. This uses no v5 candidate *labels*; it is *not* an acoustic-only phone detector. It is defined over the *same observed candidate extent*, not the complete true utterance.
- **Equal-count acoustic-region grid:** divide the ordered v5 acoustic candidate regions into as-equal-as-possible consecutive nonempty groups, one group per unit, ignoring predicted labels. This is a **region-proposal-only** baseline. If there are fewer regions than script units, the baseline abstains rather than inventing internal boundaries. Its coverage must be reported.
- **Shuffled-script negative control:** use fixed-seed permutations of the source's independent unit sequence, run the frozen v6 aligner, record fraction with `candidate_UNVERIFIED`. A difference only suggests order sensitivity, not linguistic validity or statistical significance; all-identical-label utterances are ineligible.

These controls do **not** demonstrate autonomous unit discovery. Without an independent script, the v6 model still cannot claim to recover the full pronunciation sequence.

## Scores and interpretation

We report total gold units; candidate/abstention coverage; endpoint MAE; tolerance matches (25, 50, 100 ms); and **label-indexed** endpoint precision/recall at 50 ms (not generic event-detection F1). A candidate contributes two predicted endpoints; an abstention contributes zero predicted endpoints and two missed reference endpoints. The primary comparison is v6 vs each baseline **on exactly those units for which v6 did not abstain**. All-unit baseline scores are also given, but must not be presented as directly paired improvements.

The `paired_v6_minus_*` metric uses endpoint-mean errors for paired units. Negative numbers favor v6. Its 95% interval is a descriptive, seeded bootstrap sampled by **whole WAV source**, not independently by endpoint, and is omitted when fewer than two distinct sources are available. It does not fix selection bias, multiple-comparison issues, or unmodeled speaker dependence. `per_bank` results are descriptive and may have extremely small sample sizes.

Additional validations to perform after acquiring actual independent data:

- Annotation consistency: a second independent annotator on a masked subset, disagreements recorded before model comparison.
- Source/holdout auditing: no shared training-source recordings, repetitions, or near-duplicates across train/test. If development features are derived from the same WAVs as held-out labels, do **not** claim independent source generalization.
- Run a strict complete-WAV source audit, not just PCM hashes for v5 rows; report candidate failures and gaps in v5 input explicitly.
- Compare v6 with stronger monotonic supervised/unsupervised baselines later, with their hyperparameters fixed prior to gold scoring.
- Only after independent time-localization and synthesis A/B evidence consider proposed pronunciation subtypes or recording-list alterations.

## Known limitations

The current baselines use the v5 proposal extent, which may omit important silence, leading/trailing sounds, or unproposed regions. Thus these are **not full-utterance** duration baselines. Manual onset/offset conventions can be inherently fuzzy under coarticulation. v6 uncertainty statuses are heuristic, and a small top-two path margin is not a calibrated confidence probability. Synthetic tests of v7 check **software invariants**, not predictive accuracy on real recordings.