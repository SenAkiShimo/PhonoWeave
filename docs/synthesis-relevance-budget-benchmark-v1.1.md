# Synthesis relevance v1.1 — matched-budget baseline protocol

## Decision question

Does a training-data-adaptive selection of context splits outperform fixed split policies at the **same incremental inventory budget**, on source WAVs not used for selecting or estimating the split policy? Unlike the v0.5–v1.0 series, this explicitly measures **incremental benefit against competing policies**, rather than relying on whether a phonetic context is classifiable.

## Fixed study unit and eligibility

The four predeclared pair groups are a/ua, ai/uai, an/uan and ang/uang. One extra split is one additional paired-context donor identity; budget ranges 0–4, from four merged pairs to four split pairs. This is a *conceptual* inventory budget, not a verified equal microphone-time recording budget.

Within a pitch subbank, the physical-WAV holdout selection is made **before** any feature centroid or policy score is estimated. No physical WAV appears in training and held-out populations of that subbank. Each final needs at least two distinct training WAVs and one test WAV. To score a fixed four-pair budget, **all four pairs must pass this gate within one subbank**. Incomplete data produces an explicit insufficient status, not an apparently low test loss or fabricated benchmark score. This is stricter than the two-WAV-per-final structural replication baseline in v1.0.

## Policies (frozen prior to real data)

1. `fixed_lexicographic`: alphabetic four-pair priority; no training-derived ranks
2. `linguistic_rule`: a/ua > an/uan > ai/uai > ang/uang; this is a transparent *heuristic* encoding an a priori context family order, **not a validated strong phonological rule**
3. `speaker_adaptive_train_only`: rank each pair by the improvement of its two context-specific training centroids over one pooled centroid on **training WAVs only**; no held-out outcome is used in the rank

All policies choose the same number of splits and are evaluated on the *same* held-out WAV/core features. Targets within each physical WAV are averaged before the bank/subbank mean. Report all budgets 0,1,2,3,4 as curves and separately report eligibility and denominator. Only budget 1–3 can distinguish policies; both endpoints are necessarily identical.

## Measured endpoint and experimental limits

Holdout loss is RMS Euclidean distance between 48 log-normalized FFT bins of the target OTO proxy region and the mean donor signature from *training* WAVs. This is **not** actual synthesis output or rendering; the reference may include the adjacent glide/vowel. The split case uses the same-final training centroid, the merge case uses one pooled pair centroid. The target can have a different syllabic/recording-list context, meaning this does not identify consonant-only necessity.

A success on this proxy, if a complete heldout cohort exists, is only a proof-of-concept. An actual increment over a *strong* baseline requires more than the heuristic named `linguistic_rule`: externally predeclared phonetic rules, training-only learned global priorities with speaker-level holdout, multiple independent recording sessions, and budget-equal renderer evaluation. Current eight voicebanks may not meet the required replication; an `insufficient` output is expected and meaningful.

## Run

```bash
cd ~/Downloads/PhonoWeave
git fetch origin
git switch --track origin/research/synthesis-relevance-budget-benchmark-v1.1
source .venv/bin/activate
python -m pytest -q
python -m phonoweave.synthesis_relevance_budget_benchmark_cli \
  ~/.config/phonoweave/voicebanks.json \
  --base sh \
  --output ~/Downloads/phonoweave_sh_budget_benchmark_v1.1.json
```

No source files modified, no sound rendered, no listening requested.
