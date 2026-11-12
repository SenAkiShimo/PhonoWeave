# PhonoWeave v4.2 — diagnosis and scoped cross-bank stability

## Evidence from author's real v4.1 upload

The uploaded `allophone_candidates.json` and `allophone_review.csv` agree on **51 phoneme-role candidates (20 C, 31 V)**. The CV/OTO reader selected **1,419** rows from **7,737** OTO entries; **1,729** candidate rows failed the provisional consonant-region criteria, **4,589** were not accepted as full CV, and there was **1** parser warning.

**All 51** candidates have `insufficient_source_template_replication`, **0 eligible source-template holdouts**, and **no K=1/2/3 predictive scores**. The most important data-structural finding: `context_source_templates` is **1 for every context of every candidate**. Many pitch banks reuse a family of WAV recording templates (for example `a.wav`, `ang.wav`). Leaving the entire template out removes the given context from training. Therefore it is mathematically impossible to draw genuine source-template-held-out claims from this corpus; v4.1 correctly abstained, and its full-data cluster memberships are not validated new allophones.

## v4.2 weak diagnostic instead of fake independent replication

The new executable accepts the **same existing voicebank**, reading **CV-only OTO timing** and never using existing VC split decisions to define the model. It groups all 36D OTO-timed C and V spectra by `(bank, author phoneme, following/preceding context, decoded PCM source)` with one vote per physical PCM. For a phoneme available in at least **three pitch/style banks** and at least two common contexts:

1. Hold out one entire **pitch bank**, not one original recording template.
2. Fit bank-centered per-context spectra using the other banks only.
3. Fit context partitions with K=1,2,3,4 *from training banks*.
4. Predict the held-out bank's same contexts, using bank-centering calculated from those held-out features, and report mean squared spectral prediction error.
5. Preserve the critical limitation: held-out bank normalization is **transductive**, repeated pitch renders are correlated, and this does **not** establish independent human phonetic recordings, phone identity, synthesis improvements, new splits or recording costs.

Result schema: `phonoweave.cv_oto_bank_stability.v4.2`. Status `CORRELATED_CROSS_BANK_PROXY_ONLY` is **not** a split approval. A source-template-disjoint holdout remains ineligible, regardless of a good cross-bank proxy number. K=4 is still only an exploratory cap rather than an assertion that there are at most four allophones.

## Run locally without changing your originals

```bash
cd ~/Downloads/PhonoWeave
git fetch origin
git switch research/cv-oto-bank-stability-v4.2
source .venv/bin/activate

python -m phonoweave.cv_oto_bank_stability_v42 \
  "$HOME/Library/OpenUtau/Singers/LynneQing_CVVChn_Mang" \
  --adapter mandarin \
  --output "$HOME/Downloads/phonoweave_cv_oto_v42"
```

For other languages, use `--adapter author --cv-lexicon /path/to/CV_lexicon.json`. That lexicon specifies *CV -> C,V sound identity*, not VC split outcomes.

Outputs are `cross_bank_proxy.json` and `cross_bank_review.csv`, and should remain private. Source WAV/OTO and reclists are never altered.

## What this can and cannot advance

**Can:** check whether a proposed acoustic context pattern is repeatable across differently pitched subbanks of this one singer, and avoid the inaccurate inference that existing OTO alias count equals independent evidence. Can guide which phoneme's original C/V spectra should be reviewed and rendered in controlled experiments.

**Cannot:** claim a voicebank needs extra recorded allophones, establish the optimal split count, validate source-template independence, or say which existing VC units are valuable. A genuine VC marginal-value experiment must render the **same target singing phrase** with and without a pre-existing VC candidate, compare continuity and perceptual quality under fair source/recording budgets, and not score alias taxonomy.

## Regression tests

`python -m pytest -q tests/test_cv_oto_allophones_v41.py tests/test_cv_oto_bank_stability_v42.py`

Synthetic tests specifically ensure v4.1 still abstains from strict source-template holdout while v4.2 can issue only a distinctly labeled *weaker* cross-bank diagnostic; they also cover insufficient common contexts, insufficient banks and read-only exports.
