# PhonoWeave v2.8 — source-WAV-disjoint context stability check

## Why it matters

The singer confirmed that existing \`i\`, \`ir\`, \`i0\` labels refer to intentional, distinct synthesis realizations. The suitability of \`a\`, \`ei\`, and \`hw\` as new recording contexts remains **uncertain**. This module evaluates whether the already-existing, OTO-derived 12D acoustic signatures preserve those known label distinctions across older **physical source WAV templates**. It does **not** ask the author to record or annotate anything, create new sound units, or change aliases.

## Method

From the frozen v2 \`source_index.csv\` + \`feature_vectors.json\`, retain only (same subbank, same following context, same decoded PCM WAV) groups with exactly one source-linked window per author label. Leave an entire **physical source WAV template** out inside each subbank; learn label centroids from the remaining WAV templates, with equal contribution per training WAV template.

Compare two diagnostic modes:

1. Original 12D source-linked log-spectral representation.
2. **Paired contextual residual**: subtract the mean of all three known labels *within that same bank/context/source group*. This cancels common local acoustic offsets by using the **known three-part group at test time**, so it is **not a blind classifier** or a model that can infer phones from arbitrary input audio.

A training-only negative control permutes the label alignment separately for each training WAV template; it leaves each held-out original label fixed. It deliberately breaks the cross-source common mapping. No uncalibrated significance claims are made.

## Historical results (one singer, seven model-rendered pitch/style subbanks)

- Input: **7,860** OTO-relative source-linked feature rows, **140** same-PCM three-label complete context groups, **35** whole-WAV template holdout folds (five \`vcsection*.wav\` per subbank).
- Held-out known-label assignment using original features: **414/420**, with 6 errors.
- Paired contextual residual: **419/420**, with one error.
- 32 training-template label-permutation controls for the residual classifier: mean ~**0.3361** accuracy (max ~**0.4381**); descriptive check only, folds are highly dependent.
- Actual per-label residual confusion (actual rows vs centroid predictions, label order \`i\`, \`ir\`, \`i0\`): \`[[140,0,0],[0,139,1],[0,0,140]]\`.
- All seven banks come from **one** singer's historical model-rendered voicebanks. The 420 observations are correlated and are not 420 independent performances.

These figures describe **existing known-label discriminability in filtered OTO proxy windows**, not verified phone identity, real acoustic segment boundaries, full WAV rendered synthesis, perceptual continuity, or recording-budget benefit. This is complementary to v2.5's negative feasibility finding: same-context **independent donor recording comparisons** remain unavailable. High supervised classification performance does **not** authorize new recording lines or merge/split choices.

## Run

\`\`\`bash
python -m phonoweave.native_context_stability_v28 \
  --source-archive /local/PhonoWeave_User_Native_Batch_Discovery_v2.zip \
  --label i --label ir --label i0 \
  --null-repetitions 32 \
  --output /local/new_stability_review.json
\`\`\`

The output records per-bank whole-source-WAV holdout accuracy, confusion matrices, descriptive negative-control results, relevant source metadata and explicit evidence limitations. \`attach_context_review\` can add this as \`existing_native_context_stability\` to a prior author-feedback + recording-plan JSON **without changing** \`recording_plan\`.

## Policy

Preserve the existing author-confirmed three realization labels, do not infer new phones from clustered audio, and do not ask for additional recording just to improve a metric. Original WAV/OTO and singing inventory stay unchanged. Further work should integrate this check with concrete source selection and recording-plan evidence gates instead of adding gratuitous new models.
