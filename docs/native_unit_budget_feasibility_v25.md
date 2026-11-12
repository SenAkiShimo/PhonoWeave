# PhonoWeave v2.5 — equal-budget existing-unit comparison, with strict feasibility gate

## Why this belongs in PhonoWeave

The goal is to decide which **existing speaker-specific recording units** are worth retaining as distinct synthesis choices, not to create an endless sequence of acoustic models.

This step reuses frozen source-linked acoustic evidence to ask whether two already-named realizations can be compared under a fair physical-recording-source budget:

- **Preserve names**: author-defined \`i\`, \`ir\`, \`i0\` are opaque identifiers; they are never converted to IPA, combined, or renamed.
- **Two arms, same donor material**: for an eligible held-out physical PCM source containing both target labels in the same bank/context, train a separate-label centroid and a shared-label centroid from **the exact same donor acoustic observations**. Count only unique decoded PCM hashes, not OTO alias occurrences.
- **No leakage**: exclude the held-out target PCM source from both arms; choose donors by stable source ID, not by inspecting target spectral values. Never mix source data from different subbanks or different following contexts.
- **Conservative cost interpretation**: the donor budget is equal across both arms; the experiment measures a **feature reconstruction proxy**, not actual recording minutes or rendering. The merged model has fewer logical unit IDs but still uses the same training audio amount.
- If any sample/source condition fails, return \`insufficient_independent_source_repetitions\` with source shortages and **no score or new recording request**.
- Results are attached to the existing \`recording_plan.json\` as \`existing_unit_budget_feasibility\`. Existing \`supplement_reclist.txt\`, source-review lists, aliases, audio and synthesis inventory remain untouched.

## Command

\`\`\`bash
phonoweave plan-recording /local/voicebank \
  --native-reclist /local/original_targets.json \
  --unit-budget-archive /local/PhonoWeave_User_Native_Batch_Discovery_v2.zip \
  --unit-pair i ir --unit-pair i i0 --unit-pair ir i0 \
  --donor-budget-per-label 1 \
  --max-lines 8 --json
\`\`\`

All optional analysis flags are independent of coverage-based supplement selection. The \`--unit-budget-archive\` and at least one \`--unit-pair A B\` must be supplied together. Repeated \`--unit-evidence\` can add v2.4's previous separability review without authorizing a new recording.

## Real retrospective LynneQing source audit

Source: frozen \`results/lynneqing_seven_subbanks/source_index.csv\` and \`feature_vectors.json\` in \`PhonoWeave_User_Native_Batch_Discovery_v2.zip\`; archive SHA-256 \`70cf762b807dc1b07ef18f3383c6d67390d0683687b39cb6b214c0dd1f7207b5\`.

- **7,860** OTO-relative acoustic windows from **175** distinct source PCM WAV hashes across seven pitch/style subbanks of **one** singer.
- Matched \`(subbank, following context)\` cohorts containing both pre-existing labels number **294** for each pair.
- For each pair, **zero** eligible held-out target PCM sources with at least one other independent same-label donor for **both** labels in that exact cohort.

| Existing pair | Both-label bank-context cohorts | Eligible strictly held-out comparisons | Status |
| --- | ---: | ---: | --- |
| \`i\` / \`ir\` | 294 | 0 | Insufficient independent PCM repetitions |
| \`i\` / \`i0\` | 294 | 0 | Insufficient independent PCM repetitions |
| \`ir\` / \`i0\` | 294 | 0 | Insufficient independent PCM repetitions |

Many cohorts contain one source per label (sometimes the same long WAV); a few include two physical sources for one label but only one for the other. Reusing target WAV or treating repeated OTO entries as independent would create a **false positive**. The zero means **insufficient replication**, not zero perceptual difference or evidence that combining labels is safe.

## Validation / scientific limits

- The archive contains **processed 12D source-linked OTO-relative acoustic features**, not gold phone boundaries or original full WAV audio. This module has no renderer or listening panel.
- Source-PCM hash uniqueness is only a digital independence proxy, not evidence of independent human sessions. Different pitch subbanks are not independent speakers.
- If future independent donors exist, the held-out equal-donor comparison yields exploratory spectral reconstruction error only; any actual merge/split needs full original source-WAV verification, equal *recording-time* comparison, actual renders, listening and uncertainty.
- **No new or altered recordings, no generated aliases, no automatic split/merge, no inventory edits.** An insufficient result must not auto-schedule anything.

## Acceptance gate

Only a fair, source-isolated comparison with adequate independent donor material may advance to rendered synthesis evaluation. Even an acoustic proxy advantage can never authorize production unit changes by itself.
