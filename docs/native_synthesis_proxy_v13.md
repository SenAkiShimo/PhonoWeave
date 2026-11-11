# PhonoWeave v13 — equal-donor-budget acoustic continuation proxy

**Status:** exploratory, historical, NO rendered singing or validated phone identities. This is an experimental *acoustic continuation-prediction* benchmark, not a perceptual synthesis metric and not an automatic inventory compiler. It intentionally rejects all automatic split/merge decisions.

## What is being tested?

Given one historical 600-ms `source_context_600ms.wav` audio clip already reduced by v9 to three automatically detected log-mel regions (indices 0, 1, 2; **not** verified phonemes), ask whether the first one/two regions can improve choosing acoustically representative **training-WAV** donor excerpts for the third region. No reference OTO times, script labels, IPA, phoneme IDs, or future-region audio enter donor ranking.

This is a **favorable oracle-context test**, not a deployable synthesizer: the held-out sample's earlier audio is provided as a query. A text-only synthesis request would not possess that audio. The benchmark is useful precisely because **failure even under favorable conditioning** is evidence against immediately using the proposed selector in a recording-list compiler. A success would *still* require a proper renderer and independent listening evaluation.

The repository also contains earlier `synthesis-relevance-*` and `controlled-sh-*` studies using stricter phonetic-pair budgets and a simple reference-splice renderer. This v13 module does not replace or claim a stronger endpoint than those experiments. It is aimed specifically at bridging the newer **unlabeled v9–v12 acoustic motif analysis** to a downstream donor-selection question.

## Frozen algorithm and comparisons

- Start with v9 `motif_regions.json` + `raw_spectral_embeddings.npz` and a v9 `summary.json`. Require aligned non-NaN 48D embeddings, valid SHA-256 source identity, no duplicate intervals, and one consistent PCM hash per source. Of 168 archived short clips, 167 contain all three acoustic stage indices; the one incomplete example is rejected without interpolation.
- Reuse the existing deterministic v10 carrier-family split: 14 training, 5 validation, 4 audit script/clip families. All seven subbanks from one carrier remain together; subbanks do **not** count as seven independent singers.
- Normalize each of the three stage embedding distributions using **training only**, with a shrinkage scale floor. Bank-specific donor pools are restricted to training clips, and neither arm may reuse the target physical WAV or held-out carrier group.
- **Pooled (medoid) arm:** choose `k` training clips most central to the training-only third-stage centroid in that subbank.
- **Context arm:** choose `k` clips by nearest-neighbor distance of stage 0, stage 1, or both, using ONLY the heldout recording's preceding regions. The donor's third-stage embeddings are averaged to make a *hypothetical prototype*. The target third-stage embedding is not consulted until the objective loss calculation.
- Donor budget `k∈{1,2,3,4,6}` is identical in both arms; each arm has the identical pool of training WAVs and outputs one **spectral prototype**, not a synthesized audio file. Compare third-stage normalized 48D mean squared errors (lower is better), aggregating **first over bank within carrier and then over carrier group**.
- Select mode and budget using validation groups only. Report how often a source group improves. A 64-trial exploratory negative control reassigns preceding acoustic contexts to different clips within each bank, using a **true derangement**, and lets every null trial choose its own best among all 15 mode/budget hypotheses. Predeclared screening thresholds are ≥5% validation gain, improvement on at least 60% of validation carrier groups, and exploratory max-over-variants null p≤0.05. These thresholds are *not* pre-registered ahead of all historical v9–v12 investigations and do not confer confirmatory significance.
- Only after mode, donor count and null screen are frozen, score the single selected candidate on audit carriers, using exactly the same source groups and observations for both arms. A 1000-fold carrier-group-resampled interval is **descriptive only** (four audit carrier groups, not four independent speakers or a calibrated confidence interval). An audit failure is reported but never used to retune a model.
- Optional `--v12-summary` verifies source archive SHA-256 identity only. Prior v12 proposed bank offsets are **not** imported or automatically applied, because they were not shown to generalize consistently.

## Actual retrospective LynneQing v13 run

Dataset: 503 extracted regions, 168 short audio clips, 167 complete three-stage clips, 23 carrier/script families, 7 pitch/style subbanks, single historical singer. Validation selected `preceding_0`, six training donors per query. Outcomes:

| Proxy endpoint | Same-budget pooled medoid baseline | Preceding-context selection | Relative change |
| --- | ---: | ---: | ---: |
| Development validation, group-averaged 48D MSE | 0.632929 | 0.556132 | **−12.13% error** |
| Retrospective audit, group-averaged 48D MSE | 0.406412 | 0.538174 | **+32.42% error** |

Validation: 4/5 carrier families improved, exploratory 64 derangement max-search null p≈0.01538. Audit: **0/4 carrier families improved**, across 28 bank-clip observations. Audit relative improvement (baseline minus new error, normalized) = **−32.42%**, with descriptive source-family-resampled 2.5%–97.5% interval approximately −63.58% to −18.06%.

**Scientific interpretation:** the validation-only improvement does not transfer to the historical audit. The model may be learning an idiosyncratic relation between the first and third spectral stages. Do not use it to split existing `sh` units, create labels, or select production donors. Even a positive result here would not prove subjective singing quality. v9 context clips were originally selected by an OTO-based upstream study, and audit families have been inspected in previous experimental iterations; this is **retrospective exploratory evidence**, not untouched external validation.

## Reproduce

```bash
python -m phonoweave.native_synthesis_proxy_v13 \
  --v9-results /local/frozen-v9/audio_results \
  --v12-summary /local/frozen-v12/paired_v12/summary.json \
  --output /local/NEW-v13-output
```

Output: `summary.json`, with source provenance, frozen validation choice, 15 tried mode/budget variants and audit comparison; `audit_donor_comparison.csv`, with provenance-linked donor WAV names and spectral errors. Source recordings are never edited or uploaded. Outputs are deliberately refused if the directory already exists. Tests verify target-third-stage isolation from donor ranking, source/partition integrity, equal budgets, audit invariance and misleading/null-control cases.

### Next technical move

Rather than increasing model complexity blindly, develop a **truly available-at-synthesis conditioning signal** (e.g. the existing phoneme/context graph or upstream predicted acoustics) and evaluate the downstream renderer at equal actual recording budgets. Use separate held-out original continuous WAVs and fresh recording sessions; no current result authorizes changing the voicebank inventory.